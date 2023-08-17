import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

# from dino
class DINOLoss(nn.Module):
    def __init__(self, out_dim, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch, device='cpu'):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(2)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # center ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

# from mocov3 in progress
class MOCOv3Loss(nn.Module):
    def __init__(self, temp):
        """https://arxiv.org/abs/1911.05722"""
        super(MOCOv3Loss, self).__init__()
        self.temp = temp

    def forward(self, student_output, teacher_output, epoch, device='cpu'):
        # only student_output has grad
        # input: cat([x1, x2])
        qs = F.normalize(student_output, dim=1, p=2).chunk(2)
        ks = F.normalize(teacher_output, dim=1, p=2).detach().chunk(2)
        # gather all targets
        ks = [ concat_all_gather(k) for k in ks ]
        total_loss = 0
        # Einstein sum is more intuitive
        for iq, q in enumerate(qs):
            for v in range(len(ks)):
                if v == iq:
                    continue
                logits = torch.einsum('nc,mc->nm', [q, ks[v]]) / self.temp
                N = logits.shape[0]  # batch size per GPU
                labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).to(device, non_blocking=True)

                loss = nn.CrossEntropyLoss()(logits, labels) * (2 * self.temp)
                total_loss += loss.mean()
        return total_loss

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class BYOLLoss(nn.Module):
    """"""
    def __init__(self, **kwargs):
        # train online and use online 
        super(BYOLLoss, self).__init__()
        
    def forward(self,  student_output, teacher_output, epoch=None, device='cpu'):
        # only student_output has grad
        # input: cat([x1, x2])
        qs = F.normalize(student_output, dim=1, p=2).chunk(2)
        ks = F.normalize(teacher_output, dim=1, p=2).detach().chunk(2)

        total_loss = 0
        for iq, q in enumerate(qs):
            for v in range(len(ks)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = 2 - 2 * (q * ks[v]).sum(dim=-1)
                total_loss += loss.mean()
        return total_loss