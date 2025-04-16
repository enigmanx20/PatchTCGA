[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
# PatchTCGA
Paper: [Large-scale pretraining on pathological images for fine-tuning of small pathological benchmarks](https://arxiv.org/abs/2303.15693)

![Overview](overview.png)


# DATASETS
We engineered three patch-based one large and two small datasets. They are designed to conduct large-scale training and downstream benchmarking, respectively. They have the same microns per pixel (MPP) of 0.39. MD5 checksums are available in the download links. Please refer the original paper for details.


### Patch TCGA in 200μm (PTCGA200)

Donwload Link: https://drive.google.com/drive/folders/18CmL-WLyppK1Rk29CgV7ib5MACFzg5ei?usp=drive_link

License: [NIH Genomic Data Sharing (GDS) Policy](https://datascience.cancer.gov/data-sharing/genomic-data-sharing/about-the-genomic-data-sharing-policy)

Use the snippet below to make the original archive file from divided files.
```
$ cat PTCGA200_p_* > PTCGA200.tar.gz
```
To reproduce the same training, validation, and testing split in the original paper, download and load the `3fold_dict_idx_filenames.pickle` file using dataset_utils.py.


### Patch Camelyon in 200μm (PCam200): 

Donwload Link: https://drive.google.com/drive/folders/1Oh7onawKsDW5ScamVO5ByXFgqdYJ39sK?usp=drive_link

License: CC0 [![License: CC0-1.0](https://licensebuttons.net/l/zero/1.0/80x15.png)](http://creativecommons.org/publicdomain/zero/1.0/)

### Segmentation PANDA in 200μm (SegPANDA200): 

Donwload Link: https://drive.google.com/drive/folders/1zg_C37B_1HR6miRFuTwPKmueaJzvO-GD?usp=drive_link

License: CC BY-SA-NC 4.0 [![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

# Getting Started

### Requirements
- pytorch >=1.8.1 
- torchvision
- kornia
- Pillow 8.2.0
- numpy 
- tqdm 3.60.0
- h5py

### Train
Download datasets and modify the config.py file. Then, create `runs` folder and run the following command.The training script is designed for distributed training. If you want to train on multiple nodes, provide host name lists and master node address in the config.py file and run the script in each node.

```
python train.py
```
To train BYOL, change the config ` 'self_superversed' :    'byol' `. 

Self-supervised learning was performed using the repo  below except for BYOL.
- [SimCLR (pytorch repro)](https://github.com/AndrewAtanov/simclr-pytorch)
- [MoCov2](https://github.com/facebookresearch/moco)
- [DINO](https://github.com/facebookresearch/dino)

# Pretrained Models
Pretrained models on PTCGA200.
- [SimCLRv1.5](https://drive.google.com/file/d/1tC_QwWTqWO9I0hzUArAc8yWimJsY33hK/view?usp=drive_link): ResNet50 model checkpoint trained on PTCGA200 for 40 epochs with SimCLR.
- [MoCov2](https://drive.google.com/file/d/1YOc0t0rd5aJriZPyiUGovFQbTVJZPel8/view?usp=drive_link): ResNet50 model checkpoint trained on PTCGA200 for 40 epochs with MoCov2.
- [BYOL](https://drive.google.com/file/d/15OEY995X2yRdlhKTYYlP5WQNZKLIwl-J/view?usp=drive_link): ResNet50 model checkpoint trained on PTCGA200 for 40 epochs with BYOL.
- [DINOv1](https://drive.google.com/file/d/15RbQ0QURpr_3g-x0-FG7kA8oeu6nocSr/view?usp=drive_link): ViT-S/16 model checkpoint trained on PTCGA200 for 40 epochs with DINOv1.
- [MAE](https://drive.google.com/file/d/1T13zEkzQbcSyzEbJN0OOhCZqlAEqb-yR/view?usp=drive_link): ViT-L/16 model checkpoint trained on PTCGA200 for 40 epochs with masked autoencoder (MAE).

Trained models on SegPANDA200.
- [DeepLabv3](https://drive.google.com/file/d/1bkeTG05OaU_uTfGddsJxRcVZbLy-mbCs/view?usp=drive_link): DeepLabv3 model checkpoint trained on SegPANDA200 (Test mIoU 69.61).
- [MAE-ViT-L/16](https://drive.google.com/file/d/1DWpqWCaOWsdXXUASIwoi8jqypjvdoFAy/view?usp=drive_link): ViT-L/16 model checkpoint trained on SegPANDA200 from above checkpoint (Test mIoU 70.7).

# Citation
Provisional

```
@CoRR{PatchTCGA,
  title={Large-scale pretraining on pathological images for fine-tuning of small pathological benchmarks},
  author={Masataka Kawai, Nriaki Ota, Shinsuke Yamaoka},
  booktitle={},
  year={2023}
}
```

# Acknowledgement
We thank the authors of the original datasets for their efforts. 

- [Camelyon16](https://doi.org/10.1001/jama.2017.14585.)
- [PANDA]( https://doi.org/10.1038/s41591-021-01620-2.)

We also thank the authors of the following repositories for their contributions and references.

- [Original SimCLR](https://github.com/google-research/simclr)
- [PyTorch SimCLR](https://github.com/AndrewAtanov/simclr-pytorch)
- [Original BYOL](https://github.com/deepmind/deepmind-research/tree/master/byol)
- [PyTorch BYOL](https://github.com/lucidrains/byol-pytorch)
- [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)

This work is based on results obtained from a project, JPNP20006, commissioned by the New Energy and Industrial Technology Development Organization (NEDO). 

# Citation
If you use this work, datasets, and models, please cite the following paper:
```bibtex
@InProceedings{10.1007/978-3-031-44917-8_25,
  author    = {Kawai, Masakata and Ota, Noriaki and Yamaoka, Shinsuke},
  editor    = {Xue, Zhiyun and Antani, Sameer and Zamzmi, Ghada and Yang, Feng and Rajaraman, Sivaramakrishnan and Huang, Sharon Xiaolei and Linguraru, Marius George and Liang, Zhaohui},
  title     = {Large-Scale Pretraining on Pathological Images for Fine-Tuning of Small Pathological Benchmarks},
  booktitle = {Medical Image Learning with Limited and Noisy Data},
  year      = {2023},
  publisher = {Springer Nature Switzerland},
  address   = {Cham},
  pages     = {257--267},
  isbn      = {978-3-031-44917-8}
}
```

