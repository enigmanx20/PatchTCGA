import easydict

# original preset
_LR_PRESETS = {40: 0.45, 100: 0.45, 300: 0.3, 1000: 0.2}
_WD_PRESETS = {40: 1e-6, 100: 1e-6, 300: 1e-6, 1000: 1.5e-6}
_EMA_PRESETS = {40: 0.97, 100: 0.99, 300: 0.99, 1000: 0.996}

byol_config = {  
            "image_size" :     224,
            "epochs"     :     40,
            "checkpoint" :    None, 
            "batch_size" :      64,  
            "base_batch_size" : 512, 
            "num_accum"  :       1,
            "lr"         :     0.45,  
            "cold_epochs":        0,  
            "warmup_epochs":      10,  
            "wd"         :     1.0e-6,
            "base_target_ema":    0.99,
}