# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from copy import deepcopy
import os

from easydict import EasyDict

from .va_robotwin_train_cfg import va_robotwin_train_cfg


va_rmbench_train_cfg = EasyDict(__name__="Config: VA rmbench train")
va_rmbench_train_cfg.update(deepcopy(va_robotwin_train_cfg))

va_rmbench_train_cfg.dataset_path = "/path/to/your/rmbench_lingbotva_dataset"
va_rmbench_train_cfg.empty_emb_path = os.path.join(
    va_rmbench_train_cfg.dataset_path,
    "empty_emb.pt",
)
va_rmbench_train_cfg.env_type = "robotwin_tshape"
va_rmbench_train_cfg.obs_cam_keys = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
va_rmbench_train_cfg.cfg_prob = 0.1