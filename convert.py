import argparse, os, sys, glob
sys.path.append('stable-diffusion')
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import wandb

delta_ckpt = 'logs/2024-11-11T09-29-20_teddybear-sdv4/checkpoints/delta_epoch=000005.ckpt'

delta_st = torch.load(delta_ckpt)
embed = None
if 'embed' in delta_st['state_dict']:
    embed = delta_st['state_dict']['embed'].reshape(-1, 768)
    del delta_st['state_dict']['embed']
    print(embed.shape)

