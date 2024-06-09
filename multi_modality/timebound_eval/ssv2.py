"""Checks time arrow performance on SSV2 dataset."""
import os
import sys

file_path = os.path.abspath(__file__)
repo_path = os.path.dirname(os.path.dirname(file_path))
sys.path.append(repo_path)

import copy
import datetime
import logging
import os
import time
from os.path import join
from tqdm import tqdm

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from dataset import MetaLoader
from models.umt import UMT
from tasks.pretrain import setup_dataloaders
from tasks.retrieval_utils import evaluation_wrapper
from tasks.shared_utils import setup_model
from utils.basic_utils import MetricLogger, SmoothedValue, setup_seed
from utils.config import Config
from utils.config_utils import setup_main
from utils.distributed import get_rank, is_main_process
from utils.logger import log_dict_to_wandb, setup_wandb

from utils.config import Config
from configs.model import *
from dataset.video_utils import VIDEO_READER_FUNCS
from dataset.utils import pre_text
from tasks.retrieval_utils import (
    extract_text_feats,
    # extract_vision_feats,
)
from models.criterions import get_sim


def num_params(model):
    n = np.sum([p.numel() for p in model.parameters()]) / 1e6
    print(f"Number of parameters in {type(model).__name__}: {np.round(n, 3)}M")


def define_transforms(config):
    vision_enc_name = config.model.vision_encoder.name
    if "swin" in vision_enc_name or "vit" in vision_enc_name:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif "beit" in vision_enc_name:
        mean = (0.5, 0.5, 0.5)  # for all beit model except IN1K finetuning
        std = (0.5, 0.5, 0.5)
    elif "clip" in vision_enc_name:
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
    else:
        raise ValueError
    normalize = transforms.Normalize(mean, std)

    # loaded images and videos are torch.Tensor of torch.uint8 format,
    # ordered as (T, 1 or 3, H, W) where T=1 for image
    type_transform = transforms.Lambda(lambda x: x.float().div(255.0))

    if config.inputs.video_input.random_aug:
        aug_transform = transforms.RandAugment()
    else:
        aug_transform = transforms.Lambda(lambda x: x)

    test_transform = transforms.Compose(
        [
            transforms.Resize(
                (config.inputs.image_res, config.inputs.image_res),
                interpolation=InterpolationMode.BICUBIC,
            ),
            type_transform,
            normalize,
        ]
    )

    inv_transform = transforms.Normalize(
        mean=-(np.array(mean)/np.array(std)),
        std=1./np.array(std),
    )
    return test_transform, inv_transform


def compute_video_text_similarity(video_paths, texts):
    # Compute text embeddings
    with torch.no_grad():
        text_feats, text_atts = extract_text_feats(
            texts, max_txt_l, tokenizer, model, device
        )  # (bsz, Lt, d), (bsz, Lt)

    # Compute video embeddings
    pooled_image_feats = []
    for data_path in video_paths:
        # Load frames of the video
        frames, frame_indices, video_duration = video_reader(
            data_path, num_frames, sample_type, 
            max_num_frames=max_num_frames, client=None,
            trimmed30=False,
        )
        # shared aug for video frames
        frames = fwd_transform(frames)

        image = frames.unsqueeze(0)
        image = image.to(device, non_blocking=True)
        with torch.no_grad():
            image_feat, pooled_image_feat = model.encode_vision(image, test=True)

        if config.evaluation.eval_frame_ensemble == "concat":  # default
            if len(image_feat.shape) == 4:
                from einops import rearrange
                image_feat = rearrange(image_feat, "b t l c -> b (t l) c").contiguous()
            image_feat = image_feat.unsqueeze(1)  # (bsz, 1, #frm*L, d)
        else:
            assert config.video_input.num_frames == 1, "only support single-frame"
            assert config.evaluation.eval_frame_ensemble in ["mean", "max", "lse"]
        # (bsz, 1, #frm*Li, d) or (bsz, #frm, Li, d), (bsz, #frm, d)
        pooled_image_feats.append(pooled_image_feat)
    pooled_image_feats = torch.cat(pooled_image_feats, dim=0)

    # Compute video-text similarity
    _pooled_image_feats = (
        pooled_image_feats.to(device, non_blocking=True)
        if config.evaluation.eval_offload
        else pooled_image_feats
    )
    i2t_scores, t2i_scores = get_sim(
        model.vision_proj(_pooled_image_feats), model.text_proj(text_feats[:, 0])
    )
    return i2t_scores, t2i_scores


import warnings
warnings.filterwarnings("ignore")
from glob import glob


def get_video_path(video_dir, video_id, ext="webm"):
    paths = glob(os.path.join(video_dir, f"*/{video_id}.{ext}"))
    assert len(paths) == 1
    return paths[0]


def text_correct(sim):
    """
    Given a 2x2 similarity matrix, computes text score.

    Based on WinoGround's evaluation code.
    """
    return sim[0, 0] > sim[0, 1] and sim[1, 1] > sim[1, 0]


def video_correct(sim):
    """
    Given a 2x2 similarity matrix, computes video score.

    Based on WinoGround's evaluation code.
    """
    return sim[0, 0] > sim[1, 0] and sim[1, 1] > sim[0, 1]


def group_correct(sim):
    """
    Given a 2x2 similarity matrix, computes group score.

    Based on WinoGround's evaluation code.
    """
    return text_correct(sim) and video_correct(sim)


if __name__ == "__main__":

    # Load config
    config_name = "l16"
    config = Config.from_file(
        filepath=os.path.join(repo_path, f"exp/zero_shot/ret_msrvtt/{config_name}.py"),
    )

    # Setup path to pre-trained checkpoint
    ckpt_root = "/work/piyush/pretrained_checkpoints/LargeModels/UnmaskedTeachers/"
    ckpt_name = "l16_25m"
    stage_1_ckpt_name = "l16_ptk710_f8_res224.pth"
    config.pretrained_path = os.path.join(ckpt_root, f"{ckpt_name}.pth")
    config.model.vision_encoder.pretrained = os.path.join(
        ckpt_root, stage_1_ckpt_name,
    )

    # Define the text encoder
    text_model_name = "bert"
    text_model_name = "bert_large" # For large model
    config.model.text_encoder = TextEncoders[text_model_name]

    # Define number of frames
    config.model.vision_encoder.num_frames = config.num_frames

    # Misc
    config.distributed = False
    config.scheduler.num_warmup_steps = 1
    config.scheduler.num_training_steps = 1
    config.auto_resume = False

    config.inputs.video_input.num_frames_test = config.num_frames_test
    num_frames = config.inputs.video_input.num_frames_test
    sample_type = config.inputs.video_input.sample_type_test
    video_reader_type = config.inputs.video_input.get("video_reader_type", "decord")
    media_type = "video"
    config.inputs.max_txt_l = {k: config.max_txt_l for k in config.inputs.max_txt_l}

    config.model["text_encoder"]["config"] = os.path.join(
        repo_path, config.model["text_encoder"]["config"],
    )
    config["TextEncoders"]["bert"]["config"] = os.path.join(
        repo_path, config["TextEncoders"]["bert"]["config"],
    )


    # Setup
    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)
    cudnn.benchmark = True


    # Load model
    print("[:::] Loading model")
    model_cls = eval(config.model.get('model_cls', 'UMT'))
    (
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        tokenizer,
        start_epoch,
        global_step,
    ) = setup_model(
        config,
        model_cls=model_cls,
        has_decoder=False,
        pretrain=False,
        # find_unused_parameters=True,
        find_unused_parameters=False,
    )
    model.to(device)
    model.eval()
    num_params(model)
    print("[:::] Done loading model")


    video_reader = VIDEO_READER_FUNCS[video_reader_type]
    max_num_frames = -1
    fwd_transform, inv_transform = define_transforms(config)
    max_txt_l = config.inputs.max_txt_l
    max_txt_l = max_txt_l[media_type]


    # Debug
    sample_paths = [
        os.path.join(
            repo_path,
            "../../TimeBound.v1/sample_data/folding_paper.mp4",
        )
    ]
    sample_texts = ["someone is folding a paper"]

    with torch.no_grad():
        i2t_scores, t2i_scores = compute_video_text_similarity(
            sample_paths, sample_texts,
        )
    print(i2t_scores, t2i_scores)


    # Run on the entire dataset
    print("[:::] Running on the entire dataset")

    # Load data
    csv_path = "/scratch/shared/nfs2/piyush/datasets/SSv2/metadata/time_antonyms-validation.csv"
    df = pd.read_csv(csv_path)

    data_dir = "/scratch/shared/beegfs/shared-datasets/SomethingSomething-V2/"
    video_dir = os.path.join(data_dir, "videos")

    iterator = tqdm(df.iterrows(), total=len(df))
    text_corrects = []
    video_corrects = []
    group_corrects = []
    failed = []
    for i, row in iterator:
        row = row.to_dict()
        video_path_x = get_video_path(video_dir, row["id_x"])
        video_path_y = get_video_path(video_dir, row["id_y"])
        label_x = row["label_x"]
        label_y = row["label_y"]

        video_paths = [video_path_x, video_path_y]
        texts = [label_x, label_y]

        with torch.no_grad():
            _, sim = compute_video_text_similarity(video_paths, texts)
        sim = sim.cpu().numpy()
        text_corrects.append(text_correct(sim))
        video_corrects.append(video_correct(sim))
        group_corrects.append(group_correct(sim))

        # if i == 10:
        #     break

    # Compute final metrics
    text_corrects = np.array(text_corrects)
    video_corrects = np.array(video_corrects)
    group_corrects = np.array(group_corrects)

    print("Text score:", text_corrects.mean())
    print("Video score:", video_corrects.mean())
    print("Group score:", group_corrects.mean())
