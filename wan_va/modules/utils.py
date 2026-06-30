# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import torch
from diffusers import AutoencoderKLWan
from transformers import (
    T5TokenizerFast,
    UMT5EncoderModel,
)

from .model import WanTransformer3DModel


def load_vae(
    vae_path,
    torch_dtype,
    torch_device,
):
    vae = AutoencoderKLWan.from_pretrained(
        vae_path,
        torch_dtype=torch_dtype,
    )
    return vae.to(torch_device)


def load_text_encoder(
    text_encoder_path,
    torch_dtype,
    torch_device,
):
    text_encoder = UMT5EncoderModel.from_pretrained(
        text_encoder_path,
        torch_dtype=torch_dtype,
    )
    return text_encoder.to(torch_device)


def load_tokenizer(tokenizer_path, ):
    tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path, )
    return tokenizer


def load_transformer(
    transformer_path,
    torch_dtype,
    torch_device,
    **kwargs
):
    model = WanTransformer3DModel.from_pretrained(
        transformer_path,
        torch_dtype=torch_dtype,
        **kwargs
    )
    return model.to(torch_device)


def patchify(x, patch_size):
    if patch_size is None or patch_size == 1:
        return x
    batch_size, channels, frames, height, width = x.shape
    x = x.view(batch_size, channels, frames, height // patch_size, patch_size,
               width // patch_size, patch_size)
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    x = x.view(batch_size, channels * patch_size * patch_size, frames,
               height // patch_size, width // patch_size)
    return x


class WanVAEStreamingWrapper:

    def __init__(self, vae_model):
        self.vae = vae_model
        self.encoder = vae_model.encoder
        self.quant_conv = vae_model.quant_conv

        if hasattr(self.vae, "_cached_conv_counts"):
            self.enc_conv_num = self.vae._cached_conv_counts["encoder"]
        else:
            count = 0
            for m in self.encoder.modules():
                if m.__class__.__name__ == "WanCausalConv3d":
                    count += 1
            self.enc_conv_num = count

        self.clear_cache()

    def clear_cache(self):
        self.feat_cache = [None] * self.enc_conv_num

    def encode_chunk(self, x_chunk):
        if hasattr(self.vae.config,
                   "patch_size") and self.vae.config.patch_size is not None:
            x_chunk = patchify(x_chunk, self.vae.config.patch_size)
        feat_idx = [0]
        out = self.encoder(x_chunk,
                           feat_cache=self.feat_cache,
                           feat_idx=feat_idx)
        enc = self.quant_conv(out)
        return enc

    def encode_temporal_chunks(self, x, chunk_size=4):
        """Encode a full video using DiffSynth-style temporal chunks.

        Wan VAE's cached encoder path expects the first call to contain only
        the first frame, then subsequent calls to contain 4-frame chunks. A
        single cached call with the full clip can desynchronize residual
        shortcut temporal downsampling inside diffusers' Wan encoder.
        """
        if x.shape[2] <= 0:
            raise ValueError(f"Expected at least one video frame, got shape {tuple(x.shape)}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

        self.clear_cache()
        if hasattr(self.vae.config,
                   "patch_size") and self.vae.config.patch_size is not None:
            x = patchify(x, self.vae.config.patch_size)

        num_chunks = 1 + (x.shape[2] - 1) // chunk_size
        chunks = []
        for chunk_idx in range(num_chunks):
            feat_idx = [0]
            if chunk_idx == 0:
                x_chunk = x[:, :, :1, :, :]
            else:
                start = 1 + chunk_size * (chunk_idx - 1)
                end = 1 + chunk_size * chunk_idx
                x_chunk = x[:, :, start:end, :, :]
            if x_chunk.shape[2] == 0:
                continue

            chunks.append(
                self.encoder(
                    x_chunk,
                    feat_cache=self.feat_cache,
                    feat_idx=feat_idx,
                )
            )

        out = torch.cat(chunks, dim=2)
        enc = self.quant_conv(out)
        return enc
