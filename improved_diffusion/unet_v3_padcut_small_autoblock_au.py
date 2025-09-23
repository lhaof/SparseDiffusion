from abc import abstractmethod
import math
import os
import pdb
# import profile
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    checkpoint,
)
# from torchprofile import profile_macs
from .sparse_func import GatherBlock, ScatterAvgBlock# , my_group_norm, dilate_mask
import random
# from thop import profile
import matplotlib.pyplot as plt
import cv2
from torchprofile import profile_macs

def gather_block_flops(batch_size, num_active, channels, patch_h, patch_w):
    """
    Calculate FLOPs for GatherBlock.
    Args:
        batch_size (int): Batch size.
        num_active (int): Number of active indices.
        channels (int): Number of channels in the feature map.
        patch_h (int): Height of the patch.
        patch_w (int): Width of the patch.
    Returns:
        int: FLOPs for GatherBlock.
    """
    return batch_size * num_active * channels * patch_h * patch_w

# ==========================================
# Custom FLOP Counter for ScatterAvgBlock
# ==========================================
def scatter_avg_block_flops(batch_size, num_active, channels, patch_h, patch_w):
    """
    Calculate FLOPs for ScatterAvgBlock.
    Args:
        batch_size (int): Batch size.
        num_active (int): Number of active indices.
        channels (int): Number of channels in the feature map.
        patch_h (int): Height of the patch.
        patch_w (int): Width of the patch.
    Returns:
        int: FLOPs for ScatterAvgBlock.
    """
    return batch_size * num_active * channels * patch_h * patch_w

# ==========================================
# FLOP Counter for SparseUNetModel
# ==========================================
def calculate_sparse_unet_flops(model, input_tensor, gather_info, scatter_info):
    """
    Calculate FLOPs for SparseUNetModel, including GatherBlock and ScatterAvgBlock.
    Args:
        model (nn.Module): The SparseUNetModel instance.
        input_tensor (torch.Tensor): Input tensor to the model.
        gather_info (dict): Information about GatherBlock (num_active, patch_h, patch_w).
        scatter_info (dict): Information about ScatterAvgBlock (num_active, patch_h, patch_w).
    Returns:
        float: Total FLOPs (in GFLOPs).
    """
    # Standard FLOPs for the model (excluding Gather and Scatter)
    macs = profile_macs(model, inputs=(input_tensor,))
    flops = macs * 2  # Convert MACs to FLOPs

    # GatherBlock FLOPs
    gather_flops = gather_block_flops(
        batch_size=input_tensor.size(0),
        num_active=gather_info["num_active"],
        channels=gather_info["channels"],
        patch_h=gather_info["patch_h"],
        patch_w=gather_info["patch_w"]
    )

    # ScatterAvgBlock FLOPs
    scatter_flops = scatter_avg_block_flops(
        batch_size=input_tensor.size(0),
        num_active=scatter_info["num_active"],
        channels=scatter_info["channels"],
        patch_h=scatter_info["patch_h"],
        patch_w=scatter_info["patch_w"]
    )

    # Total FLOPs
    total_flops = flops + gather_flops + scatter_flops
    return total_flops / 1e9  # Convert to GFLOPs

# =====================================Segmentation Net=====================================
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class CMUNeXtBlock(nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, k=3):
        super(CMUNeXtBlock, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    # deep wise
                    nn.Conv2d(ch_in, ch_in, kernel_size=(k, k), groups=ch_in, padding=(k // 2, k // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
                nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in * 4),
                nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in)
            ) for i in range(depth)]
        )
        self.up = conv_block(ch_in, ch_out)

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        return x


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class fusion_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(fusion_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(ch_in),
            nn.Conv2d(ch_in, ch_out * 4, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out * 4),
            nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class CMUNeXt(nn.Module):
    def __init__(self, input_channel=1, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super(CMUNeXt, self).__init__()
        # Encoder [8, 16, 32, 64, 128]
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

        self.Upf2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.Upf3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.Upf4 = nn.Upsample(scale_factor=8, mode='bilinear')

        self.outsig = nn.Sigmoid()

    def forward(self, x):
        feats = []
        x1 = self.stem(x)
        feats.append(x1.detach())
        x1 = self.encoder1(x1)
        feats.append(x1.detach())
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        feats.append(self.Upf2(x2.detach()))
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        feats.append(self.Upf3(x3.detach()))
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)# ([4, 64, 32, 32])
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5) # ([4, 64, 32, 32])
        feats.append(self.Upf4(d5.detach()))

        d4 = self.Up4(d5) # ([4, 32, 64, 64])
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)# ([4, 32, 64, 64])

        d3 = self.Up3(d4) # ([4, 16, 128, 128])
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3) # ([4, 16, 128, 128])

        d2 = self.Up2(d3)  # ([4, 8, 256, 256])
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)  # ([4, 8, 256, 256])
        d1 = self.Conv_1x1(d2)
        feats = torch.cat(feats, dim=1)
        
        return self.outsig(d1), feats.detach()


# =====================================Denoising Net=====================================
class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):

        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = torch.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        return torch.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        A counter for the `thop` package to count the operations in an
        attention operation.

        Meant to be used like:

            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )

        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        # We perform two matmuls with the same number of ops.
        # The first computes the weight matrix, the second computes
        # the combination of the value vectors.
        matmul_ops = 2 * b * (num_spatial ** 2) * c
        model.total_ops += torch.DoubleTensor([matmul_ops])


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        resolution=1,
        block_size=32,
        use_sparse=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.resolution = resolution
        self.bgpatch_mul = 1
        self.batch_size = 1

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(emb_channels, self.out_channels,),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, kernel_size=3, stride=1, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, kernel_size=1)

        self.use_sparse = use_sparse


    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if not self.use_sparse:
            return checkpoint(
                self._forward, (x, emb), self.parameters(), self.use_checkpoint
            )
        else:
            return checkpoint(
                self._sp_forward, (x, emb), self.parameters(), self.use_checkpoint
            )

    def _forward(self, x, emb):
        # pdb.set_trace()
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h

    def _sp_forward(self, x, emb):
        B = self.batch_size#x.shape[0]
        # pdb.set_trace()
        h = my_group_norm(x, self.in_layers[0], batch_size=B, bgpatch_mul=self.bgpatch_mul)
        h = self.in_layers[1:](h)
        emb_out = self.emb_layers(emb).type(h.dtype)
        
        h = my_group_norm(h, self.out_layers[0], batch_size=B, bgpatch_mul=self.bgpatch_mul, temb=emb_out)
        h = self.out_layers[1:](h)
        x = self.skip_connection(x)
        return x + h

def my_group_norm(x: torch.Tensor, norm: nn.GroupNorm, batch_size: int, bgpatch_mul: int, temb: torch.Tensor=None):
    """Args:
            x (torch.Tensor): If x is sub batch of images with shape (batch_size*n, channels, patchheight, patchwidth)

        Returns:
            torch.Tensor: shape (batch_size*n, channels, patchheight, patchwidth)
    """
    N, c, h, w = x.shape
    assert N % batch_size == 0
    n = N // batch_size
    num_groups = norm.num_groups  # 32
    group_size = c // num_groups
    x = x.view(batch_size, n, c, h, w).permute(0,2,1,3,4).contiguous()  # [B, n, C, h, w]->[B, C, n, h, w]
    if temb is not None:
        x = x + temb.view(batch_size, c, 1, 1, 1)  # [B, C, n, h, w]
    x = x.view([batch_size, num_groups, group_size, n, h, w]) 

    if bgpatch_mul>1: # 此时需要对[B, C, n, h, w]中的n进行复制扩展
        expanded_slice = x[:, :, :, -1:, :, :].expand(-1, -1, -1, bgpatch_mul-1, -1, -1)  # [B, num_groups, group_size, 1->wanted, h, w]
        # xm = torch.cat([x, expanded_slice], dim=3)  # [B, num_groups, group_size, $$, h, w]
        var, mean = torch.var_mean(torch.cat([x, expanded_slice], dim=3), unbiased=False, dim=[2, 3, 4, 5], keepdim=True)  # [B, num_groups, group_size, bgpatch_mul, h, w]
        var = torch.sqrt(var + norm.eps)  # torch.Size([3, 32, 1, 1, 1, 1])
    else:
        var, mean = torch.var_mean(x, unbiased=False, dim=[2, 3, 4, 5], keepdim=True)
        var = torch.sqrt(var + norm.eps)
    # print("var, mean:",var[0,:5], mean[0,:5])
    x = (x - mean) / var  # shape: (batch_size, groups, channels // groups * n * height * width)
    x = x.view(batch_size, c, n, h, w)

    if norm.affine:
        x = x * norm.weight.view(1, -1, 1, 1, 1) # [B, C, n, h, w]
        x = x + norm.bias.view(1, -1, 1, 1, 1)
    x = x.permute(0, 2, 1, 3, 4).contiguous().view(N, c, h, w)  # [B, C, n, h, w]->[B, n, C, h, w]->[B*n, C, h, w]
    return x


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param number_of_annotators: if specified (as an int), then this model will be
        class-conditional with `number_of_annotators` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        total_num_of_annotators=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        rrdb_blocks=3,
        condition_input_channel=1,
        consensus_training=False,
        soft_label_training=False,
        annotators_training=False,
        no_annotator_training=False,
    ):
        super().__init__()
        
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.total_num_of_annotators = total_num_of_annotators
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.total_num_of_annotators is not None and not no_annotator_training:
            self.label_emb = nn.Embedding(total_num_of_annotators, time_embed_dim)

        self.cmunext = CMUNeXt(input_channel=condition_input_channel, num_classes=1,dims=[8, 16, 32, 64, 128], depths=[1, 1, 1, 1, 1], kernels=[3, 3, 7, 7, 9])
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        self.cache_rrdb_result = None
        self.cache_msk = None
        self.consensus_training=consensus_training
        self.soft_label_training=soft_label_training
        self.annotators_training=annotators_training
        self.no_annotator_training=no_annotator_training
        
        
        
    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)
        self.cmunext.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
        self.cmunext.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    
    
    def forward(self, x, timesteps, number_of_annotators=None, conditioned_image=None, random_p=None, inference=False, first_time_step=100):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.training:
            msk, former_frames_features = self.cmunext(conditioned_image.type(self.inner_dtype))
            
            hs = []
            emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

            if number_of_annotators is not None:
                assert number_of_annotators.shape == (x.shape[0],)
                # assert random_p is not None

                # convert from [1, #annotators] to [0, #annotators - 1]
                if random_p is None:
                    alpha_emb = self.label_emb(number_of_annotators - 1)
                else:
                    alpha_emb = (1 - random_p).unsqueeze(1) * self.label_emb(number_of_annotators - 1) + (random_p.unsqueeze(1)) * self.label_emb(number_of_annotators)
                
                emb = emb + alpha_emb
            # =====================unet==========================
            h = x.type(self.inner_dtype)
            for i, module in enumerate(self.input_blocks):
                h = module(h, emb)
                if i == 0:
                    h = h + former_frames_features.detach()
                hs.append(h)
            h = self.middle_block(h, emb)
            for i, module in enumerate(self.output_blocks):
                cat_in = torch.cat([h, hs.pop()], dim=1)
                h = module(cat_in, emb)
            h = h.type(x.dtype)
            h = self.out(h)
            msk = msk.type(x.dtype)
            # pdb.set_trace()
            
            return h, msk
        else:
            if timesteps[0] == (first_time_step - 1):
                self.cache_msk, self.cache_rrdb_result = self.cmunext(conditioned_image[:1,...].type(self.inner_dtype))
                self.seg_res = self.cache_msk.squeeze()
            former_frames_features = self.cache_rrdb_result
            msk = self.cache_msk
            hs = []
            emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

            
            assert number_of_annotators.shape == (x.shape[0],)
            
            alpha_emb = self.label_emb(number_of_annotators - 1)
            

            emb = emb + alpha_emb
            h = x.type(self.inner_dtype)
            for i, module in enumerate(self.input_blocks):
                h = module(h, emb)
                if i == 0:
                    h = h + former_frames_features
                hs.append(h)
            h = self.middle_block(h, emb)
            for i, module in enumerate(self.output_blocks):
                cat_in = torch.cat([h, hs.pop()], dim=1)
                h = module(cat_in, emb)
            h = h.type(x.dtype)
            
            h = self.out(h)
            return h
        
def calculate_padding(h, w, block_size, stride):
    out_height = math.ceil(h / stride)
    out_width = math.ceil(w / stride)
    
    padding_height = max(0, ((out_height - 1) * stride + block_size - h) // 2)
    padding_width = max(0, ((out_width - 1) * stride + block_size - w) // 2)
    
    return padding_height, padding_width
def get_edges(t, b=1):
    edge = torch.ByteTensor(t.size()).zero_().to(t.device)
    edge[:, b:] = edge[:, b:] | (t[:, b:] != t[:, :-b])
    edge[:, :-b] = edge[:, :-b] | (t[:, b:] != t[:, :-b])
    edge[b:, :] = edge[b:, :] | (t[b:, :] != t[:-b, :])
    edge[:-b, :] = edge[:-b, :] | (t[b:, :] != t[:-b, :])
    return edge.float()


def reduce_mask(mask: torch.Tensor, aspect_ratios=[(1,2),(2,1),(1,3),(3,1)], padding=1, overlap_ratio=0.1, use_bg=True) -> Optional[torch.Tensor]:
    _, _, H, W = mask.shape
    mask_edge = (mask.squeeze() > 0.1).float()
    
    # Find bounding box
    foreground_coords = mask_edge.nonzero(as_tuple=False)
    min_coords = foreground_coords.min(dim=0).values
    max_coords = foreground_coords.max(dim=0).values
    min_h = max(0, int(min_coords[0].item()) - padding)
    min_w = max(0, int(min_coords[1].item()) - padding)
    max_h = min(H - 1, int(max_coords[0].item()) + padding)
    max_w = min(W - 1, int(max_coords[1].item()) + padding)
    bbox_h, bbox_w = max_h - min_h + 1, max_w - min_w + 1

    print("bounding box:", min_h, min_w, max_h, max_w, H, W, padding)
    # Find closest aspect ratio
    bbox_ratio = bbox_h / bbox_w

    closest_ratio = min(aspect_ratios, key=lambda r: abs((r[0]/r[1]) - bbox_ratio))
    rh, rw = closest_ratio
    gamma = torch.tensor([2 * bbox_h / H, 2 * bbox_w / W, 1], device=mask.device).round().max().item()
    nh, nw = gamma * rh, gamma * rw
    patch_h = int(torch.ceil(torch.tensor(bbox_h / nh, device=mask.device)).item())
    patch_w = int(torch.ceil(torch.tensor(bbox_w / nw, device=mask.device)).item())
    stride_h = int(patch_h * (1 - overlap_ratio))
    stride_w = int(patch_w * (1 - overlap_ratio))

    active_indices = []
    h_starts = list(range(min_h, max(min_h+1, max_h - patch_h + 2), stride_h))
    w_starts = list(range(min_w, max(min_w+1, max_w - patch_w + 2), stride_w))

    for start_h in h_starts:
        for start_w in w_starts:
            end_h = min(start_h + patch_h, H)
            end_w = min(start_w + patch_w, W)

            if mask_edge[start_h:end_h, start_w:end_w].sum() > 0:
                active_indices.append((start_h, start_w))

    active_indices_tensor = torch.tensor(active_indices, device=mask.device)

    if use_bg:
        # Sample one random background patch for normalization
        non_active_indices = []
        for i in range(0, H-patch_h+1):# stride_h
            for j in range(0, W-patch_w+1):# stride_w
                if mask_edge[i:i+patch_h, j:j+patch_w].sum() == 0:
                    non_active_indices.append((i, j))
        if len(non_active_indices) > 0:
            random_bg_patch = non_active_indices[torch.randint(len(non_active_indices), (1,), )] #generator=generator
            active_indices_tensor = torch.cat([active_indices_tensor, torch.tensor([random_bg_patch], device=mask.device)])

    bg_num = int(torch.round(torch.tensor((H * W - bbox_h * bbox_w) / (patch_h * patch_w), device=mask.device)).item())
    # print()
    return active_indices_tensor, bg_num, (patch_h, patch_w), (stride_h, stride_w), (bbox_h, bbox_w), (nh, nw)


class SparseUNetModel(UNetModel):
    """
    Based on the original unet model, use sparse convolution on selected area.
    """

    def __init__(self, hnpairs, use_bg, cal_bgnum, overlap_w,cut_padding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        image_size = 256
        print("hnpairs, usebg:",hnpairs,use_bg)
        self.use_bg = use_bg
        self.cal_bgnum = cal_bgnum
        self.hnpairs = hnpairs
        self.active_id = None
        self.min_res = 64
        self.blockpad = 1  # padding on the minimum resolution
        self.overlap_w = overlap_w
        self.cut_padding = cut_padding
        
        channel_mult = self.channel_mult
        num_res_blocks = self.num_res_blocks
        ds = 1
        for level, mult in enumerate(channel_mult):
            if level != len(channel_mult) - 1:
                ds *= 2

        k = 0
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                
                if ds<=4:
                    maskresolution = image_size//ds
                    child = self.output_blocks[k][0]
                    child.resolution = maskresolution
                    child.use_sparse = True
                if level and i == num_res_blocks:
                    ds //= 2
                k += 1
        

    def downsample_mask(self, mask: torch.Tensor, batch_size, min_res: Union[int, Tuple[int, int]] = 64, device="cuda"):
        assert mask.dim() == 2
        H, W = mask.shape
        if isinstance(min_res, int):
            min_h = min_res
            min_w = min_res
        else:
            min_h, min_w = min_res
        h = H
        w = W
        interpolated_mask = F.interpolate(mask.view(1, 1, H, W).float(), (min_h, min_w), mode="bilinear", align_corners=False)
        sparsity_indices, bg_num, blocksize, strides, fgsize, splitsize = reduce_mask(interpolated_mask, padding=self.cut_padding, overlap_ratio=self.overlap_w, use_bg=self.use_bg)

        if not self.use_bg:
            bg_num = 0
        elif not self.cal_bgnum:
            bg_num = 1

        self.bgpatch_mul = bg_num
        self.active_id = [sparsity_indices.cpu().numpy()[0], blocksize, strides]
        self.gather_enc = {}
        blockpad = self.blockpad
        b_h, b_w = blocksize
        while True:
            dsh = h//min_res
            dsw = w//min_res
            active_indices = sparsity_indices.clone()
            active_indices[:, 0] = active_indices[:, 0]*dsh
            active_indices[:, 1] = active_indices[:, 1]*dsw
            active_indices = active_indices.to(torch.int32).contiguous()

            sub_blocksize = [b_h*dsh, b_w*dsw]

            oside = blockpad * dsh
            sub_blocksize = [size + 2 * oside for size in sub_blocksize]

            if h==H:
                self.scatter = ScatterAvgBlock(active_indices.clone(), offset=(0,0), stride=(1,1), device=device)
                self.cuti = oside
            
            active_indices -= oside
            self.gather_enc[(h, w)] = GatherBlock((active_indices).contiguous(), sub_blocksize, device)
            
            h //= 2
            w //= 2
            if h < min_h and w < min_w:
                break
            
        for module in self.output_blocks:
            child = module[0]
            if child.resolution>=min_h:
                child.bgpatch_mul = bg_num
                child.batch_size = batch_size

    

    def forward(self, x, timesteps, number_of_annotators=None, conditioned_image=None, random_p=None, inference=False, first_time_step=100):

        B,C,H,W = x.shape
        if timesteps[0] == (first_time_step - 1):
            cache_msk, self.cache_rrdb_result = self.cmunext(conditioned_image[:1,...].type(self.inner_dtype))
            self.seg_res = cache_msk.squeeze()
            self.cache_msk = (cache_msk*2-1).repeat((x.shape[0],1,1,1)) # torch.Size([1, 1, 256, 256])
            
            self.downsample_mask(cache_msk.squeeze(),batch_size=B,min_res=self.min_res,device=x.device.type)
        
        former_frames_features = self.cache_rrdb_result
        msk = self.cache_msk

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        assert number_of_annotators.shape == (x.shape[0],)      
        alpha_emb = self.label_emb(number_of_annotators - 1)
        emb = emb + alpha_emb
        
        # =====================unet==========================
        
        h = x.type(self.inner_dtype)
        for i, module in enumerate(self.input_blocks):
            h = module(h, emb)
            if i == 0:
                h = h + former_frames_features
            hs.append(h)
        h = self.middle_block(h, emb)

        flag = True
        for i, module in enumerate(self.output_blocks):
            enc_feat = hs.pop()
            h_enc, w_enc = enc_feat.shape[2:]
            if flag and module[0].use_sparse:
                flag = False
                h = self.gather_enc[(self.min_res, self.min_res)](h) 
                
            if enc_feat.shape[-1]>=self.min_res:
                cat_in = torch.cat([h, self.gather_enc[(h_enc, w_enc)](enc_feat)], dim=1)  
            else:
                cat_in = torch.cat([h, enc_feat], dim=1)
            
            h = module(cat_in, emb)
        h = h.type(x.dtype)
        
        h = my_group_norm(h, self.out[0], batch_size=B, bgpatch_mul=self.bgpatch_mul)
        cuti = self.cuti 
        h = self.scatter(self.out[1:](h)[:,:,cuti:-cuti, cuti:-cuti], msk)
        
        return h
        

