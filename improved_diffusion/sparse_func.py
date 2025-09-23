
import warnings
import importlib
import os
from typing import Dict, Optional, Tuple, Union, List
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
runsparse_dict = {}
devices = ["cuda", "cpu"]
# for device in devices:
device = "cuda"
name = "sp_avg.%s" % device
print(name)
module = importlib.import_module(name)
runtimegather = getattr(module, "gather")
runtimescatter = getattr(module, "scatter") #  runtimegather = getattr(importlib.import_module("sige.cpu"),"gather")
runtimescatteravg = getattr(module, "scatter_avg")
# runtimescattermap = getattr(module, "get_scatter_map"), runtimescattergather, runtimescattermap
runsparse_dict[device] = [runtimegather, runtimescatter, runtimescatteravg]



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

    if bgpatch_mul!=1: # 此时需要对[B, C, n, h, w]中的n进行复制扩展
        expanded_slice = x[:, :, :, -1:, :, :].expand(-1, -1, -1, bgpatch_mul-1, -1, -1)  # [B, num_groups, group_size, 1->wanted, h, w]
        # xm = torch.cat([x, expanded_slice], dim=3)  # [B, num_groups, group_size, $$, h, w]
        var, mean = torch.var_mean(torch.cat([x, expanded_slice], dim=3), unbiased=False, dim=[2, 3, 4, 5], keepdim=True)  # [B, num_groups, group_size, bgpatch_mul, h, w]
        var = torch.sqrt(var + norm.eps)  # torch.Size([3, 32, 1, 1, 1, 1])
    else:
        var, mean = torch.var_mean(x, unbiased=False, dim=[2, 3, 4, 5], keepdim=True)
        var = torch.sqrt(var + norm.eps)
    
    x = (x - mean) / var  # shape: (batch_size, groups, channels // groups * n * height * width)
    x = x.view(batch_size, c, n, h, w)

    if norm.affine:
        x = x * norm.weight.view(1, -1, 1, 1, 1) # [B, C, n, h, w]
        x = x + norm.bias.view(1, -1, 1, 1, 1)
    x = x.permute(0, 2, 1, 3, 4).contiguous().view(N, c, h, w)  # [B, C, n, h, w]->[B, n, C, h, w]->[B*n, C, h, w]
    return x


def activation(x: torch.Tensor, activation_name: str):
    if activation_name == "relu":
        return torch.relu(x)
    elif activation_name == "sigmoid":
        return torch.sigmoid(x)
    elif activation_name == "tanh":
        return torch.tanh(x)
    elif activation_name == "swish":
        return x * torch.sigmoid(x)
    elif activation_name == "identity":
        return x
    else:
        raise ValueError("Unknown activation: [%s]!!!" % activation_name)


class GatherBlock(nn.Module):
    def __init__(
        self,
        active_indices,
        block_size: Union[int, Tuple[int, int]],
        device: str,
    ):
        super(GatherBlock, self).__init__()
        if isinstance(block_size, int):
            block_size = (block_size, block_size)
        self.block_size = block_size
        self.active_indices = active_indices
        global runsparse_dict
        self.runfunc = runsparse_dict[device][0]
        self.device=device

    def forward(self, x):
        # self.check_dtype(x, scale, shift)
        # self.check_dim(x, scale, shift)
        # b, c, h, w = x.shape # torch.Size([1, 128, 256, 256])
        output = self.runfunc(
            x.contiguous(),
            self.block_size[0],
            self.block_size[1], # block_size[1]
            self.active_indices.contiguous(),
            None,# scale.contiguous()
            None,# shift.contiguous()
            "identity",# activation_name
            False, # activation_first
        )
        return output


class ScatterBlock(nn.Module):
    def __init__(self, 
        active_indices,
        offset=(0,0),
        stride=(1,1),
        device="cuda",
    ):
        super(ScatterBlock, self).__init__()
        self.active_indices = active_indices
        self.offset = offset
        self.stride = stride
        global runsparse_dict
        self.runfunc = runsparse_dict[device][1]
        self.device=device

    def forward(self, x, original_output):
        offset = self.offset
        stride = self.stride
        # runtime = self.scatterfuncs[device]
        output = self.runfunc(
            x.contiguous(),
            original_output.contiguous(),
            offset[0],
            offset[1],
            stride[0],
            stride[1],
            self.active_indices.contiguous(),
            None,# if residual is None else residual.contiguous(),
        )
            
        return output

class ScatterAvgBlock(nn.Module):
    def __init__(self, 
        active_indices,
        offset=(0,0),
        stride=(1,1),
        device="cuda",
    ):
        super(ScatterAvgBlock, self).__init__()
        self.active_indices = active_indices
        self.offset = offset
        self.stride = stride
        global runsparse_dict
        self.runfunc = runsparse_dict[device][2]
        self.device=device

    def forward(self, x, original_output):
        offset = self.offset
        stride = self.stride
        # runtime = self.scatterfuncs[device]
        output = self.runfunc(
            x.contiguous(),
            original_output.contiguous(),
            offset[0],
            offset[1],
            stride[0],
            stride[1],
            self.active_indices.contiguous()
        )
        self.input_shape = x.shape
        self.output_shape = output.shape
        return output
# class ModuleWrapper:
#     def __init__(self, module):
#         self.module = module

# class ScatterGather(SIGEModule):
#     def __init__(self, gather: GatherBlock, activation_name: str = "identity", activation_first: bool = False, device="cuda",):
#         super(ScatterGather, self).__init__()
#         self.gather = ModuleWrapper(gather)
#         self.activation_name = activation_name
#         self.activation_first = activation_first

#         self.load_runtime("scatter_gather")
        
#         self.scatter_map = None

#         global runsparse_dict
#         self.runfunc = runsparse_dict[device][2]
#         self.get_scatter_map_runtime = runsparse_dict[device][3]


#         mask = self.gather.module.mask
#         h, w = mask.shape
#         block_size = self.gather.module.block_size
#         kernel_size = self.gather.module.kernel_size
#         offset = self.gather.module.offset
#         stride = self.gather.module.model_stride
#         active_indices = self.gather.module.active_indices
#         device = active_indices.device.type
#         runtime = self.get_scatter_map_runtime[device]
#         scatter_map = runtime(
#             h,
#             w,
#             block_size[0],
#             block_size[1],
#             kernel_size[0],
#             kernel_size[1],
#             offset[0],
#             offset[1],
#             stride[0],
#             stride[1],
#             active_indices,
#         )

#     def forward(
#         self, x: torch.Tensor, scale: Optional[torch.Tensor] = None, shift: Optional[torch.Tensor] = None
#     ) -> torch.Tensor:
#         self.check_dtype(x, scale, shift)
#         self.check_dim(x, scale, shift)
#         b, c, h, w = x.shape
#         active_indices = self.active_indices
#         block_size = self.gather.module.block_size
        
#         output = self.runfunc(
#             x.contiguous(),
#             original_outputs.contiguous(),
#             block_size[0],
#             block_size[1],
#             active_indices.contiguous(),
#             self.scatter_map.contiguous(),
#             None if scale is None else scale.contiguous(),
#             None if shift is None else shift.contiguous(),
#             self.activation_name,
#             self.activation_first,
#         )
#         return output
def dilate_mask(
    mask: Union[torch.Tensor, np.ndarray], dilation: Union[int, Tuple[int, int]]  # [C, H, W] or [H, W]
) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    if dilation[0] <= 0 and dilation[1] <= 0:
        return mask

    if isinstance(mask, torch.Tensor):
        ret = mask.clone()
    else:
        assert isinstance(mask, np.ndarray)
        ret = mask.copy()

    if len(ret.shape) == 2:
        for i in range(1, dilation[0] + 1):
            ret[:-i] |= mask[i:]
            ret[i:] |= mask[:-i]
        for i in range(1, dilation[1] + 1):
            ret[:, :-i] |= mask[:, i:]
            ret[:, i:] |= mask[:, :-i]
    elif len(ret.shape) == 3:
        for i in range(1, dilation + 1):
            ret[:, :-i] |= mask[:, i:]
            ret[:, i:] |= mask[:, :-i]
        for i in range(1, dilation[1] + 1):
            ret[:, :, :-i] |= mask[:, :, i:]
            ret[:, :, i:] |= mask[:, :, :-i]
    else:
        raise NotImplementedError("Unknown mask dimension [%d]!!!" % mask.dim())
    return ret