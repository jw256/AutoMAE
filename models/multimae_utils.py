# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on timm, DeiT, DINO, MoCo-v3, BEiT, MAE-priv and MAE code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/facebookresearch/moco-v3
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/BUPT-PRIV/MAE-priv
# https://github.com/facebookresearch/mae
# --------------------------------------------------------

import math
import warnings

import torch
import torch.nn as nn
from einops import rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def build_2d_sincos_posemb(h, w, embed_dim=1024, temperature=10000.):
    """Sine-cosine positional embeddings from MoCo-v3

    Source: https://github.com/facebookresearch/moco-v3/blob/main/vits.py
    """
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
    assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature ** omega)
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]
    pos_emb = rearrange(pos_emb, 'b (h w) d -> b d h w', h=h, w=w, d=embed_dim)
    return pos_emb


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        B, N, C = x.shape
        _, M, _ = context.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.query_norm = norm_layer(dim)
        self.context_norm = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, context):
        x = x + self.drop_path(self.self_attn(self.norm1(x)))
        x = x + self.drop_path(self.cross_attn(self.query_norm(x), self.context_norm(context)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



# EASY_0907, rgb tokenizer and decoder
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
from functools import partial
from einops import rearrange, repeat

class PatchedInputAdapter(nn.Module):
    """Adapter for spatial inputs, like images or feature maps.
    Creates tokens from patches over the image.

    :param num_channels: Number of input channels of the image/feature map
    :param stride_level: Stride level compared to the full-sized image.
        E.g. 4 for 1/4th the size of the image.
    :param patch_size_full: Int or tuple of the patch size over the full image size.
        Patch size for smaller inputs will be computed accordingly.
    :param dim_tokens: Dimension of output tokens. Can be set using init method.
    :param sincos_pos_emb: Set to True (default) to use fixed 2D sin-cos positional embeddings
    :param learnable_pos_emb: Set to True to learn positional embeddings instead
    :param image_size: Default image size. Used to initialize size of positional embeddings.
    """
    def __init__(self,
                 num_channels: int,
                 stride_level: int,
                 patch_size_full: Union[int, Tuple[int,int]],
                 dim_tokens: Optional[int] = None,
                 sincos_pos_emb: bool = True,
                 learnable_pos_emb: bool = False,
                 image_size: Union[int, Tuple[int]] = 224):

        super().__init__()
        self.num_channels = num_channels
        self.stride_level = stride_level
        self.patch_size_full = pair(patch_size_full)
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.learnable_pos_emb = learnable_pos_emb
        self.image_size = pair(image_size)
        self.num_patches = (self.image_size[0] // patch_size_full) * (self.image_size[1] // patch_size_full)

        # Actual patch height and width, taking into account stride of input
        self.P_H = max(1, self.patch_size_full[0] // stride_level)
        self.P_W = max(1, self.patch_size_full[1] // stride_level)

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768):
        """
        Initialize parts of encoder that are dependent on dimension of tokens.
        Should be called when setting up MultiMAE.

        :param dim_tokens: Dimension of tokens
        """
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        h_posemb = self.image_size[0] // (self.stride_level * self.P_H)
        w_posemb = self.image_size[1] // (self.stride_level * self.P_W)
        if self.sincos_pos_emb:
            self.pos_emb = build_2d_sincos_posemb(h=h_posemb, w=w_posemb, embed_dim=self.dim_tokens)
            self.pos_emb = nn.Parameter(self.pos_emb, requires_grad=self.learnable_pos_emb)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.dim_tokens, h_posemb, w_posemb))
            trunc_normal_(self.pos_emb, std=0.02)

        # Image -> tokens projection
        self.proj = nn.Conv2d(
            in_channels=self.num_channels, out_channels=self.dim_tokens,
            kernel_size=(self.P_H, self.P_W), stride=(self.P_H, self.P_W)
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_emb'}

    def forward(self, x):
        """
        Forward pass through input adapter, transforming image to sequence of tokens.
        Adds task and positional encodings.

        :param x: Input image tensor
        """
        B, C, H, W = x.shape
        assert self.dim_tokens is not None, 'Need to call init(dim_tokens) function first'
        assert (H % self.P_H == 0) and (W % self.P_W == 0), f'Image sizes {H}x{W} must be divisible by patch sizes {self.P_H}x{self.P_W}'
        N_H, N_W = H // self.P_H, W // self.P_W # Number of patches in height and width

        # Create patches [B, C, H, W] -> [B, (H*W), C]
        x_patch = rearrange(self.proj(x), 'b d nh nw -> b (nh nw) d')

        # Create positional embedding
        x_pos_emb = F.interpolate(self.pos_emb, size=(N_H, N_W), mode='bicubic', align_corners=False)
        x_pos_emb = rearrange(x_pos_emb, 'b d nh nw -> b (nh nw) d')

        # Add patches and positional embeddings
        x = x_patch + x_pos_emb 

        return x
    
    

class SpatialOutputAdapter(nn.Module):
    """Cross-attention adapter for spatial outputs, like images or feature maps.

    :param num_channels: Number of input channels of the image/feature map
    :param stride_level: Stride level compared to the full-sized image.
        E.g. 4 for 1/4th the size of the image.
    :param patch_size_full: Int or tuple of the patch size over the full image size.
        Patch size for smaller inputs will be computed accordingly.
    :param dim_tokens_enc: Dimension of tokens coming from encoder. Can be set using init method.
    :param dim_tokens: Dimension of decoder tokens
    :param depth: Number of additional (full self-attention) transformer layers after initial cross attention and MLP
    :param learnable_pos_emb: Set to True to learn positional embeddings instead
    :param image_size: Default image size. Used to initialize size of positional embeddings.
    :param mlp_ratio: MLP hidden dim ratio
    :param num_heads: Number of attention heads
    :param qkv_bias: Set to True to enable bias
    :param drop_rate: Probability of dropping attention layer outputs
    :param attn_drop_rate: Probability of dropping attention matrix elements
    :param drop_path_rate: DropPath drop rate
    :param norm_layer: Type of normalization layer
    :param use_task_queries: When set to True, adds task specific tokens from encoder (if available)
        to the corresponding query entries
    :param task: Task for which encoder tokens are added to the queries of the decoder (e.g. RGB if decoder is used for RGB)
    :param context_tasks: Tasks / modalities from the encoder. Used to create learned embeddings for each task.
    :param use_xattn: When set to True, attend to the tokens from the encoder through a cross-attention layer
    """

    def __init__(self,
                 num_channels: int,
                 stride_level: int,
                 patch_size_full: Union[int, Tuple[int, int]],
                 dim_tokens_enc: Optional[int] = None,
                 dim_tokens: int = 256,
                 depth: int = 0,
                 learnable_pos_emb: int = False,
                 image_size: Union[int, Tuple[int]] = 224,
                 mlp_ratio: int = 4.0,
                 num_heads: int = 8,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
                 use_task_queries: bool = True,
                 task: Optional[str] = None,
                 context_tasks: Optional[list] = None,
                 use_xattn: bool = True
                 ):
        super().__init__()
        self.num_channels = num_channels
        self.stride_level = stride_level
        self.patch_size_full = pair(patch_size_full)
        self.dim_tokens_enc = dim_tokens_enc
        self.dim_tokens = dim_tokens
        self.learnable_pos_emb = learnable_pos_emb
        self.image_size = pair(image_size)
        self.use_task_queries = use_task_queries
        self.task = task
        self.use_xattn = use_xattn

        # Actual patch height and width, taking into account stride of input
        self.P_H = max(1, self.patch_size_full[0] // stride_level)
        self.P_W = max(1, self.patch_size_full[1] // stride_level)

        if context_tasks is not None:
            self.task_embeddings = nn.ParameterDict(
                {task: nn.Parameter(torch.zeros(1, 1, self.dim_tokens)) for task in context_tasks})
            for embedding in self.task_embeddings.values():
                trunc_normal_(embedding, std=0.02)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))

        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        h_posemb = self.image_size[0] // (self.stride_level * self.P_H)
        w_posemb = self.image_size[1] // (self.stride_level * self.P_W)
        if not self.learnable_pos_emb:
            self.pos_emb = build_2d_sincos_posemb(h=h_posemb, w=w_posemb, embed_dim=self.dim_tokens)
            self.pos_emb = nn.Parameter(self.pos_emb, requires_grad=False)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, h_posemb, w_posemb, self.dim_tokens))
            trunc_normal_(self.pos_emb, std=0.02)

        # One cross attention layer followed by MLP block, an optional transformer, and an output projection
        if self.use_xattn:
            self.decoder = CrossAttention(
                dim=self.dim_tokens, num_heads=num_heads, qkv_bias=qkv_bias,
                attn_drop=attn_drop_rate, proj_drop=drop_rate)
            self.context_norm = norm_layer(self.dim_tokens)
            self.query_norm = norm_layer(self.dim_tokens)
            self.out_norm = norm_layer(self.dim_tokens)

            mlp_hidden_dim = int(self.dim_tokens * mlp_ratio)
            self.mlp = Mlp(in_features=self.dim_tokens, hidden_features=mlp_hidden_dim)

        # Optional full self-attention transformer layers
        if depth > 0:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
            self.decoder_transformer = nn.Sequential(*[
                Block(dim=self.dim_tokens, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                      attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(depth)
            ])
        else:
            self.decoder_transformer = nn.Identity()

        self.dim_patch = self.num_channels * self.P_H * self.P_W
        self.out_proj = nn.Linear(self.dim_tokens, self.dim_patch)

        if self.dim_tokens_enc is not None:
            self.init(dim_tokens_enc=dim_tokens_enc)

    def init(self, dim_tokens_enc: int = 768):
        '''
        Initialize parts of decoder that are dependent on dimension of encoder tokens.
        Should be called when setting up MultiMAE.

        :param dim_tokens_enc: Dimension of tokens coming from encoder
        '''
        self.dim_tokens_enc = dim_tokens_enc

        # Projection of encoder tokens to the patch dimension
        self.proj_context = nn.Linear(self.dim_tokens_enc, self.dim_tokens)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_emb', 'mask_token', 'task_embeddings'}

    def generate_context_embeddings(self, input_info,
                                    bs: int,
                                    size: Tuple[int, int],
                                    device: Optional[torch.device] = None):
        context_embeddings = []
        for task, info in input_info["tasks"].items():
            if self.task_embeddings is not None and task in self.task_embeddings:
                task_emb = repeat(self.task_embeddings[task], '() () d -> b n d', b=bs, n=info['num_tokens'])
            else:
                task_emb = torch.zeros((bs, info['num_tokens'], self.dim_tokens), device=device)

            if info['has_2d_posemb']:
                pos_emb = F.interpolate(self.pos_emb, size=size, mode='bilinear', align_corners=False)
                pos_emb = rearrange(pos_emb, 'b d nh nw -> b (nh nw) d')
                assert info['num_tokens'] == pos_emb.shape[1]
                task_emb = task_emb + pos_emb

            context_embeddings.append(task_emb)

        context_embeddings = torch.cat(context_embeddings, dim=1)

        return context_embeddings

    def get_queries_and_context(self, context_tokens, input_info, ids_keep, ids_restore):
        B = context_tokens.shape[0]
        H, W = input_info['image_size']
        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        if 'num_global_tokens' in input_info:
            context_tokens_without_global = context_tokens[:, :-input_info['num_global_tokens']]
        else:
            context_tokens_without_global = context_tokens

        # Add mask tokens
        mask_tokens = repeat(self.mask_token, '() () d -> b n d', b=B,
                             n=input_info['num_task_tokens'] - context_tokens_without_global.shape[1])
        context_with_mask = torch.cat([context_tokens_without_global, mask_tokens], dim=1)

        # Unshuffle context_with_mask
        context_with_mask = torch.gather(context_with_mask, dim=1,
                                         index=ids_restore.unsqueeze(-1).repeat(1, 1, context_with_mask.shape[2]))

        # Generate context_emb and add them to context
        context_emb = self.generate_context_embeddings(input_info=input_info, bs=B, size=(N_H, N_W),
                                                       device=context_tokens.device)
        context_with_mask = context_with_mask + context_emb

        # Generate queries
        if self.use_task_queries and self.task in input_info['tasks']:
            start_idx = input_info['tasks'][self.task]['start_idx']
            end_idx = input_info['tasks'][self.task]['end_idx']
            queries = context_with_mask[:, start_idx:end_idx]
        else:
            queries = repeat(self.mask_token, '() () d -> b n d', b=B, n=N_H * N_W)
            queries_pos_emb = F.interpolate(self.pos_emb, size=(N_H, N_W), mode='bilinear', align_corners=False)
            queries_pos_emb = rearrange(queries_pos_emb, 'b d nh nw -> b (nh nw) d')
            queries = queries + queries_pos_emb
            if self.task_embeddings is not None and self.task in self.task_embeddings:
                queries_task_emb = repeat(self.task_embeddings[self.task], '() () d -> b n d', b=B, n=N_H * N_W)
                queries = queries + queries_task_emb

        # Unshuffle context and keep only initial context (yes, again)
        context_tokens_without_global = torch.gather(context_with_mask, dim=1,
                                                     index=ids_keep.unsqueeze(-1).repeat(1, 1, context_with_mask.shape[2]))

        # Add back global tokens
        if 'num_global_tokens' in input_info:
            context_tokens = torch.cat(
                [context_tokens_without_global, context_tokens[:, -input_info['num_global_tokens']:]], dim=1)
        else:
            context_tokens = context_tokens_without_global

        return queries, context_tokens

    def forward(self,
                encoder_tokens: torch.Tensor,
                input_info: Dict,
                ids_keep: torch.Tensor,
                ids_restore: torch.Tensor,
                ):
        """
        Forward pass taking output tokens from encoder and optionally a subset of them corresponding
        to this output adapter's task (needs an additional mask describing position of these tokens in the queries).

        :param encoder_tokens: Output of encoder
        :param input_info: Dictionary with information about the input modalities
        :param ids_keep: IDs of unmasked tokens (tokens given to the encoder)
        :param ids_restore: IDs to unshuffle tokens
        """
        assert self.dim_tokens_enc is not None, 'Need to call init(dim_tokens_enc) function first'
        H, W = input_info['image_size']
        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)
        
        # Project encoder tokens to decoder tokens
        context_tokens = self.proj_context(encoder_tokens)

        # Get queries and context
        queries, context_tokens = self.get_queries_and_context(context_tokens, input_info, ids_keep, ids_restore)

        # Perform cross attention of queries to context tokens, followed by an MLP
        if self.use_xattn:
            x = self.decoder(self.query_norm(queries), self.context_norm(context_tokens))
            x = x + self.mlp(self.out_norm(x))
        else:
            x = queries

        # Optional transformer layers if depth > 0
        x = self.decoder_transformer(x)

        # Project each token to (C * P_H * P_W)
        x = self.out_proj(x)

        # Reshape sequence of patches into image
        x = rearrange(
            x, 'b (nh nw) (c ph pw) -> b c (nh ph) (nw pw)',
            nh=N_H, nw=N_W, ph=self.P_H, pw=self.P_W, c=self.num_channels
        )

        return x
    
    
    
class MaskedMSELoss(nn.Module):
    """L1 loss with masking
    :param patch_size: Patch size
    :param stride: Stride of task / modality
    :param norm_pix: Normalized pixel loss
    """

    def __init__(self, patch_size: int = 16, stride: int = 1, norm_pix=False):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.scale_factor = patch_size // stride
        self.norm_pix = norm_pix

    def patchify(self, imgs, nh, nw):
        p = self.scale_factor
        x = rearrange(imgs, "b c (nh p1) (nw p2) -> b (nh nw) (p1 p2 c)", nh=nh, nw=nw, p1=p, p2=p)
        return x

    def unpatchify(self, x, nh, nw):
        p = self.scale_factor
        imgs = rearrange(x, "b (nh nw) (p1 p2 c) -> b c (nh p1) (nw p2)", nh=nh, nw=nw, p1=p, p2=p)
        return imgs

    def forward(self, input, target, mask=None):

        H, W = input.shape[-2:]
        nh, nw = H // self.scale_factor, W // self.scale_factor

        if self.norm_pix:
            target = self.patchify(target, nh, nw)
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            eps = 1e-6
            target = (target - mean) / torch.sqrt(var + eps)
            target = self.unpatchify(target, nh, nw)

        loss = F.mse_loss(input, target, reduction='none')

        if mask is not None:
            if mask.sum() == 0:
                return torch.tensor(0).to(loss.device)

            # Resize mask and upsample
            mask = rearrange(mask, "b (nh nw) -> b nh nw", nh=nh, nw=nw)
            mask = F.interpolate(mask.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1)
            loss = loss.mean(dim=1)  # B, C, H, W -> B, H, W
            loss = loss * mask
            # Compute mean per sample
            loss = loss.flatten(start_dim=1).sum(dim=1) / mask.flatten(start_dim=1).sum(dim=1)
            loss = loss.nanmean() # Account for zero masks
        else:
            loss = loss.mean() # If this is ever nan, we want it to stop training

        return loss