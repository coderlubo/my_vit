from functools import partial
from turtle import forward
from typing import OrderedDict
from regex import R
from sympy import Order
from torch import nn
import torch

from utils.global_var import get_global_logger

logger = get_global_logger()

def drop_path(x, drop_prob: float=0., training: bool=False):
    """
    drop_path 函数在训练神经网络时，通过随机丢弃路径来实现正则化，从而提高模型的泛化能力。这种技术在深度残差网络中尤为常见，有助于防止过拟合。
    """

    if drop_prob == 0. or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor = random_tensor.floor()
    output = x.div(keep_prob) * random_tensor

    return output

class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        img_size = (img_size, img_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
    def forward(self, x):
        B, C, H, W = x.shape

        if H != self.img_size[0] or W != self.img_size[1]:
            logger.error("Input image size ({}*{}) doesn't match model ({}*{})".format(H, W, self.img_size[0], self.img_size[1]))
            raise

        # flatten: [B, C, H, W] -> [B, C, num_patches]
        # transpose: [B, C, num_patches] -> [B, num_patches, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0.):
        """
        Args:
            dim: 输入 token 的维度。
            num_heads: 注意力头的数量，默认为 8。
            qkv_bias: 是否在 qkv 线性层中使用偏置，默认为 False。
            qk_scale: 缩放因子，如果未提供则使用 head_dim ** -0.5。
            attn_drop_ratio: 注意力权重的 dropout 比例。
            proj_drop_ratio: 输出的 dropout 比例。
        """
        super(Attention, self).__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches+1, total_embed_dim]
        B, N, C = x.shape

        # qkv: [batch_size, num_patches+1, 3*total_embed_dim]
        # reshape: [batch_size, num_patches+1, 3, num_heads, embed_dim_per_head]
        # permute: [3, batch_size, num_heads, num_patches+1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # q, k, v: [batch_size, num_heads, num_patches+1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # transpose: [batch_size, num_heads, num_patches+1, embed_dim_per_head] -> [batch_size, num_heads, embed_dim_per_head, num_patches+1]
        # @: multiply -> [batch_size, num_heads, num_patches+1, num_patches+1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches+1, embed_dim_per_head]
        # transpose: [batch_size, num_heads, num_patches+1, embed_dim_per_head] -> [batch_size, num_patches+1, num_heads, embed_dim_per_head]
        # reshape: [batch_size, num_patches+1, num_heads*embed_dim_per_head]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0,):
        super(Mlp, self).__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """
        dim: 输入的维度。
        num_heads: 多头注意力机制中的头数。
        mlp_ratio: MLP 层的隐藏层维度与输入维度的比例。
        qkv_bias: 是否在 QKV 线性层中使用偏置。
        qk_scale: QK 缩放因子。
        drop_ratio: Dropout 比例。
        attn_drop_ratio: 注意力层的 Dropout 比例。
        drop_path_ratio: DropPath 比例。
        act_layer: 激活函数层。
        norm_layer: 归一化层。
        """
        super(Block, self).__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)

        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

def _init_vit_weights(model):
    """
    判断模块类型并初始化权重：

    if isinstance(m, nn.Linear):：如果 m 是一个全连接层（nn.Linear），则使用截断正态分布初始化权重，并将偏置初始化为零。
    nn.init.trunc_normal_(m.weight, std=.01)：使用标准差为 0.01 的截断正态分布初始化权重。
    if m.bias is not None: nn.init.zeros_(m.bias)：如果存在偏置，则将其初始化为零。
    elif isinstance(m, nn.Conv2d):：如果 m 是一个二维卷积层（nn.Conv2d），则使用 Kaiming 正态分布初始化权重，并将偏置初始化为零。
    nn.init.kaiming_normal_(m.weight, mode="fan_out")：使用 Kaiming 正态分布初始化权重，模式为 "fan_out"。
    if m.bias is not None: nn.init.zeros_(m.bias)：如果存在偏置，则将其初始化为零。
    elif isinstance(m, nn.LayerNorm):：如果 m 是一个层归一化层（nn.LayerNorm），则将权重初始化为一，偏置初始化为零。
    nn.init.zeros_(m.bias)：将偏置初始化为零。
    nn.init.ones_(m.weight)：将权重初始化为一。
    """
    if isinstance(model, nn.Linear):
        nn.init.trunc_normal_(model.weight, std=0.02)
        if model.bias is not None:
            nn.init.zeros_(model.bias)
        
    elif isinstance(model, nn.Conv2d):
        nn.init.kaiming_normal_(model.weight, mode='fan_out')
        if model.bias is not None:
            nn.init.zeros_(model.bias)

    elif isinstance(model, nn.LayerNorm):
        nn.init.zeros_(model.bias)
        nn.init.ones_(model.weight)


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, representation_size=None, distilled=False, drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None, act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        # dpr 是一个包含 depth 个浮点数的列表，这些数值从 0 到 drop_path_ratio 之间均匀分布。
        dpr = [x.item() for x in torch.linspace
        (0, drop_path_ratio, depth)]


        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i], norm_layer=norm_layer, act_layer=act_layer) for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        
        # Representation layer
        # 这段代码根据条件动态地设置模型的表示层，选择使用一个全连接层加激活函数的组合，或者使用一个恒等映射层。
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict([
                    ('fc', nn.Linear(embed_dim, representation_size)),
                    ('act', nn.Tanh())
                ])
            )
        else:
            self.has_logits = True
            self.pre_logits = nn.Identity()

        # self.head 和 self.head_dist 是模型的输出层，用于分类任务。
        # 根据 num_classes 的值，选择使用线性层或恒等层。
        # self.head_dist 仅在启用蒸馏时定义。
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None

        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x) # [B, 197, 768]

        # [1, 1, embed_dim] -> [B, 1, embed_dim]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        if self.dist_token is None:

            # [B, num_patches+1, embed_dim]
            x = torch.cat((cls_token, x), dim=1)

        else: 
            
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)

        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]
        
    def forward(self, x):
        x = self.forward_features(x)

        if self.head_dist is not None:
            
            x, x_dist = self.head(x[0]), self.head_dist(x[1])

            if self.training and not torch.jit.is_scripting():
                return x, x_dist
            else:
                return (x + x_dist) / 2
        
        else:
            x = self.head(x)
        
        return x
    

def vit_base_patch16_224(num_classes: int=1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        representation_size=None,
        num_classes=num_classes
    )

    return model

def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        representation_size=768 if has_logits else None,
        num_classes=num_classes
    )

    return model
    




        


