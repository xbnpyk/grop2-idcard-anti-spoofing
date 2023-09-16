import math
from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }

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
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None,adapt=False):
        super().__init__()
        self.adapt = adapt
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

        if self.adapt:
            self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

            self.adapt11 = Adapter()
            self.adapt21 = Adapter()
            self.adapt12 = Adapter()
            self.adapt22 = Adapter()

            gamma = 0.3
            beta = 0.5
            print('set gamma, beta as ', gamma, beta)
            self.gamma = torch.nn.Parameter(torch.ones(1, 1, dim) * gamma)
            self.beta = torch.nn.Parameter(torch.ones(1, 1, dim) * beta)

    def forward(self, x):
        if len(x) == 2:
            x, total_loss = x[0], x[1]
        else:
            total_loss = 0
        if self.gamma_1 is None:
            tmpx = x
            x = self.attn(self.norm1(x))
            if self.adapt:
                d1 = self.adapt11(x)
                d2 = self.adapt12(x)
                x = d1 + d2
                o1 = F.cosine_similarity(d1.transpose(1, 2), d2.transpose(1, 2))
                o1 = (o1 * o1).mean(0).mean(0)
            x = self.drop_path(x)
            x = x + tmpx

            tmpx = x
            x = self.mlp(self.norm2(x))
            if self.adapt:
                d1 = self.adapt21(x)
                d2 = self.adapt22(x)
                x = d1 + d2
                o2 = F.cosine_similarity(d1.transpose(1, 2), d2.transpose(1, 2))
                o2 = (o2 * o2).mean(0).mean(0)
            x = self.drop_path(x)
            x = x + tmpx
            # x = x + self.drop_path(self.attn(self.norm1(x)))
            # x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            tmpx = x
            x = self.attn(self.norm1(x))
            if self.adapt:
                d1 = self.adapt11(x)
                d2 = self.adapt12(x)
                x = d1 + d2
                o1 = F.cosine_similarity(d1.transpose(1, 2), d2.transpose(1, 2))
                o1 = (o1 * o1).mean(0).mean(0)
            x = self.drop_path(self.gamma_1 * x)
            x = x + tmpx

            tmpx = x
            x = self.mlp(self.norm2(x))
            if self.adapt:
                d1 = self.adapt21(x)
                d2 = self.adapt22(x)
                x = d1 + d2
                o2 = F.cosine_similarity(d1.transpose(1, 2), d2.transpose(1, 2))
                o2 = (o2 * o2).mean(0).mean(0)
            x = self.drop_path(self.gamma_2 * x)
            x = x + tmpx
            gamma = (1 + torch.randn(
                1,
                1,
                self.num_features,
                dtype=self.gamma.dtype,
                device=self.gamma.device) * softplus(self.gamma)).expand_as(x)
            beta = (torch.randn(
                1,
                1,
                self.num_features,
                dtype=self.beta.dtype,
                device=self.beta.device) * softplus(self.beta)).expand_as(x)
            x = gamma * x + beta
            # x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            # x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        if self.adapt:
            total_loss += o1 + o2
        return [x, total_loss]
        # return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class Adapter(nn.Module):

  def __init__(self):
    super(Adapter, self).__init__()
    self.hidden_size = 384
    self.adapter_size = 32
    self.adapter_initializer_range = 0.0002
    self.sigmoid = nn.Sigmoid()

    self.down_project = nn.Linear(self.hidden_size, self.adapter_size)
    self.activation = nn.GELU()
    self.up_project = nn.Linear(self.adapter_size, self.hidden_size)
    self.init_weights()

  def forward(self, hidden_states):
    # print(hidden_states.shape)
    down_projected = self.down_project(hidden_states)
    activated = self.activation(down_projected)
    up_projected = self.up_project(activated)

    return hidden_states + up_projected

  def init_weights(self):
    # Slightly different from the TF version which uses truncated_normal for initialization
    # cf https://github.com/pytorch/pytorch/pull/5617
    self.down_project.weight.data.normal_(
        mean=0.0, std=self.adapter_initializer_range)
    self.down_project.bias.data.zero_()
    self.up_project.weight.data.normal_(
        mean=0.0, std=self.adapter_initializer_range)
    self.up_project.bias.data.zero_()


def softplus(x):
  return torch.nn.functional.softplus(x, beta=100)

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
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
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiModalViT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 init_scale=0.,
                 use_mean_pooling=True,
                 is_multi_modal = True,
                 adapt = False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.is_multi_modal = is_multi_modal
        if is_multi_modal:
            self.patch_embed0 = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            self.patch_embed1 = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            self.patch_embed2 = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            num_patches = self.patch_embed0.num_patches * 3
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            num_patches = self.patch_embed.num_patches

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, adapt=adapt)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head2 = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head3 = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head4 = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        # trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.head.weight, std=.02)
        trunc_normal_(self.head2.weight, std=.02)
        trunc_normal_(self.head3.weight, std=.02)
        trunc_normal_(self.head4.weight, std=.02)
        self.apply(self._init_weights)

        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)
        self.head2.weight.data.mul_(init_scale)
        self.head2.bias.data.mul_(init_scale)
        self.head3.weight.data.mul_(init_scale)
        self.head3.bias.data.mul_(init_scale)
        self.head4.weight.data.mul_(init_scale)
        self.head4.bias.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head2 = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head3 = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head4 = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):

        if self.is_multi_modal:
            color, depth, ir = x[:, 3:6, :, :], x[:, 0:3, :, :], x[:, 6:, :, :]
            x0 = self.patch_embed0(color)
            x1 = self.patch_embed1(depth)
            x2 = self.patch_embed2(ir)
            x = torch.cat([x0,x1,x2], dim=1)
        else:
            x = self.patch_embed(x)

        B, _, _ = x.size()

        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)

        for blk in self.blocks:
            x , total_loss = blk(x)

        if self.fc_norm is not None:
            t = x[:, 1:, :]
            return self.fc_norm(t.mean(1)), total_loss
        else:
            return x[:, 0], total_loss

    def forward(self, x):
        x, total_loss = self.forward_features(x)
        x1 = self.head(x)
        x2 = self.head2(x)
        x3 = self.head3(x)
        x4 = self.head4(x)
        return x1,x2,x3,x4,total_loss

if __name__ == '__main__':
    model = MultiModalViT(img_size = 96, patch_size = 8, in_chans = 3, num_classes = 2)
    print(model)
    input0 = torch.zeros([1,3,96,96])
    input1 = torch.zeros([1,3,96,96])
    input2 = torch.zeros([1,3,96,96])
    input = torch.cat([input0,input1,input2], dim=1)
    out = model(input)
    print(out)