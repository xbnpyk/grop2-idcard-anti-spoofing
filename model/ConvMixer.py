import torch
import torch.nn as nn
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class get_liner(nn.Module):
    def __init__(self, dim, n_classes):
        super().__init__()
        self.linear1 = nn.Linear(dim, n_classes)
        self.linear2 = nn.Linear(dim, n_classes)
        self.linear3 = nn.Linear(dim, n_classes)
        self.linear4 = nn.Linear(dim, n_classes)

    def forward(self, x):
        return self.linear1(x) , self.linear2(x) , self.linear3(x) , self.linear4(x)

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

class FWT(nn.Module):

    def __init__(self,dim,gamma,beta):
        super().__init__()
        print('set gamma, beta as ', gamma, beta)
        self.gamma = torch.nn.Parameter(torch.ones(1, 1, dim) * gamma)
        self.beta = torch.nn.Parameter(torch.ones(1, 1, dim) * beta)
        self.num_features = dim

    def softplus(self,x):
        return torch.nn.functional.softplus(x, beta=100)

    def forward(self,x):
        gamma = (1 + torch.randn(
            1,
            1,
            self.num_features,
            dtype=self.gamma.dtype,
            device=self.gamma.device) * self.softplus(self.gamma)).expand_as(x)
        beta = (torch.randn(
            1,
            1,
            self.num_features,
            dtype=self.beta.dtype,
            device=self.beta.device) * self.softplus(self.beta)).expand_as(x)
        x = gamma * x + beta

class FeatureWiseTransformation2d_fw(nn.BatchNorm2d):
    feature_augment = False
    def __init__(self, num_features, momentum=0.1, track_running_stats=True):
        super(FeatureWiseTransformation2d_fw, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats)
        self.weight.fast = None
        self.bias.fast = None
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.zeros(num_features))
        if self.feature_augment: # initialize {gamma, beta} with {0.3, 0.5}
            self.gamma = torch.nn.Parameter(torch.ones(1, num_features, 1, 1)*0.3)
            self.beta  = torch.nn.Parameter(torch.ones(1, num_features, 1, 1)*0.5)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def forward(self, x, step=0):
        if self.weight.fast is not None and self.bias.fast is not None:
            weight = self.weight.fast
            bias = self.bias.fast
        else:
            weight = self.weight
            bias = self.bias
        if self.track_running_stats:
            out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training, momentum=self.momentum)
        else:
            out = F.batch_norm(x, torch.zeros_like(x), torch.ones_like(x), weight, bias, training=True, momentum=1)

        # apply feature-wise transformation
        if self.feature_augment and self.training:
            gamma = (1 + torch.randn(1, self.num_features, 1, 1, dtype=self.gamma.dtype, device=self.gamma.device)*softplus(self.gamma)).expand_as(out)
            beta = (torch.randn(1, self.num_features, 1, 1, dtype=self.beta.dtype, device=self.beta.device)*softplus(self.beta)).expand_as(out)
            out = gamma*out + beta
        return out

class Adapter(nn.Module):

  def __init__(self,dim):
    super(Adapter, self).__init__()
    self.hidden_size = dim
    self.adapter_size = 32
    self.adapter_initializer_range = 0.0002
    self.sigmoid = nn.Sigmoid()

    self.down_project = nn.Linear(self.hidden_size, self.adapter_size)
    self.activation = nn.GELU()
    self.up_project = nn.Linear(self.adapter_size, self.hidden_size)
    self.init_weights()

  def forward(self, hidden_states):
    print(hidden_states.shape)
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


class AdaptBlock(nn.Module):
    def __init__(self,dim,kernel_size,adapt):
        super().__init__()
        self.adapt = adapt
        self.conv1 = nn.Sequential(
                            nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                            nn.GELU(),
                            nn.BatchNorm2d(dim)
                        )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(dim, dim, kernel_size=1),
                        nn.GELU(),
                        nn.BatchNorm2d(dim)
                        )
        if self.adapt:
            # self.adapt11 = Adapter(dim)
            # self.adapt21 = Adapter(dim)
            # self.adapt12 = Adapter(dim)
            # self.adapt22 = Adapter(dim)
            # self.FWT = FWT(dim,0.3,0.5)
            self.FWT = FeatureWiseTransformation2d_fw(dim)

    def forward(self,x):
        if len(x) == 2:
            x, total_loss = x[0], x[1]
        else:
            total_loss = 0
        tmpx = x
        x = self.conv1(x)
        # if self.adapt:
        #     d1 = self.adapt11(x)
        #     d2 = self.adapt12(x)
        #     x = d1 + d2
        #     o1 = F.cosine_similarity(d1.transpose(1, 2), d2.transpose(1, 2))
        #     o1 = (o1 * o1).mean(0).mean(0)
        x = x + tmpx
        x = self.conv2(x)
        if self.adapt:
            # d1 = self.adapt21(x)
            # d2 = self.adapt22(x)
            # x = d1 + d2
            # o2 = F.cosine_similarity(d1.transpose(1, 2), d2.transpose(1, 2))
            # o2 = (o2 * o2).mean(0).mean(0)
            x = self.FWT(x)
        # if self.adapt:
        #     total_loss += o1 + o2
        return [x, total_loss]


def ConvMixer(dim, depth, kernel_size=9, patch_size=7, n_classes=1000):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size = patch_size, stride = patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        # nn.Linear(dim, n_classes)
        get_liner(dim, n_classes)
    )
class ConvMixerFusion(nn.Module):
    def __init__(self, dim, depth, img_size, kernel_size=9, patch_size=7, n_classes=1000, is_multi_modal = False):
        super().__init__()

        self.is_multi_modal = is_multi_modal
        if is_multi_modal:
            # self.patch_embed0 = PatchEmbed(
            #     img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=dim)
            # self.patch_embed1 = PatchEmbed(
            #     img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=dim)
            self.patch_embed0 = nn.Conv2d(3, dim//2, kernel_size = patch_size, stride = patch_size)
            self.patch_embed1 = nn.Conv2d(3, dim//2, kernel_size = patch_size, stride = patch_size)
            
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=dim)
            num_patches = self.patch_embed.num_patches

        self.net = nn.Sequential(
            # nn.Conv2d(3, dim, kernel_size = patch_size, stride = patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            *[nn.Sequential(
                    Residual(nn.Sequential(
                        nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                        nn.GELU(),
                        nn.BatchNorm2d(dim)
                    )),
                    nn.Conv2d(dim, dim, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
            ) for i in range(depth)],
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            # nn.Linear(dim, n_classes)
            get_liner(dim, n_classes)
        )

    def forward(self, x):
        # x = self.forward_features(x)
        # x1 = self.head(x)
        # x2 = self.head2(x)
        # x3 = self.head3(x)
        # x4 = self.head4(x)
        if self.is_multi_modal:
            color, fft = x[:, 3:6, :, :], x[:, 0:3, :, :]
            x0 = self.patch_embed0(color)
            x1 = self.patch_embed1(fft)
            x = torch.cat([x0,x1], dim=1)
        else:
            x = self.patch_embed(x)

        return self.net(x)

class ConvMixerAdapt(nn.Module):
    def __init__(self, dim, depth, img_size, kernel_size=9, patch_size=7, n_classes=1000, is_multi_modal = False,adapt = False):
        super().__init__()

        self.adapt = adapt

        self.is_multi_modal = is_multi_modal         
        # self.patch_embed = PatchEmbed(
        #     img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=dim)
        self.patch_embed = nn.Conv2d(3, dim, kernel_size = patch_size, stride = patch_size)
        self.activation = nn.GELU()
        self.norm = nn.BatchNorm2d(dim)
        self.blocks = nn.Sequential(*[AdaptBlock(dim,kernel_size,adapt) for i in range(depth)])
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.falt = nn.Flatten()
            # nn.Linear(dim, n_classes)
        self.Linear = get_liner(dim, n_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.activation(x)
        x = self.norm(x)
        x, cosloss = self.blocks(x)
        x = self.pool(x)
        x = self.falt(x)
        x = self.Linear(x)

        if self.adapt:
            return x
        return x

if __name__ == '__main__':
    # ConvMixer-1536/20 (patch size 7, kernel size 9)
    # ConvMixer-768/32 (patch size 7, kernel size 7)
    # ConvMixer-1024/20 (patch size 14, kernel size 9)
    net = ConvMixer(dim = 512, depth = 16, kernel_size=9, patch_size=14, n_classes=2)
    print(net)