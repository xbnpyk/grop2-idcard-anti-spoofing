import torch.nn as nn

def get_fusion_model(model_name, image_size, patch_size, num_class=2):

    if model_name == 'FaceBagNetFusion':
        from model.FaceBagNet import FusionNet
        net = FusionNet(num_class=num_class, type = 'A', fusion = 'se_fusion')

    elif model_name == 'ViTFusion':
        from model.MultiModalViT import MultiModalViT
        net = MultiModalViT( img_size = image_size,
                             patch_size = patch_size,
                             in_chans=3,
                             num_classes=num_class,
                             embed_dim=384,
                             depth=6,
                             num_heads=8,
                             mlp_ratio=4.,
                             qkv_bias=False,
                             qk_scale=None,
                             drop_rate=0.2,
                             attn_drop_rate=0.1,
                             drop_path_rate=0.1,
                             norm_layer=nn.LayerNorm,
                             init_values=0.,
                             use_learnable_pos_emb=True,
                             init_scale=0.,
                             use_mean_pooling=True
                             )

    return net

def get_model(model_name, image_size, patch_size, num_class=2,adapt = False):

    if model_name == 'FaceBagNet':
        from model.FaceBagNet import Net
        net = Net(num_class=num_class, type='A')
    elif model_name == 'ConvMixer':
        from model.ConvMixer import ConvMixer as Net
        net = Net(dim = 512, depth = 16, kernel_size = 9, patch_size = patch_size, n_classes = num_class)
        # net = Net(dim = 1024, depth = 12, kernel_size = 8, patch_size = patch_size, n_classes = num_class)
        # net = Net(dim = 1536, depth = 20, kernel_size = 9, patch_size = 7, n_classes = num_class) #ConvMixer-1536/20 • 7 9 
        # net = Net(dim = 768, depth = 32, kernel_size = 7, patch_size = 7, n_classes = num_class)
    elif model_name == 'ConvMixerFusion':
        from model.ConvMixer import ConvMixerFusion
        net = ConvMixerFusion(dim = 512, depth = 16, img_size = image_size, kernel_size = 9, patch_size = patch_size, n_classes = num_class, is_multi_modal = True)
    elif model_name == 'ConvMixerAdapt':
        from model.ConvMixer import ConvMixerAdapt
        net = ConvMixerAdapt(dim = 512, depth = 16, img_size = image_size, kernel_size = 9, patch_size = patch_size, n_classes = num_class, is_multi_modal = True,adapt = adapt)
    elif model_name == 'MLPMixer':
        from model.MLPMixer import MLPMixer as Net
        net = Net(image_size=image_size, channels=3, patch_size=patch_size, dim=512, depth=16,
                  num_classes=num_class, expansion_factor=4, dropout=0.)
    elif model_name == 'VisionPermutator':
        from model.ViP import Permutator as Net
        net = Net(image_size=image_size, patch_size=patch_size, dim=512, depth=16,
                  num_classes=num_class, expansion_factor=4, segments=4, dropout=0.)
    elif model_name == 'ViT':
        from model.MultiModalViT import MultiModalViT
        net = MultiModalViT( img_size = image_size,
                             patch_size = patch_size,
                             in_chans=3,
                             num_classes=2,
                             embed_dim=384,
                             depth=6,
                             num_heads=8,
                             mlp_ratio=4.,
                             qkv_bias=False,
                             qk_scale=None,
                             drop_rate=0.2,
                             attn_drop_rate=0.1,
                             drop_path_rate=0.1,
                             norm_layer=nn.LayerNorm,
                             init_values=0.,
                             use_learnable_pos_emb=True,
                             init_scale=0.,
                             use_mean_pooling=True,
                             is_multi_modal=False
                             )
    elif model_name == 'adaptViT':
        from model.MultiModalViT import MultiModalViT
        net = MultiModalViT( img_size = image_size,
                             patch_size = patch_size,
                             in_chans=3,
                             num_classes=2,
                             embed_dim=384,
                             depth=6,
                             num_heads=8,
                             mlp_ratio=4.,
                             qkv_bias=False,
                             qk_scale=None,
                             drop_rate=0.2,
                             attn_drop_rate=0.1,
                             drop_path_rate=0.1,
                             norm_layer=nn.LayerNorm,
                             init_values=0.,
                             use_learnable_pos_emb=True,
                             init_scale=0.,
                             use_mean_pooling=True,
                             is_multi_modal=False,
                             adapt = adapt
                             )
    return net

class get_liner(nn.Module):
    def __init__(self, model_name, n_classes):
        super().__init__()
        if model_name  == 'ConvMixer':
            dim = 512
        self.linear1 = nn.Linear(dim, n_classes)
        self.linear2 = nn.Linear(dim, n_classes)
        self.linear3 = nn.Linear(dim, n_classes)
        self.linear4 = nn.Linear(dim, n_classes)
        self.linear5 = nn.Linear(dim, n_classes)

    def forward(self, x):
        return self.linear1(x) , self.linear2(x) , self.linear3(x) , self.linear4(x) , self.linear5(x)
