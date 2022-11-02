import numpy as np
import pandas as pd
from einops import rearrange, repeat

import torch
import torch.nn as nn


class SLN(nn.Module):
    """
    Self-modulated LayerNorm
    """
    def __init__(self, num_features):
        super(SLN, self).__init__()
        self.ln = nn.LayerNorm(num_features)
        # self.gamma = nn.Parameter(torch.FloatTensor(1, 1, 1))
        # self.beta = nn.Parameter(torch.FloatTensor(1, 1, 1))
        self.gamma = nn.Parameter(torch.randn(1, 1, 1)) #.to("cuda")
        self.beta = nn.Parameter(torch.randn(1, 1, 1)) #.to("cuda")

    def forward(self, hl, w):
        return self.gamma * w * self.ln(hl) + self.beta * w


class MLP(nn.Module):
    def __init__(self, in_feat, hid_feat = None, out_feat = None, dropout = 0.):
        super().__init__()
        if not hid_feat:
            hid_feat = in_feat
        if not out_feat:
            out_feat = in_feat
        self.linear1 = nn.Linear(in_feat, hid_feat)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(hid_feat, out_feat)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)


class Attention(nn.Module):
    """
    Implement multi head self attention layer using the "Einstein summation convention".

    Parameters
    ----------
    dim:
        Token's dimension, EX: word embedding vector size
    num_heads:
        The number of distinct representations to learn
    dim_head:
        The dimension of the each head
    discriminator:
        Used in discriminator or not.
    """
    def __init__(self, dim, num_heads = 4, dim_head = None, discriminator = False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.dim_head = int(dim / num_heads) if dim_head is None else dim_head
        self.weight_dim = self.num_heads * self.dim_head
        self.to_qkv = nn.Linear(dim, self.weight_dim * 3, bias = False)
        self.scale_factor = dim ** -0.5
        self.discriminator = discriminator
        self.w_out = nn.Linear(self.weight_dim, dim, bias = True)

        if discriminator:
            u, s, v = torch.svd(self.to_qkv.weight)
            self.init_spect_norm = torch.max(s)

    def forward(self, x):
        assert x.dim() == 3

        if self.discriminator:
            u, s, v = torch.svd(self.to_qkv.weight)
            self.to_qkv.weight = torch.nn.Parameter(self.to_qkv.weight * self.init_spect_norm / torch.max(s))

        # Generate the q, k, v vectors
        qkv = self.to_qkv(x)
        q, k, v = tuple(rearrange(qkv, 'b t (d k h) -> k b h t d', k = 3, h = self.num_heads))

        # Enforcing Lipschitzness of Transformer Discriminator
        # Due to Lipschitz constant of standard dot product self-attention
        # layer can be unbounded, so adopt the l2 attention replace the dot product.
        if self.discriminator:
            attn = torch.cdist(q, k, p = 2)
        else:
            attn = torch.einsum("... i d, ... j d -> ... i j", q, k)
        scale_attn = attn * self.scale_factor
        scale_attn_score = torch.softmax(scale_attn, dim = -1)
        result = torch.einsum("... i j, ... j d -> ... i d", scale_attn_score, v)

        # re-compose
        result = rearrange(result, "b h t d -> b t (h d)")
        return self.w_out(result)


class DEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads = 4, dim_head = None,
        dropout = 0., mlp_ratio = 4):
        super(DEncoderBlock, self).__init__()
        self.attn = Attention(dim, num_heads, dim_head, discriminator = True)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = MLP(dim, dim * mlp_ratio, dropout = dropout)

    def forward(self, x):
        x1 = self.norm1(x)
        x = x + self.dropout(self.attn(x1))
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x


class GEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads = 4, dim_head = None,
        dropout = 0., mlp_ratio = 4):
        super(GEncoderBlock, self).__init__()
        self.attn = Attention(dim, num_heads, dim_head)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = SLN(dim)
        self.norm2 = SLN(dim)

        self.mlp = MLP(dim, dim * mlp_ratio, dropout = dropout)

    def forward(self, hl, x):
        hl_temp = self.dropout(self.attn(self.norm1(hl, x))) + hl
        hl_final = self.mlp(self.norm2(hl_temp, x)) + hl_temp
        return x, hl_final


class GTransformerEncoder(nn.Module):
    def __init__(self,
        dim,
        blocks = 6,
        num_heads = 8,
        dim_head = None,
        dropout = 0
    ):
        super(GTransformerEncoder, self).__init__()
        self.blocks = self._make_layers(dim, blocks, num_heads, dim_head, dropout)

    def _make_layers(self,
        dim,
        blocks = 6,
        num_heads = 8,
        dim_head = None,
        dropout = 0
    ):
        layers = []
        for _ in range(blocks):
            layers.append(GEncoderBlock(dim, num_heads, dim_head, dropout))
        return nn.Sequential(*layers)

    def forward(self, hl, x):
        for block in self.blocks:
            x, hl = block(hl, x)
        return x, hl


class DTransformerEncoder(nn.Module):
    def __init__(self,
        dim,
        blocks = 6,
        num_heads = 8,
        dim_head = None,
        dropout = 0
    ):
        super(DTransformerEncoder, self).__init__()
        self.blocks = self._make_layers(dim, blocks, num_heads, dim_head, dropout)

    def _make_layers(self,
        dim,
        blocks = 6,
        num_heads = 8,
        dim_head = None,
        dropout = 0
    ):
        layers = []
        for _ in range(blocks):
            layers.append(DEncoderBlock(dim, num_heads, dim_head, dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class SineLayer(nn.Module):
    """
    Paper: Implicit Neural Representation with Periodic Activ ation Function (SIREN)
    """
    def __init__(self, in_features, out_features, bias = True,is_first = False, omega_0 = 30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
class GITConfig:
    """ base GIT config, params common to all GIT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    # vocab_size代表有字典中有多少子 在这代表有多少种地层编码 block_size 在这代表所有的像素数 patch_size 代表块的大小
    def __init__(self,img_size,block_size,grid, in_chans ,patch_size, if_bert=True, **kwargs):
        self.img_size=img_size
        self.in_chans=in_chans
        # self.vocab_size = vocab_size
        self.block_size = block_size
        self.patch_size = patch_size
        self.grid = grid
        self.if_bert = if_bert
        for k, v in kwargs.items():
            setattr(self, k, v)

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=None, flatten=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1],self.img_size[2] // self.patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1]*self.grid_size[2]
        self.flatten = flatten
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x=x.permute(0,4,1,2,3)
        B, C, H, W, L = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Generator(nn.Module):
    def __init__(self, config,
        initialize_size = 8,
        # dim = 384,
        # blocks = 8,
        # num_heads = 6,
        dim_head = None,
        # dropout = 0,
        out_channels = 1):
        super(Generator, self).__init__()
        self.config = config
        df128 = pd.read_csv("data/I_J_K.csv", low_memory=False)
        self.posEmbedding = ((torch.FloatTensor(df128[['I', 'J', 'K']].values).reshape(128, 128, 128, -1)[::16, ::16,
                              ::16].reshape(-1, 3)) / 128).cuda()
        self.tok_emb = nn.Embedding(config.stratNum, 1)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.embed_dim))
        #self.pos_emb1D = nn.Parameter(torch.randn(self.initialize_size * 8, dim))
        self.PixelShuffle = nn.PixelShuffle(2)
        self.drop = nn.Dropout(config.embd_pdrop)
        #编码
        self.patch_embed = PatchEmbed(img_size=config.img_size, patch_size=config.patch_size, in_chans=config.in_chans,
                                      embed_dim=config.embed_dim)
        self.poSpatch_embed = PatchEmbed(img_size=config.img_size, patch_size=config.patch_size, in_chans=3,
                                         embed_dim=config.embed_dim)

        #self.mlp = nn.Linear(1024, (self.initialize_size * 8) * self.config.embed_dim)
        self.Transformer_Encoder = GTransformerEncoder(config.embed_dim, config.blocks, config.num_heads, dim_head, config.dropout)

        # Implicit Neural Representation
        self.w_out = nn.Sequential(
            SineLayer(config.embed_dim, config.embed_dim * 2, is_first = True, omega_0 = 30.),
            SineLayer(config.embed_dim * 2, initialize_size * 8 * out_channels, is_first = False, omega_0 = 30)
        )
        self.sln_norm = SLN(self.config.embed_dim)
        output_size = int((config.img_size[0] * config.img_size[1] * config.img_size[2]) / (config.block_size))
        self.head = nn.Linear(config.embed_dim, output_size * config.stratNum, bias=False)
        self.pos_head = nn.Linear(3, config.embed_dim)
        self.nos_head = nn.Linear(1, config.embed_dim)


    def forward(self, idx):
        b, h, w, l, c = idx.size()
        idx = idx.long()
        stratNum = self.config.stratNum
        token_embeddings = (idx / stratNum).float()
        nos = self.nos_head(torch.randn(b, 512, 1).cuda())
        token_embeddings = self.patch_embed(token_embeddings)
        position_embeddings = self.posEmbedding.repeat(b, 1, 1)
        position_embeddings = self.pos_head(position_embeddings)
        x = nos + token_embeddings + position_embeddings
        # x = self.mlp(noise).view(-1, self.initialize_size * 8, self.dim)
        x, hl = self.Transformer_Encoder(self.pos_emb, x)
        x = self.sln_norm(hl, x)
        x = self.w_out(x)  # Replace to siren
        result = self.head(x).view(b, -1, stratNum)

        return result


class Discriminator(nn.Module):
    def __init__(self,config,
        # in_channels = 1,
        # patch_size = 8,
        # extend_size = 2,
        # dim = 384,
        # blocks = 6,
        # num_heads = 6,
        dim_head = None,
        # dropout = 0
    ):
        super(Discriminator, self).__init__()
        self.patch_embed = PatchEmbed(img_size=config.img_size, patch_size=config.patch_size, in_chans=config.in_chans,
                                     embed_dim=config.embed_dim)
        dim=config.embed_dim
        #self.token_dim = in_channels * (self.patch_size ** 2)
        #self.project_patches = nn.Linear(self.token_dim, dim)
        #self.emb_dropout = nn.Dropout(dropout)

        #self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        #self.pos_emb1D = nn.Parameter(torch.randn(self.token_dim + 1, dim))
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, 1)
        )
        self.Transformer_Encoder = DTransformerEncoder(config.embed_dim, config.blocks, config.num_heads, dim_head,
                                                       config.dropout)
        #self.Transformer_Encoder = DTransformerEncoder(dim, config.blocks, config.num_heads, config.dim_head, config.dropout)


    def forward(self, img):
        img = torch.unsqueeze(img,-1)
        img = img.float()
        # b, h, w, l, c = idx.size()
        img_patches = self.patch_embed(img)
        # Generate overlappimg image patches
        #stride_h = (img.shape[2] - self.patch_size) // 8 + 1
        #stride_w = (img.shape[3] - self.patch_size) // 8 + 1
        #stride_c = (img.shape[4] - self.patch_size) // 8 + 1
        #img_patches = img.unfold(2, self.patch_size, stride_h).unfold(3, self.patch_size, stride_w).unfold(4, self.patch_size, stride_c)
        #img_patches = img_patches.contiguous().view(
            #img_patches.shape[0], img_patches.shape[2] * img_patches.shape[3], img_patches.shape[1] * img_patches.shape[4] * img_patches.shape[5]
        #)
        #img_patches = self.project_patches(img_patches)
        #batch_size, tokens, _ = img_patches.shape
        # Prepend the classifier token
        #cls_token = repeat(self.cls_token, '() n d -> b n d', b = batch_size)
        #img_patches = torch.cat((cls_token, img_patches), dim = 1)
        # Plus the positional embedding
        #img_patches = img_patches + self.pos_emb1D[: tokens + 1, :]
        #img_patches = self.emb_dropout(img_patches)
        result = self.Transformer_Encoder(img_patches)
        logits = self.mlp_head(result[:, 0, :])
        logits = nn.Sigmoid()(logits)
        return logits


def test_both():
    B, dim = 10, 1024
    G = Generator(initialize_size = 8, dropout = 0.1)
    noise = torch.FloatTensor(np.random.normal(0, 1, (B, dim)))
    fake_img = G(noise)
    D = Discriminator(patch_size = 8, dropout = 0.1)
    D_logits = D(fake_img)
    print(D_logits)
    print(f"Max: {torch.max(D_logits)}, Min: {torch.min(D_logits)}")
