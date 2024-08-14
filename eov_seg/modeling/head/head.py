import math
from numpy import pad
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

import torch
import torch.nn.functional as F
from torch import nn

from typing import Optional, List
from torch import nn, Tensor
from detectron2.layers import Conv2d, DepthwiseSeparableConv2d, get_norm


def get_classification_logits(x, text_classifier, logit_scale, num_templates=None):
    # x in shape of [B, *, C]
    # text_classifier in shape of [num_classes, C]
    # logit_scale is a learnable scalar https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/model.py#L201
    # return: [B, *, num_classes]
    x = F.normalize(x, dim=-1)
    logit_scale = torch.clamp(logit_scale.exp(), max=100)
    pred_logits = logit_scale * x @ text_classifier.T # B, *, N + 1
    # max ensembel as in OpenSeg/ODISE
    final_pred_logits = []
    cur_idx = 0
    for num_t in num_templates:
        final_pred_logits.append(pred_logits[:, :, cur_idx: cur_idx + num_t].max(-1).values)
        cur_idx += num_t
    final_pred_logits.append(pred_logits[:, :, -1]) # the last classifier is for void
    final_pred_logits = torch.stack(final_pred_logits, dim=-1)
    return final_pred_logits


# Ref: https://github.com/NVlabs/ODISE/blob/e97b06c424c575fec9fc5368dd4b3e050d91abc4/odise/modeling/meta_arch/odise.py#L923
class MaskPooling(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x, mask):
        """
        Args:
            x: [B, C, H, W]
            mask: [B, Q, H, W]
        """
        if not x.shape[-2:] == mask.shape[-2:]:
            # reshape mask to x
            mask = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)
        with torch.no_grad():
            mask = mask.detach()
            mask = (mask > 0).to(mask.dtype)
            denorm = mask.sum(dim=(-1, -2), keepdim=True) + 1e-8

        mask_pooled_x = torch.einsum(
            "bchw,bqhw->bqc",
            x,
            mask / denorm,
        )
        return mask_pooled_x


class FFN(nn.Module):
    def __init__(self, embed_dims=256, feedforward_channels=1024, num_fcs=2, add_identity=True):
        super(FFN, self).__init__()
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(nn.Linear(in_channels, feedforward_channels),
                              nn.ReLU(True),
                              nn.Dropout(0.0)
                              ))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(0.0))
        self.layers = nn.Sequential(*layers)
        self.add_identity = add_identity
        self.dropout_layer = nn.Dropout(0.0)

    def forward(self, x, identity=None):
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class VAS(nn.Module):
    def __init__(self, in_channels, out_channels, guide_channels, embed_channels, num_heads=1):
        super(VAS, self).__init__()

        assert (out_channels % num_heads == 0 and embed_channels % num_heads == 0), 'out_channels and embed_channels must be divisible by num_heads.'
        self.num_heads = num_heads
        self.head_channels = embed_channels // num_heads

        self.embed_conv = DepthwiseSeparableConv2d(in_channels, embed_channels, norm1="SyncBN", norm2="SyncBN", activation1=nn.GELU(), activation2=nn.GELU()) if embed_channels != in_channels else None
        # self.embed_conv = Conv2d(in_channels, embed_channels, 1, norm=get_norm("SyncBN", embed_channels))
        self.guide_fc = nn.Linear(guide_channels, embed_channels)
        self.offset = nn.Parameter(torch.zeros(1, num_heads, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_heads, 1, 1))

    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        B, _, H, W = x.shape
        guide = guide.unsqueeze(0).repeat(x.shape[0], 1, 1)
        guide = self.guide_fc(guide)
        guide = guide.reshape(B, -1, self.num_heads, self.head_channels)
        embed = self.embed_conv(x) if self.embed_conv is not None else x
        embed = embed.reshape(B, self.num_heads, self.head_channels, H, W)

        attn_weight = torch.einsum('bmchw,bnmc->bmhwn', embed, guide)
        attn_weight = attn_weight / (self.head_channels**0.5)
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = attn_weight.max(dim=-1)[0]
        attn_weight = attn_weight * self.scale + self.offset

        x = x.reshape(B, self.num_heads, -1, H, W)
        x = x * attn_weight.unsqueeze(2)
        x = x.reshape(B, -1, H, W)
        return x


class DyDepthwiseConvAtten(nn.Module):
    def __init__(self, cfg):
        super(DyDepthwiseConvAtten, self).__init__()
        self.hidden_dim = cfg.MODEL.EOV_SEG.HIDDEN_DIM
        self.num_proposals = cfg.MODEL.EOV_SEG.NUM_PROPOSALS
        self.kernel_size = cfg.MODEL.EOV_SEG.CONV_KERNEL_SIZE_1D  # 3

        # self.depth_weight_linear = nn.Linear(hidden_dim, kernel_size)
        # self.point_weigth_linear = nn.Linear(hidden_dim, num_proposals)
        self.weight_linear = nn.Linear(self.hidden_dim, self.kernel_size)
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, query, value):
        assert query.shape == value.shape
        B, N, C = query.shape

        # dynamic depth-wise conv
        # dy_depth_conv_weight = self.depth_weight_linear(query).view(B, self.num_proposals, 1,self.kernel_size) # B, N, 1, K
        # dy_point_conv_weight = self.point_weigth_linear(query).view(B, self.num_proposals, self.num_proposals, 1)
        dy_conv_weight = self.weight_linear(query).view(B, self.num_proposals, 1, self.kernel_size)
        # dy_depth_conv_weight = dy_conv_weight[:, :, :self.kernel_size].view(B,self.num_proposals,1,self.kernel_size)
        # dy_point_conv_weight = dy_conv_weight[:, :, self.kernel_size:].view(B,self.num_proposals,self.num_proposals,1)

        res = []
        value = value.unsqueeze(1)
        for i in range(B):
            # input: [1, N, C]
            # weight: [N, 1, K]
            # output: [1, N, C]
            out = F.conv1d(input=value[i], weight=dy_conv_weight[i], groups=N, padding="same")
            # input: [1, N, C]
            # weight: [N, N, 1]
            # output: [1, N, C]
            # out = F.conv1d(input=out, weight=dy_point_conv_weight[i], padding='same')
            res.append(out)
        point_out = torch.cat(res, dim=0)
        point_out = self.norm(point_out)
        return point_out


class TDEE(nn.Module):
    def __init__(self,
                 in_channels=256,
                 feat_channels=256,
                 out_channels=None,
                 router_sigmoid=True,
                 activate_out=False):
        super(TDEE, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels_raw = out_channels
        self.router_sigmoid = router_sigmoid
        self.activate_out = activate_out

        self.out_channels = out_channels if out_channels else in_channels

        self.num_params_in = self.feat_channels
        self.num_params_out = self.feat_channels
        self.dynamic_layer = nn.Linear(
            self.in_channels, self.num_params_in + self.num_params_out)
        self.input_layer = nn.Linear(self.in_channels,
                                     self.num_params_in + self.num_params_out,
                                     1)
        self.input_router = nn.Linear(self.in_channels, self.feat_channels, 1)
        self.update_router = nn.Linear(self.in_channels, self.feat_channels, 1)

        self.norm_in = nn.LayerNorm(self.feat_channels)
        self.norm_out = nn.LayerNorm(self.feat_channels)
        self.input_norm_in = nn.LayerNorm(self.feat_channels)
        self.input_norm_out = nn.LayerNorm(self.feat_channels)

        self.activation = nn.GELU()

        self.fc_layer = nn.Linear(self.feat_channels, self.out_channels, 1)
        self.fc_norm = nn.LayerNorm(self.out_channels)

    def forward(self, update_feature, input_feature):
        parameters = self.dynamic_layer(update_feature)
        param_in = parameters[..., :self.num_params_in]
        param_out = parameters[..., -self.num_params_out:]

        input_feats = self.input_layer(input_feature)
        input_in = input_feats[..., :self.num_params_in]
        input_out = input_feats[..., -self.num_params_out:]

        # P_t
        router_feats = input_in * param_in

        update_router = self.norm_in(self.update_router(router_feats))
        input_router = self.input_norm_in(self.input_router(router_feats))

        if self.router_sigmoid:
            update_router = update_router.sigmoid() # α_s
            input_router = input_router.sigmoid()  # α_m

        param_out = self.norm_out(param_out)
        input_out = self.input_norm_out(input_out)

        if self.activate_out:
            param_out = self.activation(param_out)
            input_out = self.activation(input_out)

        # ^E_I
        features = update_router * param_out + input_router * input_out

        features = self.fc_layer(features)
        features = self.fc_norm(features)
        features = self.activation(features)

        return features


class CrossAttenHead(nn.Module):
    def __init__(self, cfg, transformer_dim):
        super(CrossAttenHead, self).__init__()
        self.num_cls_fcs = cfg.MODEL.EOV_SEG.NUM_CLS_FCS
        self.num_mask_fcs = cfg.MODEL.EOV_SEG.NUM_MASK_FCS
        self.num_classes = cfg.MODEL.EOV_SEG.NUM_CLASSES
        self.conv_kernel_size_2d = cfg.MODEL.EOV_SEG.CONV_KERNEL_SIZE_2D

        self.hidden_dim = cfg.MODEL.EOV_SEG.HIDDEN_DIM
        self.num_proposals = cfg.MODEL.EOV_SEG.NUM_PROPOSALS
        self.hard_mask_thr = 0.5

        # Two-way Dynamic Embedding Experts(In the purple section of Fig. 4)
        self.tdee = TDEE(in_channels=cfg.MODEL.EOV_SEG.HIDDEN_DIM,
                                                feat_channels=cfg.MODEL.EOV_SEG.HIDDEN_DIM,
                                                out_channels=cfg.MODEL.EOV_SEG.HIDDEN_DIM,)

        self.f_atten = DyDepthwiseConvAtten(cfg)
        self.f_dropout = nn.Dropout(0.0)
        self.f_atten_norm = nn.LayerNorm(self.hidden_dim * self.conv_kernel_size_2d**2)

        self.k_atten = DyDepthwiseConvAtten(cfg)
        self.k_dropout = nn.Dropout(0.0)
        self.k_atten_norm = nn.LayerNorm(self.hidden_dim * self.conv_kernel_size_2d**2) 

        self.s_atten = nn.MultiheadAttention(embed_dim=self.hidden_dim * self.conv_kernel_size_2d**2,
                                             num_heads=8,
                                             dropout=0.0)
        self.s_dropout = nn.Dropout(0.0)
        self.s_atten_norm = nn.LayerNorm(self.hidden_dim * self.conv_kernel_size_2d**2)

        self.ffn = FFN(self.hidden_dim, feedforward_channels=2048)
        self.ffn_norm = nn.LayerNorm(self.hidden_dim)

        self.mask_fcs = nn.ModuleList()
        for _ in range(self.num_mask_fcs):
            self.mask_fcs.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=False))
            self.mask_fcs.append(nn.LayerNorm(self.hidden_dim))
            self.mask_fcs.append(nn.ReLU(True))
        self.fc_mask = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.mask_pooling = MaskPooling()
        self._mask_pooling_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim))

        self._spatial_mask_pooling_proj = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, self.hidden_dim))
        self.class_embed = MLP(self.hidden_dim, self.hidden_dim, cfg.MODEL.OV_HEAD.EMBED_DIM, 3)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, vit_feats, features, proposal_kernels, mask_preds, text_classifier=None, num_templates=None):
        """
        vit_feats: [B, C, H/4, W/4]
        features: [B, C, H/4, W/4]
        proposal_kernels: [B, N, C]
        mask_preds: [B, N, H, W]
        text_classifier: [num_classes, C]
        """
        # print(1, text_classifier.shape)
        B, C, H, W = features.shape

        soft_sigmoid_masks = mask_preds.sigmoid()
        nonzero_inds = soft_sigmoid_masks > self.hard_mask_thr
        hard_sigmoid_masks = nonzero_inds.float()  # [B, N, H, W]

        # pre-attention
        # [B, N, C]
        f = torch.einsum('bnhw,bchw->bnc', hard_sigmoid_masks, features)
        # [B, N, C, K, K] -> [B, N, C * K * K]
        k = proposal_kernels.view(B, self.num_proposals, -1)

        # ----
        f_tmp = self.f_atten(k, f)
        f = f + self.f_dropout(f_tmp)
        f = self.f_atten_norm(f)

        f_tmp = self.k_atten(k, f)
        f = f + self.k_dropout(f_tmp)
        k = self.k_atten_norm(f)
        # ----

        # [N, B, C]
        k = k.permute(1, 0, 2)

        k_tmp = self.s_atten(query=k, key=k, value=k)[0]
        k = k + self.s_dropout(k_tmp)
        k = self.s_atten_norm(k.permute(1, 0, 2))  # [B, N, C * K * K]

        obj_feat = self.ffn_norm(self.ffn(k))

        cls_feat = obj_feat  # [B, N, C]
        mask_feat = obj_feat

        for reg_layer in self.mask_fcs:
            mask_feat = reg_layer(mask_feat)
        mask_kernels = self.fc_mask(mask_feat)
        new_mask_preds = torch.einsum("bqc,bchw->bqhw", mask_kernels, features)

        maskpool_embeddings = self.mask_pooling(x=features, mask=new_mask_preds)
        maskpool_embeddings = self._mask_pooling_proj(maskpool_embeddings)

        spatial_maskpool_embeddings = self.mask_pooling(vit_feats, mask=new_mask_preds)
        spatial_maskpool_embeddings = self._spatial_mask_pooling_proj(spatial_maskpool_embeddings)
        maskpool_embeddings = self.tdee(maskpool_embeddings, spatial_maskpool_embeddings)

        class_embed = self.class_embed(maskpool_embeddings + cls_feat)
        cls_score = get_classification_logits(class_embed, text_classifier, self.logit_scale, num_templates)

        return cls_score, new_mask_preds, obj_feat


class LightweightDecoder(nn.Module):
    def __init__(self, cfg, vit_backbone_shape, num_stages, criterion):
        super(LightweightDecoder, self).__init__()
        transformer_dim = vit_backbone_shape["layer"].channels
        self.num_stages = num_stages
        self.criterion = criterion
        self.object_kernels = nn.Embedding(cfg.MODEL.EOV_SEG.NUM_PROPOSALS, cfg.MODEL.EOV_SEG.HIDDEN_DIM)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim, transformer_dim, kernel_size=2, stride=2),
            nn.GELU(),
        )

        # Vocabulary-Aware Selection(In the yellow section of Fig. 4)
        self.vas = VAS(in_channels=cfg.MODEL.EOV_SEG.HIDDEN_DIM, out_channels=cfg.MODEL.EOV_SEG.HIDDEN_DIM, guide_channels=cfg.MODEL.OV_HEAD.EMBED_DIM, embed_channels=cfg.MODEL.OV_HEAD.EMBED_DIM)
        
        self.mask_heads = nn.ModuleList()
        for _ in range(self.num_stages):
            self.mask_heads.append(CrossAttenHead(cfg, transformer_dim))

    def forward(self, vit_feats, features, targets, text_classifier=None, num_templates=None):
        features = self.vas(features, text_classifier)

        object_kernels = self.object_kernels.weight[None].repeat(features.shape[0], 1, 1)
        mask_preds = torch.einsum('bnc,bchw->bnhw', object_kernels, features)
        vit_feats = self.output_upscaling(vit_feats)

        all_cls_scores = []
        all_masks_preds = []
        all_stage_loss = {}

        # Lightweight Decoder - Layer(In the pink section of Fig. 4)
        for stage in range(self.num_stages):
            mask_head = self.mask_heads[stage]
            cls_scores, mask_preds, object_kernels = mask_head(vit_feats, features, object_kernels, mask_preds, text_classifier=text_classifier, num_templates=num_templates)
            all_cls_scores.append(cls_scores)
            all_masks_preds.append(mask_preds)

            if targets is not None:
                preds = {'pred_logits': cls_scores, 'pred_masks': mask_preds}
                single_stage_loss = self.criterion(preds, targets)
                for key, value in single_stage_loss.items():
                    all_stage_loss[f's{stage}_{key}'] = value

        return all_stage_loss, all_cls_scores, all_masks_preds
