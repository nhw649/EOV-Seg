"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 
"""
import time
import torch
import torch.nn.functional as F
import math
from detectron2.utils import comm

import open_clip

from detectron2.modeling import Backbone, ShapeSpec
from .build import BACKBONE_REGISTRY


@BACKBONE_REGISTRY.register()
class CLIP(Backbone):
    def __init__(self, cfg, input_shape=None, backbone_type="cnn"):
        super().__init__()
        self.backbone_type = backbone_type
        if backbone_type == "cnn":
            model_name = cfg.MODEL.OV_HEAD.CLIP_MODEL_NAME
            pretrained = cfg.MODEL.OV_HEAD.CLIP_PRETRAINED_WEIGHTS
        elif backbone_type == "vit":
            model_name = cfg.MODEL.OV_HEAD.CLIP_VIT_MODEL_NAME
            pretrained = cfg.MODEL.OV_HEAD.CLIP_VIT_PRETRAINED_WEIGHTS
        # download on local rank 0 first
        if comm.get_local_rank() == 0:
            open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        comm.synchronize()

        self.model_name = model_name
        self.pretrained = pretrained

        self.clip_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.text_tokenizer = open_clip.get_tokenizer(model_name)

        model_name = model_name.lower()
        if 'convnext_' in model_name:
            self.model_type = 'convnext'
            if '_base' in model_name:
                self.output_channels = [128, 128, 256, 512, 1024]
            elif '_large' in model_name:
                self.output_channels = [192, 192, 384, 768, 1536]
            elif '_xxlarge' in model_name:
                self.output_channels = [384, 384, 768, 1536, 3072]

        elif 'rn' in model_name:
            self.model_type = 'resnet'
            if model_name.replace('-quickgelu', '') in ['rn50', 'rn101']:
                self.output_channels = [64, 256, 512, 1024, 2048]
            elif model_name == 'rn50x4':
                self.output_channels = [80, 320, 640, 1280, 2560]
            elif model_name == 'rn50x16':
                self.output_channels = [96, 384, 768, 1536, 3072]
            elif model_name == 'rn50x64':
                self.output_channels = [128, 512, 1024, 2048, 4096]

        elif 'vit' in model_name:
            self.model_type = 'vit'
            if 'b-32' in model_name:
                self.output_channels = 768
                self.output_strides = 32
            if 'b-16' in model_name:
                self.output_channels = 768
                self.output_strides = 16
            elif 'l-14' in model_name:
                self.output_channels = 1024
                self.output_strides = 14

        if backbone_type == "cnn":
            self._out_feature_strides = {
                "stem": 2,
                "res2": 4,
                "res3": 8,
                "res4": 16,
                "res5": 32,
                "clip_embedding": -1
            }
            self._out_feature_channels = {
                "stem": self.output_channels[0],
                "res2": self.output_channels[1],
                "res3": self.output_channels[2],
                "res4": self.output_channels[3],
                "res5": self.output_channels[4],
                "clip_embedding": self.dim_latent
            }
        elif backbone_type == "vit":
            self._out_feature_strides = {
                "layer": self.output_strides,
            }
            self._out_feature_channels = {
                "layer": self.output_channels,
            }

        self.eval()
        self.freeze_everything()

    def freeze_everything(self):
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.clip_model.transformer.get_cast_dtype()

        x = self.clip_model.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.clip_model.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x, attn_mask=self.clip_model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip_model.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    def tokenize_text(self, text):
        return self.text_tokenizer(text)

    def extract_features(self, x):
        return {
            'convnext': self.extract_features_convnext,
            'resnet': self.extract_features_resnet,
            'vit': self.extract_features_vit,
        }[self.model_type](x)

    def visual_prediction_forward(self, x, masks=None):
        return {
            'convnext': self.visual_prediction_forward_convnext,
            'resnet': self.visual_prediction_forward_resnet,
        }[self.model_type](x, masks)

    def extract_features_convnext(self, x):
        out = {}
        x = self.clip_model.visual.trunk.stem(x)
        out['stem'] = x.contiguous() # os4
        for i in range(4):
            x = self.clip_model.visual.trunk.stages[i](x)
            out[f'res{i+2}'] = x.contiguous() # res 2 (os4), 3 (os8), 4 (os16), 5 (os32)
        
        x = self.clip_model.visual.trunk.norm_pre(x)
        out['clip_vis_dense'] = x.contiguous()
        # out['clip_attnpool'] = self.clip_model.visual.attnpool(x)
        return out

    def global_attenpool(self, x):
        batch, channel, height, width = x.shape

        positional_embedding = self.clip_model.visual.attnpool.positional_embedding.to(x.dtype)
        spatial_pos_embed = positional_embedding[1:, None, :] # HW x 1 x C
        global_pos_embed = positional_embedding[:1, None, :] # HW x 1 x C

        orig_size = int(math.sqrt(spatial_pos_embed.shape[0]))
        spatial_pos_embed = spatial_pos_embed.permute(1, 2, 0).reshape(1, channel, orig_size, orig_size)
        spatial_pos_embed = F.interpolate(spatial_pos_embed, size=(height, width), mode='bilinear', align_corners=False) # 1 x C x H x W
        spatial_pos_embed = spatial_pos_embed.permute(2, 3, 0, 1).reshape(height*width, 1, channel)
        x = x.reshape(batch, channel, height * width).permute(2, 0, 1)  # BCHW -> (HW)BC
        x = x + spatial_pos_embed  # (HW)NC

        x = torch.cat([x.mean(dim=0, keepdim=True) + global_pos_embed, x], dim=0)
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.clip_model.visual.attnpool.num_heads,
            q_proj_weight=self.clip_model.visual.attnpool.q_proj.weight,
            k_proj_weight=self.clip_model.visual.attnpool.k_proj.weight,
            v_proj_weight=self.clip_model.visual.attnpool.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.clip_model.visual.attnpool.q_proj.bias,
                                    self.clip_model.visual.attnpool.k_proj.bias,
                                    self.clip_model.visual.attnpool.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.,
            out_proj_weight=self.clip_model.visual.attnpool.c_proj.weight,
            out_proj_bias=self.clip_model.visual.attnpool.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.clip_model.visual.attnpool.training,
            need_weights=False,
        )

        return x.permute(1, 0, 2)

    def extract_features_resnet(self, x):
        out = {}
        x = self.clip_model.visual.act1(self.clip_model.visual.bn1(self.clip_model.visual.conv1(x)))
        x = self.clip_model.visual.act2(self.clip_model.visual.bn2(self.clip_model.visual.conv2(x)))
        x = self.clip_model.visual.act3(self.clip_model.visual.bn3(self.clip_model.visual.conv3(x)))
        out['stem'] = x.contiguous() # os2
        x = self.clip_model.visual.avgpool(x)
        x = self.clip_model.visual.layer1(x)
        out['res2'] = x.contiguous() # os4
        x = self.clip_model.visual.layer2(x)
        out['res3'] = x.contiguous() # os8
        x = self.clip_model.visual.layer3(x)
        out['res4'] = x.contiguous() # os16
        x = self.clip_model.visual.layer4(x)
        out['res5'] = x.contiguous() # os32
        out['clip_vis_dense'] = x
        # out['clip_attnpool'] = self.global_attenpool(x)  # [B, HW+1, C]
        return out
    
    def extract_features_vit(self, x):
        x = self.clip_model.visual.conv1(x)  # shape = [*, width, grid, grid]
        H, W = x.shape[-2:]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self._expand_token(self.clip_model.visual.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        B, N, C = x.shape
        positional_embedding = self.clip_model.visual.positional_embedding.to(x.dtype)
        spatial_pos_embed = positional_embedding[1:, None, :]  # HW x 1 x C
        class_pos_embed = positional_embedding[:1, None, :]  # 1 x 1 x C
        orig_size = int(math.sqrt(spatial_pos_embed.shape[0]))
        # new_size = int(math.sqrt(x[:, 1:, :].shape[1]))
        spatial_pos_embed = spatial_pos_embed.permute(1, 2, 0).reshape(1, C, orig_size, orig_size)  # [1, C, H, W]
        spatial_pos_embed = F.interpolate(spatial_pos_embed, size=(H, W), mode='bilinear', align_corners=False)  # 1 x C x H x W
        spatial_pos_embed = spatial_pos_embed.permute(2, 3, 0, 1).reshape(H * W, 1, C)
        spatial_pos_embed = torch.cat([class_pos_embed, spatial_pos_embed], dim=0).permute(1, 0, 2)
        x = x + spatial_pos_embed  # B(HW)C

        x = self.clip_model.visual.ln_pre(x)

        # ablation for tab.3
        x = x.permute(1, 0, 2)  # [B, N, C] -> [N, B, C]
        # x = self.clip_model.visual.transformer(x)[1:]
        # x = self.clip_model.visual.transformer.resblocks[0](x)[1:]
        x = self.clip_model.visual.transformer.resblocks[0](x)[1:]
        # x = self.clip_model.visual.transformer.resblocks[1](x)
        # x = self.clip_model.visual.transformer.resblocks[2](x)
        # x = self.clip_model.visual.transformer.resblocks[3](x)
        # x = self.clip_model.visual.transformer.resblocks[4](x)
        # x = self.clip_model.visual.transformer.resblocks[5](x)[1:]
        # x = self.clip_model.visual.transformer.resblocks[6](x)
        # x = self.clip_model.visual.transformer.resblocks[7](x)
        # x = self.clip_model.visual.transformer.resblocks[8](x)
        # x = self.clip_model.visual.transformer.resblocks[9](x)
        # x = self.clip_model.visual.transformer.resblocks[10](x)
        # x = self.clip_model.visual.transformer.resblocks[11](x)[1:]
        x = x.permute(1, 2, 0).contiguous()  # [B, C, N]
        x = x.view(B, C, H, W)
        return x

    def visual_prediction_forward_convnext(self, x, masks):
        batch, num_query, channel = x.shape
        x = x.reshape(batch*num_query, channel, 1, 1) # fake 2D input
        x = self.clip_model.visual.trunk.head(x)
        x = self.clip_model.visual.head(x)
        return x.view(batch, num_query, x.shape[-1]) # B x num_queries x 640

    def visual_prediction_forward_resnet(self, x, masks):
        batch, channel, height, width = x.shape
        if masks.shape[-2] != height or masks.shape[-1] != width:
            masks = F.inteprolate(masks, size=(height, width), mode='bilinear', align_corners=False)
        num_masks = masks.shape[1]

        positional_embedding = self.clip_model.visual.attnpool.positional_embedding.to(x.dtype)
        spatial_pos_embed = positional_embedding[1:, None, :] # HW x 1 x C
        orig_size = int(math.sqrt(spatial_pos_embed.shape[0]))
        spatial_pos_embed = spatial_pos_embed.permute(1, 2, 0).reshape(1, channel, orig_size, orig_size)
        spatial_pos_embed = F.interpolate(spatial_pos_embed, size=(height, width), mode='bilinear', align_corners=False) # 1 x C x H x W
        spatial_pos_embed = spatial_pos_embed.permute(2, 3, 0, 1).reshape(height*width, 1, channel)
        x = x.reshape(batch, channel, height * width).permute(2, 0, 1)  # BCHW -> (HW)BC
        key_value = x + spatial_pos_embed

        masks = masks.reshape(batch, num_masks, height * width)
        masks = (masks > 0).to(masks.dtype)
        query = x.mean(0, keepdim=True) + positional_embedding[:1, None, :]
        num_masks = num_masks.to(query.device)
        query = query.repeat_interleave(num_masks, dim=0)

        attn_mask = masks < 0.5
        attn_mask = attn_mask.unsqueeze(1).expand(-1, self.clip_model.visual.attnpool.num_heads, -1, -1)
        attn_mask = attn_mask.reshape(batch * self.clip_model.visual.attnpool.num_heads,
                                    query.shape[0], key_value.shape[0])

        x = F.multi_head_attention_forward(
            query=query, key=key_value, value=key_value,
            embed_dim_to_check=key_value.shape[-1],
            num_heads=self.clip_model.visual.attnpool.num_heads,
            q_proj_weight=self.clip_model.visual.attnpool.q_proj.weight,
            k_proj_weight=self.clip_model.visual.attnpool.k_proj.weight,
            v_proj_weight=self.clip_model.visual.attnpool.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.clip_model.visual.attnpool.q_proj.bias,
                                    self.clip_model.visual.attnpool.k_proj.bias,
                                    self.clip_model.visual.attnpool.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.,
            out_proj_weight=self.clip_model.visual.attnpool.c_proj.weight,
            out_proj_bias=self.clip_model.visual.attnpool.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.clip_model.visual.attnpool.training,
            need_weights=False,
            attn_mask=attn_mask
        )[0].permute(1, 0, 2) # B x N x C
        return x

    def get_text_classifier(self, text_list, device):
        self.eval()
        with torch.no_grad():
            # reference for templates: https://github.com/mlfoundations/open_clip/blob/91f6cce16b7bee90b3b5d38ca305b5b3b67cc200/src/training/imagenet_zeroshot_data.py
            text_tokens = self.tokenize_text(text_list)
            text_tokens = text_tokens.to(device)
            # we return un-normalized text feature.
            text_features = self.encode_text(text_tokens, normalize=False)
            return text_features

    def forward(self, x):
        self.eval()
        with torch.no_grad():
            return self.extract_features(x)
    
    @property
    def dim_latent(self):
        return self.clip_model.text_projection.shape[-1]
    
    def output_shape(self):
        if self.backbone_type == "cnn":
            return {
                name: ShapeSpec(
                    channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
                )
                for name in ["stem", "res2", "res3", "res4", "res5", "clip_embedding"]
            }
        elif self.backbone_type == "vit":
            return {
                name: ShapeSpec(
                    channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
                )
                for name in ["layer"]
            }

    @property
    def size_divisibility(self):
        return -1

    def _expand_token(self, token, batch_size: int):
        return token.view(1, 1, -1).expand(batch_size, -1, -1)

    def get_similarity_map(sm, shape):
        # min-max norm
        sm = (sm - sm.min(1, keepdim=True)[0]) / (sm.max(1, keepdim=True)[0] - sm.min(1, keepdim=True)[0])

        # reshape
        if len(sm) == 3:
            side = int(sm.shape[1] ** 0.5)  # square output
            sm = sm.reshape(sm.shape[0], side, side, -1).permute(0, 3, 1, 2)

        # interpolate
        sm = torch.nn.functional.interpolate(sm, shape, mode='bilinear')
        sm = sm.permute(0, 2, 3, 1)

        return sm