import copy
import math

import torch
from torch import nn

from models.bricks.base_transformer import TwostageTransformer
from models.bricks.basic import MLP
from models.bricks.position_encoding import get_sine_pos_embed
from models.bricks.relation_transformer import (
    PositionRelationEmbedding,
)
from util.misc import inverse_sigmoid

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple, Dict

# 假设我们已经有了这些辅助模块
from models.bricks.basic import MLP, DropPath
from models.bricks.position_encoding import get_sine_pos_embed


class SsmTransformer(TwostageTransformer):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        num_classes: int,
        num_feature_levels: int = 4,
        two_stage_num_proposals: int = 300,
    ):
        super().__init__(num_feature_levels, encoder.embed_dim)
        # model parameters
        self.two_stage_num_proposals = two_stage_num_proposals
        self.num_classes = num_classes

        # model structure
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_class_head = nn.Linear(self.embed_dim, num_classes)
        self.encoder_bbox_head = MLP(self.embed_dim, self.embed_dim, 4, 3)
        self.pos_trans = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.pos_trans_norm = nn.LayerNorm(self.embed_dim)

        self.init_weights()

    def init_weights(self):
        # initilize encoder and hybrid classification layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.encoder_class_head.bias, bias_value)
        # initiailize encoder and hybrid regression layers
        nn.init.constant_(self.encoder_bbox_head.layers[-1].weight, 0.0)
        nn.init.constant_(self.encoder_bbox_head.layers[-1].bias, 0.0)

        # initialize pos_trans
        nn.init.xavier_uniform_(self.pos_trans.weight)

    def forward(
        self,
        multi_level_feats,
        multi_level_masks,
        multi_level_pos_embeds,
    ):
        # get input for encoder
        # [B, L, C]: L is the total number of pixels across all feature levels (L = sum(Hi*Wi) for all levels)
        feat_flatten = self.flatten_multi_level(multi_level_feats)
        # [B, L]
        mask_flatten = self.flatten_multi_level(multi_level_masks)
        # [B, L, C]: C is the embedding dimension of the feature map, the same as the input dimension of the encoder
        lvl_pos_embed_flatten = self.get_lvl_pos_embed(multi_level_pos_embeds)
        # spatial_shapes: [num_levels, 2]: Each row contains [Hi, Wi] for a specific feature level
        # level_start_index: [num_levels]: The starting index of each feature level in the flattened feature map
        # valid_ratios: [B, num_levels, 2]: The ratio of the feature map size to the original image size for each feature level
        spatial_shapes, level_start_index, valid_ratios = self.multi_level_misc(multi_level_masks)
        # reference_points: [B, L, num_levels, 2]: normalized (x, y) coordinates for each spatial position adjusted by valid ratios
        # proposals: [B, L, 4]
        reference_points, proposals = self.get_reference(spatial_shapes, valid_ratios)

        # encoder
        # memory: [B, L, C]
        memory = self.encoder(
            query=feat_flatten,
            query_pos=lvl_pos_embed_flatten,
            spatial_shapes=spatial_shapes,
            query_key_padding_mask=mask_flatten,
            level_start_index=level_start_index,
            reference_points=reference_points,
        )

        # get encoder output, classes and coordinates
        # output_memory: [B, L, C]
        # output_proposals: [B, L, 4], 经过逆sigmoid变换，无效位置设为无穷大
        output_memory, output_proposals = self.get_encoder_output(memory, proposals, mask_flatten)
        # enc_outputs_class: [B, L, num_classes]
        enc_outputs_class = self.encoder_class_head(output_memory)
        # enc_outputs_coord: [B, L, 4]
        enc_outputs_coord = self.encoder_bbox_head(output_memory) + output_proposals
        # enc_outputs_coord: [B, L, 4]
        enc_outputs_coord = enc_outputs_coord.sigmoid()

        # select topk
        topk = self.two_stage_num_proposals
        # 索引0很可能是一个特殊的"对象性"或"前景概率"通道，而不是特定类别的概率。这个通道被用来评估一个位置包含任何对象的可能性
        topk_index = torch.topk(enc_outputs_class[:, :, 0], topk, dim=1)[1].unsqueeze(-1)
        # topk_enc_outputs_coord: [B, topk, 4]
        topk_enc_outputs_coord = enc_outputs_coord.gather(1, topk_index.expand(-1, -1, 4))

        # get query(target) and reference points
        # NOTE: original implementation calculates query and query_pos together.
        # To keep the interface the same with Dab, DN and DINO, we split the
        # calculation of query_pos into the DeformableDecoder
        # reference_points: [B, topk, 4]
        reference_points = topk_enc_outputs_coord.detach()
        # nn.Linear can not perceive the arrangement order of elements
        # so exchange_xy=True/False does not matter results
        query_sine_embed = get_sine_pos_embed(
            reference_points, self.embed_dim // 2, exchange_xy=False
        )
        # [B, topk, self.embed_dim]
        target = self.pos_trans_norm(self.pos_trans(query_sine_embed))

        # decoder
        outputs_classes, outputs_coords = self.decoder(
            query=target,
            value=memory,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
        )

        return outputs_classes, outputs_coords, enc_outputs_class, enc_outputs_coord


class SsmTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.embed_dim = encoder_layer.embed_dim

        self.init_weights()

    def init_weights(self):
        # initialize encoder layers
        for layer in self.layers:
            if hasattr(layer, "init_weights"):
                layer.init_weights()

    def forward(
        self,
        query,
        spatial_shapes,
        level_start_index,
        reference_points,
        query_pos=None,
        query_key_padding_mask=None,
    ):
        for layer in self.layers:
            query = layer(
                query,
                query_pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                query_key_padding_mask,
            )

        return query


class SsmTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        d_ffn=1024,
        dropout=0.1,
        n_heads=8,
        activation=nn.ReLU(inplace=True),
        n_levels=4,
        n_points=4,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # self attention
        self.self_attn = MultiScaleDeformableAttention(embed_dim, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # ffn
        self.linear1 = nn.Linear(embed_dim, d_ffn)
        self.activation = activation
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, embed_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.init_weights()

    def init_weights(self):
        # initialize Linear layer
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, query):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(query))))
        query = query + self.dropout3(src2)
        query = self.norm2(query)
        return query

    def forward(
        self,
        query,
        query_pos,
        reference_points,
        spatial_shapes,
        level_start_index,
        query_key_padding_mask=None,
    ):
        # self attention
        src2 = self.self_attn(
            query=self.with_pos_embed(query, query_pos),
            reference_points=reference_points,
            value=query,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=query_key_padding_mask,
        )
        query = query + self.dropout1(src2)
        query = self.norm1(query)

        # ffn
        query = self.forward_ffn(query)

        return query


# class SsmTransformerDecoder(nn.Module):
#     def __init__(self, decoder_layer, num_layers, num_classes):
#         super().__init__()
#         # parameters
#         self.embed_dim = decoder_layer.embed_dim
#         self.num_heads = decoder_layer.num_heads
#         self.num_layers = num_layers
#         self.num_classes = num_classes

#         # decoder layers and embedding
#         self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
#         # NOTE: the ref_point_head of Deformable is split from pos_trans and pos_norm,
#         # which is different from DINO
#         self.ref_point_head = nn.Sequential(
#             nn.Linear(2 * self.embed_dim, self.embed_dim), nn.LayerNorm(self.embed_dim)
#         )

#         # iterative bounding box refinement
#         class_head = nn.Linear(self.embed_dim, num_classes)
#         bbox_head = MLP(self.embed_dim, self.embed_dim, 4, 3)
#         self.class_head = nn.ModuleList([copy.deepcopy(class_head) for _ in range(num_layers)])
#         self.bbox_head = nn.ModuleList([copy.deepcopy(bbox_head) for _ in range(num_layers)])

#         self.position_relation_embedding = PositionRelationEmbedding(16, self.num_heads)

#         self.init_weights()

#     def init_weights(self):
#         # initialize decoder layers
#         for layer in self.layers:
#             if hasattr(layer, "init_weights"):
#                 layer.init_weights()
#         # initialize decoder classification layers
#         prior_prob = 0.01
#         bias_value = -math.log((1 - prior_prob) / prior_prob)
#         for class_head in self.class_head:
#             nn.init.constant_(class_head.bias, bias_value)
#         # initiailize decoder regression layers
#         for bbox_head in self.bbox_head:
#             nn.init.constant_(bbox_head.layers[-1].weight, 0.0)
#             nn.init.constant_(bbox_head.layers[-1].bias, 0.0)

#         # initialize ref_point_head
#         nn.init.xavier_uniform_(self.ref_point_head[0].weight)

#     def forward(
#         self,
#         query,
#         reference_points,
#         value,
#         spatial_shapes,
#         level_start_index,
#         valid_ratios,
#         key_padding_mask=None,
#         attn_mask=None,
#     ):
#         # NOTE: the difference between DeformableDecoder and DabDecoder is that
#         # Deformable does not introduce reference refinement for query pos
#         query_sine_embed = get_sine_pos_embed(
#             reference_points, self.embed_dim // 2, exchange_xy=False
#         )
#         query_pos = self.ref_point_head(query_sine_embed)

#         outputs_classes, outputs_coords = [], []
#         valid_ratio_scale = torch.cat([valid_ratios, valid_ratios], -1)[:, None]

#         for layer_idx, layer in enumerate(self.layers):
#             reference_points_input = reference_points.detach()[:, :, None] * valid_ratio_scale

#             query = layer(
#                 query=query,
#                 query_pos=query_pos,
#                 reference_points=reference_points_input,
#                 value=value,
#                 spatial_shapes=spatial_shapes,
#                 level_start_index=level_start_index,
#                 key_padding_mask=key_padding_mask,
#                 self_attn_mask=attn_mask,
#             )

#             # get output
#             output_class = self.class_head[layer_idx](query)
#             output_coord = self.bbox_head[layer_idx](query) + inverse_sigmoid(reference_points)
#             output_coord = output_coord.sigmoid()
#             outputs_classes.append(output_class)
#             outputs_coords.append(output_coord)

#             if layer_idx == self.num_layers - 1:
#                 break

#             # NOTE: Here we integrate position_relation_embedding into DN-Deformable-DETR
#             src_boxes = tgt_boxes if layer_idx >= 1 else reference_points
#             tgt_boxes = output_coord
#             pos_relation = self.position_relation_embedding(src_boxes, tgt_boxes).flatten(0, 1)
#             if attn_mask is not None:
#                 pos_relation.masked_fill_(attn_mask, float("-inf"))

#             # iterative bounding box refinement
#             reference_points = output_coord.detach()

#         outputs_classes = torch.stack(outputs_classes)
#         outputs_coords = torch.stack(outputs_coords)
#         return outputs_classes, outputs_coords






class Box2DDistFun(nn.Module):
    """2D版本的边界框距离函数"""
    def __init__(self, out_dim=16):
        super().__init__()
        self.out_dim = out_dim
        # 使用MLP将相对位置映射到高维空间
        self.mlp = MLP(4, out_dim, out_dim, 2)  # 输入是4维：x,y相对位置和宽高比例

    def forward(self, key_pos, query_center, query_size, query_labels=None):
        """
        计算关键点相对于查询框的空间关系编码
        Args:
            key_pos: [B, L, 2] - 关键点位置 (x,y)
            query_center: [B, Q, 2] - 查询框中心点 (x,y)
            query_size: [B, Q, 2] - 查询框尺寸 (w,h)
            query_labels: [B, Q] - 查询框标签 (可选)
        Returns:
            dist_encoding: [B, L, Q, out_dim] - 距离编码
        """
        B, L, _ = key_pos.shape
        _, Q, _ = query_center.shape

        # 计算关键点到查询框中心的相对位置
        # [B, L, 1, 2] - [B, 1, Q, 2] = [B, L, Q, 2]
        rel_pos = key_pos.unsqueeze(2) - query_center.unsqueeze(1)

        # 归一化相对位置（除以查询框尺寸）
        # 避免除零
        eps = 1e-6
        query_size = query_size.clamp(min=eps)
        # [B, L, Q, 2] / [B, 1, Q, 2] = [B, L, Q, 2]
        rel_pos_norm = rel_pos / (query_size.unsqueeze(1) + eps)

        # 计算关键点是否在框内的比例值 (0-1之间的值)
        # 计算关键点到框边界的距离
        dist_to_border = torch.abs(rel_pos_norm)
        in_box = (dist_to_border <= 0.5).all(dim=-1, keepdim=True).float()

        # 组合特征: [相对位置(归一化), 在框内的指示]
        # [B, L, Q, 4]
        combined_features = torch.cat([rel_pos_norm, in_box, 1.0-in_box], dim=-1)

        # 使用MLP映射到高维空间
        # [B, L, Q, out_dim]
        dist_encoding = self.mlp(combined_features)

        return dist_encoding


def _get_activation_fn(activation):
    """获取激活函数"""
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "silu":
        return F.silu
    else:
        raise RuntimeError(f"activation should be relu/gelu/silu, not {activation}")


class SsmTransformerDecoder(nn.Module):
    """SSM Transformer解码器"""
    def __init__(
        self,
        decoder_layer,
        num_layers,
        num_classes,
        serialization_strategies=None  # 新增：序列化策略列表
    ):
        super().__init__()
        # parameters
        self.embed_dim = decoder_layer.embed_dim
        self.num_heads = decoder_layer.num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes

        # decoder layers and embedding
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.layers[-1].last_layer = True # 最后一个解码器层

        # 设置序列化策略
        if serialization_strategies is None:
            # 默认策略：所有层使用相同的序列化方式
            self.serialization_strategies = ['default'] * num_layers
        else:
            assert len(serialization_strategies) == num_layers, "序列化策略数量必须与层数相同"
            self.serialization_strategies = serialization_strategies
        # 为每一层设置序列化策略
        for i, layer in enumerate(self.layers):
            layer.serialization_strategy = self.serialization_strategies[i]

        # NOTE: the ref_point_head of Deformable is split from pos_trans and pos_norm,
        # which is different from DINO
        self.ref_point_head = nn.Sequential(
            nn.Linear(2 * self.embed_dim, self.embed_dim), nn.LayerNorm(self.embed_dim)
        )

        # iterative bounding box refinement
        class_head = nn.Linear(self.embed_dim, num_classes)
        bbox_head = MLP(self.embed_dim, self.embed_dim, 4, 3)
        self.class_head = nn.ModuleList([copy.deepcopy(class_head) for _ in range(num_layers)])
        self.bbox_head = nn.ModuleList([copy.deepcopy(bbox_head) for _ in range(num_layers)])

        self.init_weights()

    def init_weights(self):
        # initialize decoder layers
        for layer in self.layers:
            if hasattr(layer, "init_weights"):
                layer.init_weights()
        # initialize decoder classification layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for class_head in self.class_head:
            nn.init.constant_(class_head.bias, bias_value)
        # initiailize decoder regression layers
        for bbox_head in self.bbox_head:
            nn.init.constant_(bbox_head.layers[-1].weight, 0.0)
            nn.init.constant_(bbox_head.layers[-1].bias, 0.0)

        # initialize ref_point_head
        nn.init.xavier_uniform_(self.ref_point_head[0].weight)

    def forward(
        self,
        query,              # [B, Q, C] - 查询
        reference_points,   # [B, Q, 4] - 参考点
        value,              # [B, L, C] - 值
        spatial_shapes,    # [num_levels, 2] - 空间形状
        level_start_index, # [num_levels] - 层级起始索引
        valid_ratios,      # [B, num_levels, 2] - 有效比例
        key_padding_mask=None,  # [B, L] - 键填充掩码
    ):
        # NOTE: the difference between DeformableDecoder and DabDecoder is that
        # Deformable does not introduce reference refinement for query pos
        query_sine_embed = get_sine_pos_embed(
            reference_points, self.embed_dim // 2, exchange_xy=False
        )
        query_pos_embed = self.ref_point_head(query_sine_embed)

        outputs_classes, outputs_coords = [], []
        # valid_ratio_scale: [B, 1, num_levels, 4]
        valid_ratio_scale = torch.cat([valid_ratios, valid_ratios], -1)[:, None]

        # 从空间形状计算位置信息
        # 为每个特征点生成归一化坐标
        memory_pos = self.get_memory_pos(spatial_shapes, level_start_index, value.device)

        for layer_idx, layer in enumerate(self.layers):
            # reference_points_input: [B, Q, num_levels, 4]
            reference_points_input = reference_points.detach()[:, :, None] * valid_ratio_scale

            # 通过解码器层
            query, value = layer(
                query=query,
                memory=value,
                query_pos_embed=query_pos_embed,
                memory_pos=memory_pos,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                memory_key_padding_mask=key_padding_mask,
                layer_idx=layer_idx,
            )

            # get output
            output_class = self.class_head[layer_idx](query)
            output_coord = self.bbox_head[layer_idx](query) + inverse_sigmoid(reference_points)
            output_coord = output_coord.sigmoid()
            outputs_classes.append(output_class)
            outputs_coords.append(output_coord)

            if layer_idx == self.num_layers - 1:
                break

            # iterative bounding box refinement
            reference_points = output_coord.detach()

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        return outputs_classes, outputs_coords

    def get_memory_pos(self, spatial_shapes, level_start_index, device):
        """
        为每个特征点生成归一化坐标

        Args:
            spatial_shapes: [num_levels, 2] - 每个特征层的空间形状 (H, W)
            level_start_index: [num_levels] - 每个特征层在展平特征中的起始索引
            device: 设备

        Returns:
            memory_pos: [B, L, 2] - 每个特征点的归一化坐标 (x, y)
        """
        num_levels = spatial_shapes.shape[0]

        # 为每个特征层生成网格坐标
        level_pos_list = []
        for level in range(num_levels):
            H, W = spatial_shapes[level]

            # 生成网格坐标
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij'
            )

            # 归一化坐标到 [0, 1]
            grid_x = (grid_x + 0.5) / W
            grid_y = (grid_y + 0.5) / H

            # 展平并堆叠
            pos = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
            level_pos_list.append(pos)

        # 连接所有特征层的位置
        memory_pos = torch.cat(level_pos_list, dim=0)

        # 扩展批次维度 (假设批次大小为1，实际使用时会被广播)
        memory_pos = memory_pos.unsqueeze(0)

        return memory_pos


class SsmTransformerDecoderLayer(nn.Module):
    """SSM Transformer解码器层"""
    def __init__(
        self,
        d_model=256,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        num_proposal=300, # 300 two_stage_num_proposals, and ssm_chunk_size
        ssm_expand=2,
        ssm_use_biscan=True,
        last_layer=False
    ):
        super().__init__()
        self.d_model = d_model # embed_dim
        self.nhead = nhead
        self.last_layer = last_layer
        self.weight_dist = -0.1  # 距离权重衰减系数
        self.serialization_strategy = 'default'  # 默认序列化策略

        # 自注意力层
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # 交叉注意力层 (可选择使用SSM或标准交叉注意力)
        self.ssm = MultiHead2DISSM(
            d_model=d_model,
            d_state=num_proposal,
            d_dist=16,  # 距离编码维度
            chunk_size=num_proposal,
            nheads=nhead,
            expand=ssm_expand,
            use_biscan=ssm_use_biscan,
            dropout=dropout,
        )
        self.spatial_dist = Box2DDistFun(out_dim=16)

        # memory: 残差连接和FFN处理
        self.dropout2_memory = nn.Dropout(dropout)
        self.norm2_memory = nn.LayerNorm(d_model)
        # 前馈网络
        self.linear1_memory = nn.Linear(d_model, dim_feedforward)
        self.dropout_memory = nn.Dropout(dropout)
        self.linear2_memory = nn.Linear(dim_feedforward, d_model)
        self.dropout3_memory = nn.Dropout(dropout)
        self.norm3_memory = nn.LayerNorm(d_model)

        # query: 残差连接和FFN处理
        self.dropout2_query = nn.Dropout(dropout)
        self.norm2_query = nn.LayerNorm(d_model)
        # 前馈网络
        self.linear1_query = nn.Linear(d_model, dim_feedforward)
        self.dropout_query = nn.Dropout(dropout)
        self.linear2_query = nn.Linear(dim_feedforward, d_model)
        self.dropout3_query = nn.Dropout(dropout)
        self.norm3_query = nn.LayerNorm(d_model)

        # 激活函数
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.init_weights()

    def init_weights(self):
        """初始化权重"""
        # initialize self_attention
        nn.init.xavier_uniform_(self.self_attn.in_proj_weight)
        nn.init.xavier_uniform_(self.self_attn.out_proj.weight)
        # initialize Linear layer
        nn.init.xavier_uniform_(self.linear1_memory.weight)
        nn.init.xavier_uniform_(self.linear2_memory.weight)
        nn.init.xavier_uniform_(self.linear1_query.weight)
        nn.init.xavier_uniform_(self.linear2_query.weight)

    def with_pos_embed(self, tensor, pos):
        """添加位置编码"""
        return tensor if pos is None else tensor + pos

    def forward_ffn_memory(self, x):
        """前馈网络前向传播"""
        x2 = self.linear2_memory(self.dropout_memory(self.activation(self.linear1_memory(x))))
        x = x + self.dropout3_memory(x2)
        x = self.norm3_memory(x)
        return x

    def forward_ffn_query(self, x):
        """前馈网络前向传播"""
        x2 = self.linear2_query(self.dropout_query(self.activation(self.linear1_query(x))))
        x = x + self.dropout3_query(x2)
        x = self.norm3_query(x)
        return x

    def local_weight(self, memory_pos, query_center, query_size, strategy='default'):
        """
        计算2D版本的局部权重，支持不同的序列化策略

        Args:
            memory_pos: [B, L, 2] - 关键点位置
            query_center: [B, Q, 2] - 查询框中心
            query_size: [B, Q, 2] - 查询框尺寸
            strategy: 序列化策略

        Returns:
            weights: [B, L, Q] - 权重矩阵
        """
        B, L, _ = memory_pos.shape
        _, Q, _ = query_center.shape

        # 计算查询框的半径（对角线长度的一半）
        query_radius = torch.sqrt(torch.sum(query_size**2, dim=-1) / 4).clamp_min(16.0)  # 最小半径16像素

        # 计算关键点到查询中心的距离
        dist = torch.cdist(memory_pos, query_center, p=2)  # [B, L, Q]

        # 根据不同策略计算权重
        if strategy == 'default' or strategy == 'distance':
            # 基于距离的衰减权重
            weights = torch.exp(self.weight_dist * ((dist - query_radius.unsqueeze(1)).clamp_min(0.0)))

        elif strategy == 'attention':
            # 基于注意力的权重（使用softmax归一化）
            # 距离越小，注意力权重越大
            attention_scores = -dist / query_radius.unsqueeze(1)
            weights = F.softmax(attention_scores, dim=1)

        elif strategy == 'topk':
            # 只保留每个查询的Top-K个最近点
            k = min(L, 64)  # 可以根据需要调整K值
            # 对每个查询找到最近的K个点
            _, indices = torch.topk(-dist, k=k, dim=1)  # [B, k, Q]
            weights = torch.zeros_like(dist)

            # 为每个批次和查询设置权重
            for b in range(B):
                for q in range(Q):
                    weights[b, indices[b, :, q], q] = 1.0

        elif strategy == 'hybrid':
            # 混合策略：结合距离衰减和Top-K
            k = min(L, 128)  # 可以根据需要调整K值
            _, indices = torch.topk(-dist, k=k, dim=1)  # [B, k, Q]

            # 先用距离衰减计算权重
            weights = torch.exp(self.weight_dist * ((dist - query_radius.unsqueeze(1)).clamp_min(0.0)))

            # 然后只保留Top-K个点的权重
            mask = torch.zeros_like(weights)
            for b in range(B):
                for q in range(Q):
                    mask[b, indices[b, :, q], q] = 1.0

            weights = weights * mask

        else:
            # 默认回退到距离衰减
            weights = torch.exp(self.weight_dist * ((dist - query_radius.unsqueeze(1)).clamp_min(0.0)))

        return weights

    def get_serialization_order(self, memory_pos, query_center, strategy='default'):
        """
        根据不同策略获取序列化顺序

        Args:
            memory_pos: [B, L, 2] - 记忆位置
            query_center: [B, Q, 2] - 查询中心
            strategy: 序列化策略

        Returns:
            indices: [B, L] - 序列化顺序索引
        """
        B, L, _ = memory_pos.shape
        _, Q, _ = query_center.shape

        if strategy == 'default' or strategy == 'raster':
            # 默认光栅顺序（保持原始顺序）
            return torch.arange(L, device=memory_pos.device).unsqueeze(0).expand(B, -1)

        elif strategy == 'spiral':
            # 螺旋序列化：从图像中心向外螺旋
            # 计算每个点到图像中心的距离
            image_center = torch.tensor([0.5, 0.5], device=memory_pos.device).view(1, 1, 2)
            dist_to_center = torch.norm(memory_pos - image_center, dim=2)  # [B, L]

            # 按距离排序
            _, indices = torch.sort(dist_to_center, dim=1)
            return indices

        elif strategy == 'query_centered':
            # 以查询为中心的序列化
            # 对于每个批次，计算所有记忆点到所有查询点的平均距离
            dist = torch.cdist(memory_pos, query_center, p=2)  # [B, L, Q]
            avg_dist = dist.mean(dim=2)  # [B, L]

            # 按平均距离排序
            _, indices = torch.sort(avg_dist, dim=1)
            return indices

        elif strategy == 'zigzag':
            # Z字形扫描
            # 假设记忆点是按照光栅顺序排列的
            # 我们可以根据原始索引重新排列
            indices = torch.arange(L, device=memory_pos.device).unsqueeze(0).expand(B, -1)

            # 获取原始图像的高度和宽度（假设是正方形）
            side_len = int(math.sqrt(L))

            # 创建Z字形扫描索引
            zigzag_indices = torch.zeros(L, device=memory_pos.device, dtype=torch.long)
            idx = 0
            for i in range(side_len):
                if i % 2 == 0:  # 从左到右
                    for j in range(side_len):
                        if idx < L:
                            zigzag_indices[idx] = i * side_len + j
                            idx += 1
                else:  # 从右到左
                    for j in range(side_len-1, -1, -1):
                        if idx < L:
                            zigzag_indices[idx] = i * side_len + j
                            idx += 1

            # 应用Z字形扫描索引
            return zigzag_indices.unsqueeze(0).expand(B, -1)

        else:
            # 默认光栅顺序
            return torch.arange(L, device=memory_pos.device).unsqueeze(0).expand(B, -1)

            # query, value = layer(
            #     query=query,
            #     memory=value,
            #     query_pos_embed=query_pos_embed,
            #     memory_pos=memory_pos,
            #     reference_points=reference_points_input,
            #     spatial_shapes=spatial_shapes,
            #     level_start_index=level_start_index,
            #     memory_key_padding_mask=key_padding_mask,
            #     layer_idx=layer_idx,
            # )

    # TODO: 需要修改
    def forward(
        self,
        query,              # [B, Q, D] - 目标查询
        memory,             # [B, L, D] - 编码器记忆
        query_pos_embed,     # [B, Q, D] - 查询位置编码
        memory_pos,         # [B, L, 2] - 记忆位置编码
        reference_points,   # [B, Q, num_levels, 2] - 参考点 (x, y)
        spatial_shapes,     # [num_levels, 2] - 空间形状
        level_start_index,  # [num_levels] - 层级起始索引
        memory_key_padding_mask=None,  # [B, L] - 记忆键填充掩码
        layer_idx=None,     # 层索引，用于选择不同的序列化策略
    ):
        # 自注意力
        query_with_pos = key_with_pos = self.with_pos_embed(query, query_pos_embed)
        query2 = self.self_attn(
            query=query_with_pos,
            key=key_with_pos,
            value=query,
            need_weights=False,
        )[0]
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        # 提取参考点的中心和尺寸
        # 对于多尺度特征，取第一个级别的参考点
       # If reference_points has shape [B, Q, 4] with format (x,y,w,h)
        center_points = reference_points[..., 0, :2]  # Extract (x,y) -> [B, Q, 2]
        box_sizes = reference_points[..., 0, 2:]      # Extract (w,h) -> [B, Q, 2]

        # 根据层索引选择序列化策略
        if layer_idx is not None:
            # 不同层可以使用不同的序列化策略
            strategies = ['default', 'spiral', 'query_centered', 'zigzag']
            strategy = strategies[layer_idx % len(strategies)]
        else:
            strategy = self.serialization_strategy

        # 获取序列化顺序
        # serialization_indices: [B, L] - 序列化顺序索引
        serialization_indices = self.get_serialization_order(memory_pos, center_points, strategy)

        # 根据序列化顺序重排记忆和位置
        B = memory.shape[0]
        reordered_memory = torch.zeros_like(memory)
        reordered_memory_pos = torch.zeros_like(memory_pos)

        for b in range(B):
            reordered_memory[b] = memory[b, serialization_indices[b]]
            reordered_memory_pos[b] = memory_pos[b, serialization_indices[b]]

        # 计算局部权重 (使用重排后的位置)
        weights = self.local_weight(reordered_memory_pos, center_points, box_sizes, strategy)

        # 计算空间距离编码 (使用重排后的位置)
        dist = self.spatial_dist(
            key_pos=reordered_memory_pos,
            query_center=center_points,
            query_size=box_sizes,
        )

        # 应用SSM
        memory2, query2 = self.ssm(
            in_key=reordered_memory,
            in_query=query,
            dist=dist,
            key_pos=reordered_memory_pos,
            mask=weights,
        )

        # 将处理后的记忆恢复原始顺序
        restored_memory2 = torch.zeros_like(memory2)
        for b in range(B):
            # 创建反向索引映射
            reverse_indices = torch.zeros_like(serialization_indices[b])
            reverse_indices[serialization_indices[b]] = torch.arange(memory.shape[1], device=memory.device)
            restored_memory2[b] = memory2[b, reverse_indices]

        memory2 = restored_memory2

        # 残差连接和FFN处理
        if not self.last_layer:
            # 场景点特征的残差连接和FFN
            memory = memory + self.dropout2_memory(memory2)
            memory = self.norm2_memory(memory)
            memory = self.forward_ffn_memory(memory)

        # 查询特征的残差连接和FFN
        query = query + self.dropout2_query(query2)
        query = self.norm2_query(query)
        query = self.forward_ffn_query(query)

        return query, memory


from issm_triton.issm_combined import ISSM_chunk_scan_combined
from issm_triton.layernorm_gated import RMSNorm as RMSNormGated

class MultiHead2DISSM(nn.Module):
    """2D版本的多头ISSM扫描模块，使用ISSM_chunk_scan_combined"""
    def __init__(
        self,
        d_model: int = 256,        # 输入维度
        d_state: int = 64,         # 状态维度 same as num_proposal
        d_dist: int = 16,          # 距离编码维度
        chunk_size: int = 256,     # 块大小 same as num_proposal
        nheads: int = 8,           # 注意力头数
        ngroups: int = 1,          # 组数
        expand: int = 2,           # 扩展因子
        use_biscan: bool = True,   # 是否使用双向扫描
        A_init_range=(1, 16),      # A matrix initialization range
        dt_min: float = 0.0001,    # Minimum time step
        dt_max: float = 0.1,       # Maximum time step
        dt_init_floor: float = 1e-4,# Time step initialization lower bound
        dt_limit: Tuple[float, float] = (0.0, float("inf")),  # dt限制范围
        layer_idx=None,
    ):
        super().__init__()

        # 基本参数
        self.d_model = d_model
        self.d_state = d_state
        self.d_dist = d_dist
        self.chunk_size = chunk_size
        self.nheads = nheads
        self.ngroups = ngroups
        self.expand = expand
        self.use_biscan = use_biscan
        self.d_inner = self.expand * self.d_model
        self.headdim = self.d_inner // self.nheads
        self.dt_limit = dt_limit
        self.layer_idx = layer_idx

        # 投影层
        # 输入投影: 特征 -> [z, x, b/c偏置, dt偏置]
        d_in_key_proj = 2 * self.d_inner + 2 * self.ngroups + self.nheads
        self.key_proj = nn.Linear(self.d_model, d_in_key_proj, bias=False)
        # 查询投影: 特征 -> 初始状态
        self.query_proj = nn.Linear(self.d_model, self.d_inner, bias=False)
        # 距离编码投影: 距离 -> B/C基础值
        self.bc_proj = nn.Linear(self.d_dist, 2 * self.ngroups, bias=False)
        # 距离编码投影: 距离 -> dt基础值
        self.dt_proj = nn.Linear(self.d_dist, self.nheads, bias=False)

        # 使用原始DEST的dt初始化方法
        # 初始化dt偏置 (时间步长偏置)
        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # Initialize state transition parameters
        # 状态空间参数
        # 初始化A矩阵 (状态转移矩阵)
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty((self.nheads), dtype=torch.float32).uniform_(*A_init_range)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        # D "skip" parameter  初始化D矩阵 (跳跃连接)
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True

        # 输出投影
        self.out_key_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        self.out_query_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        # 归一化层
        self.key_norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False)
        self.query_norm = nn.LayerNorm(self.d_inner)


    def forward(self, in_key, in_query, dist, key_pos=None, mask=None):
        """
        前向传播函数
        Args:
            in_key: [B, L, D] - 输入序列特征 input
            in_query: [B, Q, D] - 查询序列特征 state
            dist: [B, L, Q, M] - 距离编码矩阵 M: 距离编码维度
            key_pos: [B, L, 2] - 关键点位置 (可选)
            mask: [B, L, Q] - 掩码矩阵 (可选)
        Returns:
            out_key: [B, L, D] - 处理后的关键点特征
            out_query: [B, Q, D] - 处理后的查询点特征
        """
        batch, seq_len, _ = in_key.shape
        _, num_queries, _ = in_query.shape

        # 1. 投影变换
        # [batch_size, seq_len, 2*self.d_inner + 2*self.ngroups + self.nheads]
        zxbcdt = self.key_proj(in_key)
        # z: [batch_size, seq_len, self.d_inner]
        # xbc: [batch_size, seq_len, self.d_inner + 2 * self.ngroups]
        # dt: [batch_size, seq_len, self.nheads]
        z, xbc, dt_bias = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.ngroups, self.nheads],
            dim=-1
        )

        # 分离状态和偏置
        # x: [batch_size, seq_len, self.d_inner]
        # b_bias: [batch_size, seq_len, self.ngroups]
        # c_bias: [batch_size, seq_len, self.ngroups]
        x, b_bias, c_bias = torch.split(
            xbc,
            [self.d_inner, self.ngroups, self.ngroups],
            dim=-1
        )

        # 如果使用双向扫描，准备反向数据
        if self.use_biscan:
            # might need to add extra conv layer to improve performance
            x_back = x.clone()
            b_bias_back = b_bias.clone()
            c_bias_back = c_bias.clone()

        # 处理查询特征 - 这是关键区别
        # 将查询特征投影为初始状态
        # initial_states: [batch_size, num_queries, self.d_inner]
        initial_states = self.query_proj(in_query)

        # 重排维度为 [B, H, D, Q] - 注意这里与原始DEST保持一致
        # 每个查询点都有自己的初始状态，作为一个整体传入扫描函数
        # initial_states: [batch_size, nheads, headdim, num_queries]
        initial_states = rearrange(initial_states, "b q (h d) -> b h d q", h=self.nheads)

        # 2. 生成状态空间模型参数
        # 状态转移矩阵A
        A = -torch.exp(self.A_log)  # [H]
        A = repeat(A, "h -> h d", d=self.d_state)  # [H, D]

        # 从距离编码生成B和C矩阵
        bc = self.bc_proj(dist)  # [B, L, Q, 2*ngroups]
        b_base, c_base = torch.split(bc, [self.ngroups, self.ngroups], dim=-1)

        # 组合基础值和偏置
        # 注意这里的维度变换，使B和C的形状为 [batch, seqlen, nheads/ngroups, dstate]
        # it is equivalent to 3D DEST version
        B = b_base + b_bias.unsqueeze(2)  # [B, L, Q, ngroups]
        C = c_base + c_bias.unsqueeze(2)  # [B, L, Q, ngroups]
        B = B.permute(0, 1, 3, 2)  # [B, L, ngroups, Q]
        C = C.permute(0, 1, 3, 2)  # [B, L, ngroups, Q]

        # 如果使用双向扫描，也生成反向参数
        if self.use_biscan:
            B_back = b_base + b_bias_back.unsqueeze(2)
            C_back = c_base + c_bias_back.unsqueeze(2)
            B_back = B_back.permute(0, 1, 3, 2)
            C_back = C_back.permute(0, 1, 3, 2)

        # 生成时间步长
        dt_base = self.dt_proj(dist)  # [B, L, Q, nheads]
        dt_base = dt_base.permute(0, 1, 3, 2)  # [B, L, nheads, Q]
        # 结合两种偏置计算最终的时间步长 dt [B, L, nheads, Q]
        dt = F.softplus(dt_base + dt_bias.unsqueeze(-1) + self.dt_bias.reshape(1, 1, -1, 1))

        # 8. 应用mask（如果提供）
        if mask != None:
            if mask.dtype == torch.float32:
                dt = dt * mask.unsqueeze(2)
            else:
                dt[mask.unsqueeze(2).repeat(1, 1, self.nheads, 1)] = 0.0

        # 3. 执行扫描 - 一次性处理所有查询
        # 注意这里与原始DEST保持一致，不需要循环处理每个查询
        module_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        module_kwargs["return_final_states"] = True
        # 执行扫描
        key, last_states = self.scan(x, initial_states, dt, A, B, C, module_kwargs)

        # 如果使用双向扫描
        if self.use_biscan:
            # 反转序列
            x_back = torch.flip(x_back, dims=[1])
            dt_back = torch.flip(dt, dims=[1])
            B_back = torch.flip(B_back, dims=[1])
            C_back = torch.flip(C_back, dims=[1])
            # 执行反向扫描
            key_back, last_states_back = self.scan(x_back, initial_states, dt_back, A, B_back, C_back, module_kwargs)
            # 反转回来并平均
            key_back = torch.flip(key_back, dims=[1])
            key = (key + key_back) / 2
            last_states = (last_states + last_states_back) / 2

        # 4. 输出处理
        # 重排key的维度并应用归一化
        key = rearrange(key, "b l h d -> b l (h d)")
        key = self.key_norm(key, z)
        out_key = self.out_key_proj(key)

        # 处理最终状态 - 注意这里的维度变换
        last_states = rearrange(last_states, "b h d q -> b q (h d)")
        last_states = self.query_norm(last_states)
        out_query = self.out_query_proj(last_states)

        return out_key, out_query

    def scan(self, x, initial_states, dt, A, B, C, module_kwargs):
        """
        Perform unidirectional or bidirectional scan
        Args:
            x: (B, L, D) - Input sequence
            initial_states: (B, K, D) - Initial states
            dt: (B, L, nheads) - Time steps
            A, B, C: Parameters for the scan
            module_kwargs: Additional parameters
        Returns:
            y: (B, K, D) - Output sequence
            last_states: (B, K, D) - Final states
        """
        # 对于每个时间步 t：
        # h[t] = h[t-1] + (Ah[t-1] + Bx[t])dt  # 状态更新
        # y[t] = Ch[t]                         # 输出计算
        y, last_states = ISSM_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            A,
            B,
            C,
            chunk_size=self.chunk_size,
            D=self.D,
            z=None,
            initial_states=initial_states,
            **module_kwargs,
        )
        return y, last_states