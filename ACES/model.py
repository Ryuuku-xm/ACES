from torch import nn
import torch
from layers.Embed import DataEmbedding
from layers.Encoder import EncoderLayer


class HDSM(nn.Module):
    def __init__(self, configs, args):
        super(HDSM, self).__init__()
        self.task_name = args.task
        self.embed_type = args.embed
        self.configs = configs

        # Embedding
        self.embedding = DataEmbedding(self.embed_type)

        # n * EncoderLayer
        self.encoder = nn.ModuleList(
            [EncoderLayer() for _ in range(configs.n_layers)]
        )

        # group-cross-attention
        self.group_encoder = EncoderLayer()

        # Classification Layer
        self.predictor = nn.Linear(configs.d_model, 1)

    def forward(self, inputs):
        token_outputs_0 = self.embedding(inputs[:, :, :4]).permute(1, 0, 2)
        token_outputs_1 = self.embedding(inputs[:, :, 4:8]).permute(1, 0, 2)
        token_outputs_2 = self.embedding(inputs[:, :, 8:]).permute(1, 0, 2)

        mask_0 = self_attn_mask(inputs[:, :, :4])
        mask_1 = self_attn_mask(inputs[:, :, 4:8])
        mask_2 = self_attn_mask(inputs[:, :, 8:])

        # 自注意力
        for layer in self.encoder:
            token_outputs_0 = layer(token_outputs_0, token_outputs_0, mask_0)
            token_outputs_1 = layer(token_outputs_1, token_outputs_1, mask_1)
            token_outputs_2 = layer(token_outputs_2, token_outputs_2, mask_2)

        # 交叉注意力
        cross_output_1 = self.group_encoder(token_outputs_0, token_outputs_1, mask_0)
        cross_output_2 = self.group_encoder(token_outputs_0, token_outputs_2, mask_0)

        combined_output = (self.configs.alpha_ability * token_outputs_0 + (1 - self.configs.alpha_ability) / 2
                           * (cross_output_1 + cross_output_2))

        # TODO 拼接手工特征

        # Classification
        dimredu_outputs = combined_output.mean(dim=0)  # 降维

        logits = self.predictor(dimredu_outputs).squeeze(dim=1)

        predictions = torch.clamp(logits, 0, 14)

        return predictions


def self_attn_mask(inputs):
    seq_kq = inputs[:, :, 0]
    pad_attn_mask = seq_kq.data.eq(0)
    return pad_attn_mask
