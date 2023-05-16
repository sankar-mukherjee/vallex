import torch
import torch.nn as nn
from vallex.embedding import SinePositionalEmbedding, TokenEmbedding, PositionalEncoding
from typing import Iterator, Tuple, Union
from vallex.transformer import AdaptiveLayerNorm
from vallex.utils import make_pad_mask
import torch.nn.functional as F
import random

NUM_TEXT_TOKENS = 512
NUM_AUDIO_TOKENS = 1024  # EnCodec RVQ bins

# Define the model class
class VALLE(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        norm_first: bool = True
        ):
        super(VALLE, self).__init__()

        self.text_embedding = TokenEmbedding(d_model, NUM_TEXT_TOKENS)  # W_x
        self.ar_embedding = TokenEmbedding(d_model, NUM_AUDIO_TOKENS + 1)
        self.nar_embeddings = nn.ModuleList(
            [TokenEmbedding(d_model, NUM_AUDIO_TOKENS + 1)]
            + [TokenEmbedding(d_model, NUM_AUDIO_TOKENS) for i in range(7)]
        )  # W_a
        
        # PreNet
        self.text_prenet = nn.Identity()
        self.audio_prenet = nn.Identity()

        # TODO
        # self.text_position = SinePositionalEmbedding(d_model,dropout=0.1,scale=False)
        # self.audio_positions = nn.ModuleList([SinePositionalEmbedding(d_model,dropout=0.1,scale=False) for i in range(8)])
        self.text_position = PositionalEncoding(d_model,dropout=0.1)
        self.audio_positions = nn.ModuleList(
            [PositionalEncoding(d_model,dropout=0.1) for i in range(8)]
        )

        self.stage_embeddings = nn.ModuleList(
            [TokenEmbedding(d_model, 1) for i in range(8)]
        )

        self.ar_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model,
                num_heads,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=norm_first,
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model) if norm_first else None,
        )

        self.nar_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model,
                num_heads,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=norm_first,
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model) if norm_first else None,
            # TODO
            # norm=AdaptiveLayerNorm(d_model, norm=nn.LayerNorm(d_model)) if norm_first else None,
        )

        self.predict_layers = nn.ModuleList(
            [nn.Linear(d_model, NUM_AUDIO_TOKENS + 1, bias=False)]
            + [nn.Linear(d_model, NUM_AUDIO_TOKENS, bias=False) for i in range(7)]
        )
        # We share the parameters of the output projection layer with the parameters of the acoustic embedding Wa
        self.predict_layers[0].weight = self.ar_embedding.weight
        # We also share the parameters of the acoustic embedding layer and the output prediction layer,
        # which means the weights of the j-th prediction layer are the same as the (j + 1)-th acoustic embedding layer.
        for j in range(1, 7):
            self.predict_layers[j].weight = self.nar_embeddings[j + 1].weight
            
        self.rng = random.Random(0)
    
    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        y_lens: torch.Tensor,
        ):
        
        x = self.text_embedding(x)
        x = self.text_prenet(x)
        x = self.text_position(x)

        x_mask = make_pad_mask(x_lens).to(x.device)
        y_mask = make_pad_mask(y_lens).to(y.device)

        y_mask_int = y_mask.type(torch.int64)
        codes = y.type(torch.int64) * (1 - y_mask_int.unsqueeze(dim=-1))

        # AR Decoder
        y = F.pad(codes[..., 0], (0, 1), value=0) + NUM_AUDIO_TOKENS * F.pad(y_mask_int, (0, 1), value=1)
        y, ar_targets = y[:, :-1], y[:, 1:] # inputs, targets

        y_emb = self.ar_embedding(y)
        y_emb = self.audio_prenet(y_emb)
        y_pos = self.audio_positions[0](y_emb)
        y_len = y_lens.max()

        tgt_mask = torch.triu(
            torch.ones(y_len, y_len, device=y.device, dtype=torch.bool),
            diagonal=1
        )        
        y_dec = self.ar_decoder(
            tgt=y_pos,
            memory=x,
            tgt_mask=tgt_mask,
            memory_mask=None,
            tgt_key_padding_mask=y_mask,
            memory_key_padding_mask=x_mask,
        )
        ar_logits = self.predict_layers[0](y_dec).permute(0, 2, 1)

        # Non-AR Decoders
        # Random sampling one out of 7 quantizers
        train_stage = self.rng.choices(
            (1, 2, 3, 4, 5, 6, 7), weights=[1.0 / 7] * 7, k=1
        )[0]
        
        y_emb = self.nar_embeddings[0](y)
        for j in range(1, train_stage):
            # Formula (4) (5)
            y_emb = y_emb + self.nar_embeddings[j](codes[..., j])

        y_pos = self.audio_positions[train_stage](y_emb)
        nar_targets = codes[..., train_stage] + NUM_AUDIO_TOKENS * y_mask_int

        y_dec = self.nar_decoder(
            tgt=y_pos,
            memory=x,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=y_mask,
            memory_key_padding_mask=x_mask,
        )
        nar_logits = self.predict_layers[train_stage](y_dec).permute(0, 2, 1)

        return codes, ar_logits, ar_targets, nar_logits, nar_targets
