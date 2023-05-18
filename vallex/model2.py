import random
from typing import Iterator, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from vallex.embedding import (PositionalEncoding, SinePositionalEmbedding,
                              TokenEmbedding)
from vallex.transformer import (AdaptiveLayerNorm, LayerNorm,
                                TransformerDecoderLayer,
                                TransformerEncoderLayer)
from vallex.utils import make_pad_mask, topk_sampling

NUM_TEXT_TOKENS = 512
NUM_AUDIO_TOKENS = 1024  # EnCodec RVQ bins

# Define the model class
class VALLE(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        norm_first: bool = True,
        decoder_cls: Union[
            nn.TransformerDecoder, nn.TransformerEncoder
        ] = nn.TransformerDecoder,
        decoder_layer_cls: Union[
            TransformerDecoderLayer, TransformerEncoderLayer
        ] = TransformerDecoderLayer,
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
        self.text_position = SinePositionalEmbedding(d_model,dropout=0.1,scale=False)
        self.audio_positions = nn.ModuleList([SinePositionalEmbedding(d_model,dropout=0.1,scale=False) for i in range(8)])

        self.stage_embeddings = nn.ModuleList(
            [TokenEmbedding(d_model, 1) for i in range(8)]
        )

        self.ar_decoder = decoder_cls(
            decoder_layer_cls(
                d_model,
                num_heads,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=norm_first,
            ),
            num_layers=num_layers,
            norm=LayerNorm(d_model) if norm_first else None,
        )

        self.nar_decoder = decoder_cls(
            decoder_layer_cls(
                d_model,
                num_heads,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=norm_first,
                adaptive_layer_norm=True,
            ),
            num_layers=num_layers,
            norm=AdaptiveLayerNorm(d_model, norm=nn.LayerNorm(d_model)) if norm_first else None,
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
    
    def stage_named_parameters(
        self, stage: int = 1
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        assert stage > 0
        if stage == 1:
            for pair in self.named_parameters():
                if pair[0].startswith("ar_"):
                    yield pair

        if stage == 2:
            for pair in self.named_parameters():
                if pair[0].startswith("nar_"):
                    yield pair

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
        y_dec, _ = self.ar_decoder(
            tgt=(y_pos, None),
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

        y_dec, _ = self.nar_decoder(
            tgt=(y_pos, self.stage_embeddings[train_stage].weight),
            memory=x,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=y_mask,
            memory_key_padding_mask=x_mask,
        )
        nar_logits = self.predict_layers[train_stage](y_dec).permute(0, 2, 1)

        return codes, ar_logits, ar_targets, nar_logits, nar_targets

    def inference(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        enroll_x_lens: Union[torch.Tensor, None] = None,
        top_k: int = -100,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
          x:
            A 2-D tensor of shape (1, S).
          x_lens:
            A 1-D tensor of shape (1,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (1, T, 8).
          top_k: (`optional`) int
            The number of highest probability tokens to keep for top-k-filtering. Default to -100.
          temperature: (`optional`) float
            The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
        Returns:
          Return the predicted audio code matrix and cross-entropy loss.
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        assert y.shape[0] == 1, y.shape

        assert torch.all(x_lens > 0)

        x = self.text_embedding(x)
        x = self.text_prenet(x)
        x = self.text_position(x)
        # NOTE: x has been padded in TextTokenCollater
        x_mask = make_pad_mask(x_lens).to(x.device)

        prompts = y
        prompts_len = y.shape[1]

        # AR Decoder
        # TODO: Managing decoder steps avoid repetitive computation
        y = prompts[..., 0]
        while True:
            y_emb = self.ar_embedding(y)
            y_emb = self.audio_prenet(y_emb)
            y_pos = self.audio_positions[0](y_emb)

            tgt_mask = torch.triu(
                torch.ones(y.shape[1], y.shape[1], device=y.device, dtype=torch.bool),
                diagonal=1,
            )

            y_dec, _ = self.ar_decoder(
                (y_pos, None),
                x,
                tgt_mask=tgt_mask,
                memory_mask=None,
                memory_key_padding_mask=x_mask,
            )
            logits = self.predict_layers[0](y_dec[:, -1])
            if top_k > 0:
                samples = topk_sampling(logits, top_k=top_k, top_p=1.0, temperature=temperature)
            else:
                samples = torch.multinomial(F.softmax(logits, dim=-1),num_samples=1)

            if (
                samples[0, 0] == NUM_AUDIO_TOKENS
                or (y.shape[1] - prompts_len) > x_lens.max() * 16
            ):
                print(f"VALL-E EOS [{prompts_len} -> {y.shape[1]}]")
                break

            y = torch.concat([y, samples], dim=1)

        codes = [y[:, prompts_len:]]
        # Non-AR Decoders

        y_emb = self.nar_embeddings[0](y)
        for i, (predict_layer, embedding_layer) in enumerate(
            zip(
                self.predict_layers[1:],
                self.nar_embeddings[1:],
            )
        ):
            y_pos = self.audio_positions[i + 1](y_emb)
            y_dec, _ = self.nar_decoder(
                (y_pos, self.stage_embeddings[i + 1].weight),
                x,
                tgt_mask=None,
                memory_mask=None,
                memory_key_padding_mask=x_mask,
            )
            logits = predict_layer(y_dec[:, prompts_len:])

            samples = torch.argmax(logits, dim=-1)
            codes.append(samples)
            # Formula (4) (5)
            if i < 6:
                y_emb[:, :prompts_len] += embedding_layer(prompts[..., i + 1])
                y_emb[:, prompts_len:] += embedding_layer(samples)

        assert len(codes) == 8
        return torch.stack(codes, dim=-1)
    
