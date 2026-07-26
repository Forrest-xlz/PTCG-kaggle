"""Transformer architecture from the source notebook."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import torch


@dataclass(frozen=True)
class ModelConfig:
    card_count: int
    attack_count: int
    encoder_size: int = 22_000
    num_encoder_words: int = 24
    decoder_main_features: int = 8
    # cg.api.SelectContext.RECOVER_SPECIAL_CONDITION is 48 in the competition
    # API.  This matches the dynamic constant used by the source notebook.
    recover_special_condition: int = 48
    d_model: int = 128
    num_heads: int = 2
    d_feedforward: int = 256
    encoder_layers: int = 1
    decoder_layers: int = 1
    norm_mode: str = "postnorm"

    def __post_init__(self) -> None:
        if self.norm_mode not in {"prenorm", "postnorm"}:
            raise ValueError("norm_mode must be prenorm or postnorm")

    @property
    def decoder_attack_offset(self) -> int:
        return 14

    @property
    def decoder_card_offset(self) -> int:
        return self.decoder_attack_offset + self.attack_count

    @property
    def decoder_size(self) -> int:
        return self.decoder_card_offset + (1 + self.decoder_main_features + self.recover_special_condition) * self.card_count

    def to_dict(self) -> dict:
        return asdict(self)


class DecoderLayer(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_feedforward: int,
        norm_mode: str,
    ):
        super().__init__()
        self.prenorm = norm_mode == "prenorm"
        self.attention = torch.nn.MultiheadAttention(d_model, num_heads)
        self.fc1 = torch.nn.Linear(d_model, d_feedforward)
        self.fc2 = torch.nn.Linear(d_feedforward, d_model)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        if self.prenorm:
            query = self.norm1(x)
            attended, _ = self.attention(
                query, encoder_out, encoder_out, need_weights=False
            )
            x = x + attended
            return x + self.fc2(
                torch.nn.functional.relu(self.fc1(self.norm2(x)))
            )
        y, _ = self.attention(x, encoder_out, encoder_out, need_weights=False)
        residual = self.norm1(x + y)
        y = self.fc2(torch.nn.functional.relu(self.fc1(residual)))
        return self.norm2(residual + y)


class PTCGTransformer(torch.nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder_bag = torch.nn.EmbeddingBag(config.encoder_size, config.d_model, mode="sum")
        prenorm = config.norm_mode == "prenorm"
        layer = torch.nn.TransformerEncoderLayer(
            config.d_model,
            config.num_heads,
            config.d_feedforward,
            dropout=0,
            norm_first=prenorm,
        )
        final_norm = torch.nn.LayerNorm(config.d_model) if prenorm else None
        self.encoder = torch.nn.TransformerEncoder(
            layer,
            config.encoder_layers,
            norm=final_norm,
            enable_nested_tensor=False,
        )
        self.decoder_bag = torch.nn.EmbeddingBag(config.decoder_size, config.d_model, mode="sum")
        self.decoder = torch.nn.ModuleList(
            DecoderLayer(
                config.d_model,
                config.num_heads,
                config.d_feedforward,
                config.norm_mode,
            )
            for _ in range(config.decoder_layers)
        )
        self.decoder_fc = torch.nn.Linear(config.d_model, 1)

    def forward(
        self,
        index_encoder,
        value_encoder,
        offset_encoder,
        index_decoder,
        offset_decoder,
    ):
        cfg = self.config
        encoded = self.encoder_bag(index_encoder, offset_encoder, value_encoder)
        encoded = encoded.reshape(-1, cfg.num_encoder_words, cfg.d_model).transpose(0, 1)
        batch_size = encoded.size(1)
        encoder_out = self.encoder(encoded)
        policy = self.decoder_bag(index_decoder, offset_decoder)
        policy = policy.reshape(batch_size, -1, cfg.d_model).transpose(0, 1)
        # Every decoder layer cross-attends to the same encoder output. There
        # is deliberately no self-attention between candidate actions.
        for layer in self.decoder:
            policy = layer(policy, encoder_out)
        # Return raw logits. Cross entropy applies log-softmax internally, and
        # argmax(logits) is identical to argmax(softmax(logits)) at inference.
        return self.decoder_fc(policy).transpose(0, 1).reshape(batch_size, -1)
