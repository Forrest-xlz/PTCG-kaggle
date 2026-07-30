"""Transformer architecture from the source notebook."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import torch
import torch.nn.functional as F

from model.card_features import CARD_FEATURE_DIM


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
    card_feature_ratio: float = 0.5

    def __post_init__(self) -> None:
        if self.norm_mode not in {"prenorm", "postnorm"}:
            raise ValueError("norm_mode must be prenorm or postnorm")
        if not 0 < self.card_feature_ratio <= 1:
            raise ValueError("card_feature_ratio must be in (0, 1]")
        if self.card_feature_dim < 1:
            raise ValueError(
                "card_feature_ratio * d_model must be at least 1"
            )

    @property
    def decoder_attack_offset(self) -> int:
        return 14

    @property
    def decoder_card_offset(self) -> int:
        return self.decoder_attack_offset + self.attack_count

    @property
    def decoder_size(self) -> int:
        return self.decoder_card_offset + (1 + self.decoder_main_features + self.recover_special_condition) * self.card_count

    @property
    def card_feature_dim(self) -> int:
        return int(self.d_model * self.card_feature_ratio)

    def to_dict(self) -> dict:
        return asdict(self)


def _fill_card_range(
    mapping: torch.Tensor,
    start: int,
    card_count: int,
) -> int:
    end = start + card_count
    if end > mapping.numel():
        raise ValueError("card feature range exceeds embedding vocabulary")
    mapping[start:end] = torch.arange(card_count)
    return end


def _encoder_card_ids(config: ModelConfig) -> torch.Tensor:
    """Map encoder vocabulary indices to Card IDs or the zero sentinel."""
    card_count = config.card_count
    mapping = torch.full(
        (config.encoder_size,),
        card_count,
        dtype=torch.long,
    )
    position = 0

    # Two shared bench layouts and two active-Pokemon layouts.
    for _ in range(4):
        position += 2  # null flag and HP
        for _ in range(3):  # Pokemon, tools, attached energies
            position = _fill_card_range(mapping, position, card_count)

    # Two player summaries, each ending in a discard-card block.
    for _ in range(2):
        position += 4 + 7 + 5
        position = _fill_card_range(mapping, position, card_count)

    # Own hand, known deck composition, and stadium.
    for _ in range(3):
        position = _fill_card_range(mapping, position, card_count)

    position += 3  # global scalar features
    if position > config.encoder_size:
        raise ValueError(
            "encoder_size is too small for the configured card vocabulary"
        )
    return mapping


def _decoder_card_ids(config: ModelConfig) -> torch.Tensor:
    """Map decoder vocabulary indices to Card IDs or the zero sentinel."""
    mapping = torch.full(
        (config.decoder_size,),
        config.card_count,
        dtype=torch.long,
    )
    card_region = torch.arange(
        config.decoder_size - config.decoder_card_offset,
        dtype=torch.long,
    )
    mapping[config.decoder_card_offset:] = (
        card_region % config.card_count
    )
    return mapping


class CardAwareEmbeddingBag(torch.nn.EmbeddingBag):
    """EmbeddingBag that adds a caller-supplied shared card representation."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        index_to_card_id: torch.Tensor,
    ):
        super().__init__(num_embeddings, embedding_dim, mode="sum")
        if index_to_card_id.shape != (num_embeddings,):
            raise ValueError(
                "index_to_card_id must match the embedding vocabulary"
            )
        self.register_buffer(
            "index_to_card_id",
            index_to_card_id,
            persistent=False,
        )

    def forward(
        self,
        input: torch.Tensor,
        offsets: torch.Tensor | None = None,
        per_sample_weights: torch.Tensor | None = None,
        *,
        projected_card_features: torch.Tensor,
    ) -> torch.Tensor:
        learned = super().forward(
            input,
            offsets,
            per_sample_weights=per_sample_weights,
        )
        card_ids = self.index_to_card_id[input]
        static_weights = (
            None
            if per_sample_weights is None
            else per_sample_weights.to(
                dtype=projected_card_features.dtype
            )
        )
        static = F.embedding_bag(
            card_ids,
            projected_card_features,
            offsets,
            mode="sum",
            per_sample_weights=static_weights,
            include_last_offset=self.include_last_offset,
        )
        return learned + static


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
    def __init__(
        self,
        config: ModelConfig,
        card_feature_table: torch.Tensor,
    ):
        super().__init__()
        self.config = config
        card_feature_table = torch.as_tensor(
            card_feature_table,
            dtype=torch.float32,
        )
        expected_shape = (config.card_count, CARD_FEATURE_DIM)
        if tuple(card_feature_table.shape) != expected_shape:
            raise ValueError(
                "card_feature_table must have shape "
                f"{expected_shape}, found {tuple(card_feature_table.shape)}"
            )
        self.register_buffer(
            "card_feature_table",
            card_feature_table.clone(),
        )
        self.card_feature_projection = torch.nn.Linear(
            CARD_FEATURE_DIM,
            config.card_feature_dim,
        )
        self.encoder_bag = CardAwareEmbeddingBag(
            config.encoder_size,
            config.d_model,
            index_to_card_id=_encoder_card_ids(config),
        )
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
        self.decoder_bag = CardAwareEmbeddingBag(
            config.decoder_size,
            config.d_model,
            index_to_card_id=_decoder_card_ids(config),
        )
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

    def project_card_features(self) -> torch.Tensor:
        projected = self.card_feature_projection(
            self.card_feature_table
        )
        projected = F.pad(
            projected,
            (0, self.config.d_model - self.config.card_feature_dim),
        )
        # The last row is the sentinel used by every non-card bag index.
        return torch.cat(
            [
                projected,
                projected.new_zeros((1, self.config.d_model)),
            ],
            dim=0,
        )

    def forward(
        self,
        index_encoder,
        value_encoder,
        offset_encoder,
        index_decoder,
        offset_decoder,
    ):
        cfg = self.config
        projected_card_features = self.project_card_features()
        encoded = self.encoder_bag(
            index_encoder,
            offset_encoder,
            per_sample_weights=value_encoder,
            projected_card_features=projected_card_features,
        )
        encoded = encoded.reshape(-1, cfg.num_encoder_words, cfg.d_model).transpose(0, 1)
        batch_size = encoded.size(1)
        encoder_out = self.encoder(encoded)
        policy = self.decoder_bag(
            index_decoder,
            offset_decoder,
            projected_card_features=projected_card_features,
        )
        policy = policy.reshape(batch_size, -1, cfg.d_model).transpose(0, 1)
        # Every decoder layer cross-attends to the same encoder output. There
        # is deliberately no self-attention between candidate actions.
        for layer in self.decoder:
            policy = layer(policy, encoder_out)
        # Return raw logits. Cross entropy applies log-softmax internally, and
        # argmax(logits) is identical to argmax(softmax(logits)) at inference.
        return self.decoder_fc(policy).transpose(0, 1).reshape(batch_size, -1)
