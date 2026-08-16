"""Transformer architecture from the source notebook."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import torch
import torch.nn.functional as F

from model.attack_features import ATTACK_FEATURE_DIM
from model.card_features import CARD_FEATURE_DIM


ENCODER_TOKENS = 27
POKEMON_ENCODER_TOKENS = 18
BENCH_SLOTS = 8
PLAYER_BENCH_COUNT_INDEX = 10
OWN_SUMMARY_DIM = 84
OPPONENT_SUMMARY_DIM = 82
GLOBAL_SUMMARY_DIM = 73
SELECT_TYPE_COUNT = 11
OPTION_TYPE_COUNT = 17
OPTION_CONTEXT_COUNT = 49
OPTION_VALUE_COUNT = 63
OPTION_PLAYER_RELATION_COUNT = 3
OPTION_AREA_COUNT = 13
OPTION_SPECIAL_CONDITION_COUNT = 6
OPTION_NUMERIC_DIM = 5
POKEMON_DYNAMIC_WORD_DIM = 23
POKEMON_DYNAMIC_DIM = 46
ATTACK_DYNAMIC_DIM = 6
HISTORY_STEPS = 3
HISTORY_STRUCTURAL_DIM = 8

HISTORY_OPTION_TYPE_INDEX = 0
HISTORY_SOURCE_AREA_INDEX = 1
HISTORY_TARGET_AREA_INDEX = 2
HISTORY_SOURCE_RELATION_INDEX = 3
HISTORY_TARGET_RELATION_INDEX = 4
HISTORY_NUMBER_INDEX = 5
HISTORY_COUNT_INDEX = 6
HISTORY_SPECIAL_CONDITION_INDEX = 7

OPTION_PLAYER_RELATION_INDEX = 7
OPTION_AREA_INDEX = 8
OPTION_IN_PLAY_AREA_INDEX = 9

OPTION_TYPE_PLAY = 7
OPTION_TYPE_ATTACH = 8
OPTION_TYPE_EVOLVE = 9
OPTION_TYPE_RETREAT = 12

CARD_REGION_NAMES = (
    "own_bench",
    "opponent_bench",
    "own_active",
    "opponent_active",
    "own_discard",
    "opponent_discard",
    "own_hand",
    "opponent_hand",
    "own_deck",
    "own_known_deck",
    "opponent_deck",
    "own_prize",
    "opponent_prize",
    "stadium",
    "looking",
    "unknown",
)
CARD_REGION_INDEX = {
    name: index for index, name in enumerate(CARD_REGION_NAMES)
}

# AreaType values 0..12 mapped to relative card regions. Attached tools and
# Energy cards inherit the Active/Bench region of their Pokemon.
_OWN_AREA_REGIONS = (
    CARD_REGION_INDEX["unknown"],
    CARD_REGION_INDEX["own_deck"],
    CARD_REGION_INDEX["own_hand"],
    CARD_REGION_INDEX["own_discard"],
    CARD_REGION_INDEX["own_active"],
    CARD_REGION_INDEX["own_bench"],
    CARD_REGION_INDEX["own_prize"],
    CARD_REGION_INDEX["stadium"],
    CARD_REGION_INDEX["unknown"],
    CARD_REGION_INDEX["unknown"],
    CARD_REGION_INDEX["unknown"],
    CARD_REGION_INDEX["unknown"],
    CARD_REGION_INDEX["looking"],
)
_OPPONENT_AREA_REGIONS = (
    CARD_REGION_INDEX["unknown"],
    CARD_REGION_INDEX["opponent_deck"],
    CARD_REGION_INDEX["opponent_hand"],
    CARD_REGION_INDEX["opponent_discard"],
    CARD_REGION_INDEX["opponent_active"],
    CARD_REGION_INDEX["opponent_bench"],
    CARD_REGION_INDEX["opponent_prize"],
    CARD_REGION_INDEX["stadium"],
    CARD_REGION_INDEX["unknown"],
    CARD_REGION_INDEX["unknown"],
    CARD_REGION_INDEX["unknown"],
    CARD_REGION_INDEX["unknown"],
    CARD_REGION_INDEX["looking"],
)


@dataclass(frozen=True)
class ModelConfig:
    card_count: int
    attack_count: int
    encoder_size: int = 23_000
    d_model: int = 128
    num_heads: int = 2
    d_feedforward: int = 256
    encoder_layers: int = 1
    decoder_layers: int = 1
    norm_mode: str = "postnorm"
    transformer_activation: str = "relu"
    transformer_dropout: float = 0.0
    dropout_embedding: bool = False
    dropout_attention_probs: bool = False
    dropout_attention_output: bool = False
    dropout_ffn_output: bool = False
    summary_mlp_layers: int = 1
    card_mlp_layers: int = 1
    option_numeric_mlp_layers: int = 1
    option_token_mlp_layers: int = 0
    card_mlp_scope: str = "shared"
    pokemon_appear_embedding: bool = False
    bench_token_mlp_layers: int = 0
    active_token_mlp_layers: int = 0
    discard_token_mlp_layers: int = 0
    hand_token_mlp_layers: int = 0
    deck_token_mlp_layers: int = 0
    known_deck_token_mlp_layers: int = 0
    region_token_mlp_residual: bool = True
    history_encoding: str = "off"
    history_action_mlp_layers: int = 1
    history_sequence_mlp_layers: int = 2

    def __post_init__(self) -> None:
        if self.norm_mode not in {"prenorm", "postnorm"}:
            raise ValueError("norm_mode must be prenorm or postnorm")
        if self.transformer_activation not in {"relu", "gelu", "geglu"}:
            raise ValueError(
                "transformer_activation must be relu, gelu, or geglu"
            )
        if not 0.0 <= float(self.transformer_dropout) < 1.0:
            raise ValueError("transformer_dropout must be in [0, 1)")
        for name in (
            "dropout_embedding",
            "dropout_attention_probs",
            "dropout_attention_output",
            "dropout_ffn_output",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.summary_mlp_layers < 1:
            raise ValueError("summary_mlp_layers must be >= 1")
        if self.card_mlp_layers < 0:
            raise ValueError("card_mlp_layers must be >= 0")
        if self.card_mlp_scope not in {"shared", "region"}:
            raise ValueError("card_mlp_scope must be shared or region")
        if (
            type(self.option_numeric_mlp_layers) is not int
            or self.option_numeric_mlp_layers < 1
        ):
            raise ValueError(
                "option_numeric_mlp_layers must be an integer >= 1"
            )
        if (
            type(self.option_token_mlp_layers) is not int
            or self.option_token_mlp_layers < 0
        ):
            raise ValueError("option_token_mlp_layers must be an integer >= 0")
        if type(self.pokemon_appear_embedding) is not bool:
            raise ValueError("pokemon_appear_embedding must be a boolean")
        for name in (
            "bench_token_mlp_layers",
            "active_token_mlp_layers",
            "discard_token_mlp_layers",
            "hand_token_mlp_layers",
            "deck_token_mlp_layers",
            "known_deck_token_mlp_layers",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be an integer >= 0")
        if type(self.region_token_mlp_residual) is not bool:
            raise ValueError("region_token_mlp_residual must be a boolean")
        if self.history_encoding not in {
            "off",
            "basic",
            "structural",
            "full",
        }:
            raise ValueError(
                "history_encoding must be off, basic, structural, or full"
            )
        if (
            type(self.history_action_mlp_layers) is not int
            or self.history_action_mlp_layers < 0
        ):
            raise ValueError(
                "history_action_mlp_layers must be an integer >= 0"
            )
        minimum_sequence_layers = int(self.history_encoding != "off")
        if (
            type(self.history_sequence_mlp_layers) is not int
            or self.history_sequence_mlp_layers < minimum_sequence_layers
        ):
            raise ValueError(
                "history_sequence_mlp_layers must be an integer >= 1 "
                "when history_encoding is enabled"
            )

    def to_dict(self) -> dict:
        return asdict(self)


def _projection_mlp(
    input_dim: int,
    d_model: int,
    layers: int,
) -> torch.nn.Module:
    if layers < 1:
        raise ValueError("projection MLP must contain at least one layer")
    modules: list[torch.nn.Module] = [
        torch.nn.Linear(input_dim, d_model)
    ]
    for _ in range(layers - 1):
        modules.extend(
            [torch.nn.ReLU(), torch.nn.Linear(d_model, d_model)]
        )
    return modules[0] if len(modules) == 1 else torch.nn.Sequential(*modules)


def _fill_card_range(
    card_mapping: torch.Tensor,
    region_mapping: torch.Tensor,
    start: int,
    card_count: int,
    region: int,
) -> int:
    end = start + card_count
    if end > card_mapping.numel():
        raise ValueError("card feature range exceeds embedding vocabulary")
    card_mapping[start:end] = torch.arange(card_count)
    region_mapping[start:end] = region
    return end


def _encoder_card_mappings(
    config: ModelConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map encoder vocabulary indices to Card IDs and semantic regions."""
    card_count = config.card_count
    card_mapping = torch.full(
        (config.encoder_size,),
        card_count,
        dtype=torch.long,
    )
    region_mapping = torch.full(
        (config.encoder_size,),
        CARD_REGION_INDEX["unknown"],
        dtype=torch.long,
    )
    position = 0

    # Two shared bench layouts and two active-Pokemon layouts.
    field_regions = (
        CARD_REGION_INDEX["own_bench"],
        CARD_REGION_INDEX["opponent_bench"],
        CARD_REGION_INDEX["own_active"],
        CARD_REGION_INDEX["opponent_active"],
    )
    for region in field_regions:
        position += 2  # null flag and HP
        for _ in range(3):  # Pokemon, tools, attached energies
            position = _fill_card_range(
                card_mapping,
                region_mapping,
                position,
                card_count,
                region,
            )

    # Own discard, opponent discard, own hand, full deck, known-in-deck, stadium.
    zone_regions = (
        CARD_REGION_INDEX["own_discard"],
        CARD_REGION_INDEX["opponent_discard"],
        CARD_REGION_INDEX["own_hand"],
        CARD_REGION_INDEX["own_deck"],
        CARD_REGION_INDEX["own_known_deck"],
        CARD_REGION_INDEX["stadium"],
    )
    for region in zone_regions:
        position = _fill_card_range(
            card_mapping,
            region_mapping,
            position,
            card_count,
            region,
        )

    if position > config.encoder_size:
        raise ValueError(
            "encoder_size is too small for the configured card vocabulary"
        )
    return card_mapping, region_mapping


def _encoder_card_ids(config: ModelConfig) -> torch.Tensor:
    """Backward-compatible helper returning only encoder Card IDs."""
    return _encoder_card_mappings(config)[0]


class CardAwareEmbeddingBag(torch.nn.EmbeddingBag):
    """EmbeddingBag that adds shared or region-specific card features."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        index_to_card_id: torch.Tensor,
        index_to_card_region: torch.Tensor,
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
        if index_to_card_region.shape != (num_embeddings,):
            raise ValueError(
                "index_to_card_region must match the embedding vocabulary"
            )
        self.register_buffer(
            "index_to_card_region",
            index_to_card_region,
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
        if projected_card_features.ndim == 2:
            static_indices = card_ids
            static_table = projected_card_features
        elif projected_card_features.ndim == 3:
            region_ids = self.index_to_card_region[input]
            card_vocabulary = projected_card_features.size(1)
            static_indices = region_ids * card_vocabulary + card_ids
            static_table = projected_card_features.flatten(0, 1)
        else:
            raise ValueError(
                "projected_card_features must have two or three dimensions"
            )
        static_weights = (
            None
            if per_sample_weights is None
            else per_sample_weights.to(
                dtype=projected_card_features.dtype
            )
        )
        static = F.embedding_bag(
            static_indices,
            static_table,
            offsets,
            mode="sum",
            per_sample_weights=static_weights,
            include_last_offset=self.include_last_offset,
        )
        return learned + static


class EncoderLayer(torch.nn.TransformerEncoderLayer):
    """TransformerEncoderLayer with independently switchable dropouts."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_feedforward: int,
        norm_mode: str,
        activation: str = "relu",
        dropout: float = 0.0,
        dropout_attention_probs: bool = False,
        dropout_attention_output: bool = False,
        dropout_ffn_output: bool = False,
    ):
        super().__init__(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_feedforward,
            dropout=0.0,
            activation="relu",
            norm_first=norm_mode == "prenorm",
        )
        self.transformer_activation = activation
        if activation == "geglu":
            self.linear1 = torch.nn.Linear(
                d_model,
                2 * d_feedforward,
            )
        self.activation = self._activate
        self.activation_relu_or_gelu = 1 if activation == "relu" else 0
        probability = float(dropout)
        self.self_attn.dropout = (
            probability if dropout_attention_probs else 0.0
        )
        self.dropout.p = 0.0
        self.dropout1.p = (
            probability if dropout_attention_output else 0.0
        )
        self.dropout2.p = (
            probability if dropout_ffn_output else 0.0
        )

    def _activate(self, value: torch.Tensor) -> torch.Tensor:
        if self.transformer_activation == "relu":
            return F.relu(value)
        if self.transformer_activation == "gelu":
            return F.gelu(value, approximate="tanh")
        value, gate = value.chunk(2, dim=-1)
        return value * F.gelu(gate, approximate="tanh")


class DecoderLayer(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_feedforward: int,
        norm_mode: str,
        activation: str = "relu",
        dropout: float = 0.0,
        dropout_attention_probs: bool = False,
        dropout_attention_output: bool = False,
        dropout_ffn_output: bool = False,
    ):
        super().__init__()
        self.prenorm = norm_mode == "prenorm"
        probability = float(dropout)
        self.transformer_activation = activation
        self.attention = torch.nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=(
                probability if dropout_attention_probs else 0.0
            ),
        )
        self.fc1 = torch.nn.Linear(
            d_model,
            d_feedforward * (2 if activation == "geglu" else 1),
        )
        self.fc2 = torch.nn.Linear(d_feedforward, d_model)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.attention_output_dropout = torch.nn.Dropout(
            probability if dropout_attention_output else 0.0
        )
        self.ffn_output_dropout = torch.nn.Dropout(
            probability if dropout_ffn_output else 0.0
        )

    def _activate(self, value: torch.Tensor) -> torch.Tensor:
        if self.transformer_activation == "relu":
            return F.relu(value)
        if self.transformer_activation == "gelu":
            return F.gelu(value, approximate="tanh")
        value, gate = value.chunk(2, dim=-1)
        return value * F.gelu(gate, approximate="tanh")

    def _feed_forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.ffn_output_dropout(
            self.fc2(self._activate(self.fc1(value)))
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.prenorm:
            query = self.norm1(x)
            attended, _ = self.attention(
                query,
                encoder_out,
                encoder_out,
                key_padding_mask=encoder_padding_mask,
                need_weights=False,
            )
            x = x + self.attention_output_dropout(attended)
            return x + self._feed_forward(self.norm2(x))
        y, _ = self.attention(
            x,
            encoder_out,
            encoder_out,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
        )
        residual = self.norm1(
            x + self.attention_output_dropout(y)
        )
        return self.norm2(residual + self._feed_forward(residual))


class PTCGTransformer(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        card_feature_table: torch.Tensor,
        attack_feature_table: torch.Tensor,
    ):
        super().__init__()
        self.config = config
        self.encoder_token_count = ENCODER_TOKENS + int(
            config.history_encoding != "off"
        )
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
        attack_feature_table = torch.as_tensor(
            attack_feature_table,
            dtype=torch.float32,
        )
        expected_attack_shape = (
            config.attack_count,
            ATTACK_FEATURE_DIM,
        )
        if tuple(attack_feature_table.shape) != expected_attack_shape:
            raise ValueError(
                "attack_feature_table must have shape "
                f"{expected_attack_shape}, found "
                f"{tuple(attack_feature_table.shape)}"
            )
        self.register_buffer(
            "attack_feature_table",
            attack_feature_table.clone(),
        )
        self.card_feature_projection: torch.nn.Module | None = None
        self.card_feature_projections: torch.nn.ModuleDict | None = None
        if config.card_mlp_layers > 0 and config.card_mlp_scope == "shared":
            self.card_feature_projection = _projection_mlp(
                CARD_FEATURE_DIM,
                config.d_model,
                config.card_mlp_layers,
            )
        elif config.card_mlp_layers > 0:
            self.card_feature_projections = torch.nn.ModuleDict(
                {
                    name: _projection_mlp(
                        CARD_FEATURE_DIM,
                        config.d_model,
                        config.card_mlp_layers,
                    )
                    for name in CARD_REGION_NAMES
                }
            )
        self.own_summary_projection = _projection_mlp(
            OWN_SUMMARY_DIM,
            config.d_model,
            config.summary_mlp_layers,
        )
        self.opponent_summary_projection = _projection_mlp(
            OPPONENT_SUMMARY_DIM,
            config.d_model,
            config.summary_mlp_layers,
        )
        self.global_summary_projection = _projection_mlp(
            GLOBAL_SUMMARY_DIM,
            config.d_model,
            config.summary_mlp_layers,
        )
        encoder_card_ids, encoder_card_regions = _encoder_card_mappings(
            config
        )
        self.encoder_bag = CardAwareEmbeddingBag(
            config.encoder_size,
            config.d_model,
            index_to_card_id=encoder_card_ids,
            index_to_card_region=encoder_card_regions,
        )
        self.pokemon_appear_embedding = (
            torch.nn.Embedding(3, config.d_model, padding_idx=0)
            if config.pokemon_appear_embedding
            else None
        )
        self.own_bench_token_mlp = self._make_token_mlp(
            config.bench_token_mlp_layers
        )
        self.opponent_bench_token_mlp = self._make_token_mlp(
            config.bench_token_mlp_layers
        )
        self.own_active_token_mlp = self._make_token_mlp(
            config.active_token_mlp_layers
        )
        self.opponent_active_token_mlp = self._make_token_mlp(
            config.active_token_mlp_layers
        )
        self.own_discard_token_mlp = self._make_token_mlp(
            config.discard_token_mlp_layers
        )
        self.opponent_discard_token_mlp = self._make_token_mlp(
            config.discard_token_mlp_layers
        )
        self.own_hand_token_mlp = self._make_token_mlp(
            config.hand_token_mlp_layers
        )
        self.own_deck_token_mlp = self._make_token_mlp(
            config.deck_token_mlp_layers
        )
        self.own_known_deck_token_mlp = self._make_token_mlp(
            config.known_deck_token_mlp_layers
        )
        self.register_buffer(
            "own_area_card_regions",
            torch.tensor(_OWN_AREA_REGIONS, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "opponent_area_card_regions",
            torch.tensor(_OPPONENT_AREA_REGIONS, dtype=torch.long),
            persistent=False,
        )
        prenorm = config.norm_mode == "prenorm"
        layer = EncoderLayer(
            config.d_model,
            config.num_heads,
            config.d_feedforward,
            config.norm_mode,
            config.transformer_activation,
            config.transformer_dropout,
            config.dropout_attention_probs,
            config.dropout_attention_output,
            config.dropout_ffn_output,
        )
        final_norm = torch.nn.LayerNorm(config.d_model) if prenorm else None
        self.encoder = torch.nn.TransformerEncoder(
            layer,
            config.encoder_layers,
            norm=final_norm,
            enable_nested_tensor=False,
        )
        self.encoder_input_norm = (
            torch.nn.LayerNorm(config.d_model)
            if config.dropout_embedding
            else None
        )
        self.action_input_norm = (
            torch.nn.LayerNorm(config.d_model)
            if config.dropout_embedding
            else None
        )
        self.embedding_dropout = torch.nn.Dropout(
            config.transformer_dropout
            if config.dropout_embedding
            else 0.0
        )
        self.option_type_embedding = torch.nn.Embedding(
            OPTION_TYPE_COUNT, config.d_model
        )
        self.option_context_embedding = torch.nn.Embedding(
            OPTION_CONTEXT_COUNT, config.d_model
        )
        self.option_candidate_embedding = torch.nn.Embedding(
            config.card_count + 1,
            config.d_model,
            padding_idx=config.card_count,
        )
        self.option_target_embedding = torch.nn.Embedding(
            config.card_count + 1,
            config.d_model,
            padding_idx=config.card_count,
        )
        self.option_attack_embedding = torch.nn.Embedding(
            config.attack_count + 1,
            config.d_model,
            padding_idx=config.attack_count,
        )
        self.option_number_embedding = torch.nn.Embedding(
            OPTION_VALUE_COUNT, config.d_model, padding_idx=0
        )
        self.option_count_embedding = torch.nn.Embedding(
            OPTION_VALUE_COUNT, config.d_model, padding_idx=0
        )
        self.option_player_relation_embedding = torch.nn.Embedding(
            OPTION_PLAYER_RELATION_COUNT, config.d_model, padding_idx=0
        )
        self.option_area_embedding = torch.nn.Embedding(
            OPTION_AREA_COUNT, config.d_model, padding_idx=0
        )
        self.option_in_play_area_embedding = torch.nn.Embedding(
            OPTION_AREA_COUNT, config.d_model, padding_idx=0
        )
        self.option_special_condition_embedding = torch.nn.Embedding(
            OPTION_SPECIAL_CONDITION_COUNT, config.d_model, padding_idx=0
        )
        self.option_numeric_projection = _projection_mlp(
            OPTION_NUMERIC_DIM,
            config.d_model,
            config.option_numeric_mlp_layers,
        )
        self.pokemon_dynamic_projection = torch.nn.Linear(
            POKEMON_DYNAMIC_DIM, config.d_model
        )
        self.attack_dynamic_projection = torch.nn.Linear(
            ATTACK_DYNAMIC_DIM, config.d_model
        )
        self.option_token_mlp = self._make_token_mlp(
            config.option_token_mlp_layers
        )
        self.attack_feature_projection = torch.nn.Linear(
            ATTACK_FEATURE_DIM, config.d_model
        )
        self.no_action_embedding = torch.nn.Parameter(
            torch.zeros(config.d_model)
        )
        self.history_select_type_embedding = None
        self.history_context_embedding = None
        self.history_no_action_embedding = None
        self.history_action_mlp = None
        self.history_sequence_mlp = None
        if config.history_encoding != "off":
            self.history_select_type_embedding = torch.nn.Embedding(
                SELECT_TYPE_COUNT, config.d_model
            )
            self.history_context_embedding = torch.nn.Embedding(
                OPTION_CONTEXT_COUNT, config.d_model
            )
            self.history_no_action_embedding = torch.nn.Parameter(
                torch.zeros(config.d_model)
            )
            self.history_option_type_embedding = torch.nn.Embedding(
                OPTION_TYPE_COUNT, config.d_model
            )
            if config.history_encoding == "structural":
                self.history_source_area_embedding = torch.nn.Embedding(
                    OPTION_AREA_COUNT, config.d_model, padding_idx=0
                )
                self.history_target_area_embedding = torch.nn.Embedding(
                    OPTION_AREA_COUNT, config.d_model, padding_idx=0
                )
                self.history_source_relation_embedding = torch.nn.Embedding(
                    OPTION_PLAYER_RELATION_COUNT,
                    config.d_model,
                    padding_idx=0,
                )
                self.history_target_relation_embedding = torch.nn.Embedding(
                    OPTION_PLAYER_RELATION_COUNT,
                    config.d_model,
                    padding_idx=0,
                )
                self.history_number_embedding = torch.nn.Embedding(
                    OPTION_VALUE_COUNT, config.d_model, padding_idx=0
                )
                self.history_count_embedding = torch.nn.Embedding(
                    OPTION_VALUE_COUNT, config.d_model, padding_idx=0
                )
                self.history_special_condition_embedding = (
                    torch.nn.Embedding(
                        OPTION_SPECIAL_CONDITION_COUNT,
                        config.d_model,
                        padding_idx=0,
                    )
                )
            elif config.history_encoding == "full":
                self.history_candidate_embedding = torch.nn.Embedding(
                    config.card_count + 1,
                    config.d_model,
                    padding_idx=config.card_count,
                )
                self.history_target_embedding = torch.nn.Embedding(
                    config.card_count + 1,
                    config.d_model,
                    padding_idx=config.card_count,
                )
                self.history_attack_embedding = torch.nn.Embedding(
                    config.attack_count + 1,
                    config.d_model,
                    padding_idx=config.attack_count,
                )
                self.history_number_embedding = torch.nn.Embedding(
                    OPTION_VALUE_COUNT, config.d_model, padding_idx=0
                )
                self.history_count_embedding = torch.nn.Embedding(
                    OPTION_VALUE_COUNT, config.d_model, padding_idx=0
                )
                self.history_player_relation_embedding = torch.nn.Embedding(
                    OPTION_PLAYER_RELATION_COUNT,
                    config.d_model,
                    padding_idx=0,
                )
                self.history_area_embedding = torch.nn.Embedding(
                    OPTION_AREA_COUNT, config.d_model, padding_idx=0
                )
                self.history_in_play_area_embedding = torch.nn.Embedding(
                    OPTION_AREA_COUNT, config.d_model, padding_idx=0
                )
                self.history_special_condition_embedding = (
                    torch.nn.Embedding(
                        OPTION_SPECIAL_CONDITION_COUNT,
                        config.d_model,
                        padding_idx=0,
                    )
                )
                self.history_candidate_static_projection = None
                self.history_target_static_projection = None
                if config.card_mlp_layers > 0:
                    self.history_candidate_static_projection = (
                        _projection_mlp(
                            CARD_FEATURE_DIM,
                            config.d_model,
                            config.card_mlp_layers,
                        )
                    )
                    self.history_target_static_projection = _projection_mlp(
                        CARD_FEATURE_DIM,
                        config.d_model,
                        config.card_mlp_layers,
                    )
                self.history_attack_static_projection = torch.nn.Linear(
                    ATTACK_FEATURE_DIM, config.d_model
                )
                self.history_pokemon_dynamic_projection = torch.nn.Linear(
                    POKEMON_DYNAMIC_DIM, config.d_model
                )
                self.history_attack_dynamic_projection = torch.nn.Linear(
                    ATTACK_DYNAMIC_DIM, config.d_model
                )
            self.history_action_mlp = self._make_token_mlp(
                config.history_action_mlp_layers
            )
            self.history_sequence_mlp = _projection_mlp(
                HISTORY_STEPS * config.d_model,
                config.d_model,
                config.history_sequence_mlp_layers,
            )
        self.decoder = torch.nn.ModuleList(
            DecoderLayer(
                config.d_model,
                config.num_heads,
                config.d_feedforward,
                config.norm_mode,
                config.transformer_activation,
                config.transformer_dropout,
                config.dropout_attention_probs,
                config.dropout_attention_output,
                config.dropout_ffn_output,
            )
            for _ in range(config.decoder_layers)
        )
        self.decoder_fc = torch.nn.Linear(config.d_model, 1)

    def _make_token_mlp(self, layers: int) -> torch.nn.Module | None:
        if layers == 0:
            return None
        return _projection_mlp(
            self.config.d_model,
            self.config.d_model,
            layers,
        )

    def _apply_token_mlp(
        self,
        tokens: torch.Tensor,
        mlp: torch.nn.Module | None,
    ) -> torch.Tensor:
        if mlp is None:
            return tokens
        transformed = mlp(tokens)
        if self.config.region_token_mlp_residual:
            return tokens + transformed
        return transformed

    def apply_region_token_mlps(
        self,
        encoded: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            (
                self._apply_token_mlp(
                    encoded[:, 0:8], self.own_bench_token_mlp
                ),
                self._apply_token_mlp(
                    encoded[:, 8:16], self.opponent_bench_token_mlp
                ),
                self._apply_token_mlp(
                    encoded[:, 16:17], self.own_active_token_mlp
                ),
                self._apply_token_mlp(
                    encoded[:, 17:18], self.opponent_active_token_mlp
                ),
                encoded[:, 18:20],
                self._apply_token_mlp(
                    encoded[:, 20:21], self.own_discard_token_mlp
                ),
                self._apply_token_mlp(
                    encoded[:, 21:22], self.opponent_discard_token_mlp
                ),
                self._apply_token_mlp(
                    encoded[:, 22:23], self.own_hand_token_mlp
                ),
                self._apply_token_mlp(
                    encoded[:, 23:24], self.own_deck_token_mlp
                ),
                self._apply_token_mlp(
                    encoded[:, 24:25], self.own_known_deck_token_mlp
                ),
                encoded[:, 25:27],
            ),
            dim=1,
        )

    def project_card_features(self) -> torch.Tensor:
        if (
            self.card_feature_projection is None
            and self.card_feature_projections is None
        ):
            projected = self.card_feature_table.new_zeros(
                (self.config.card_count, self.config.d_model)
            )
        elif self.card_feature_projection is not None:
            projected = self.card_feature_projection(
                self.card_feature_table
            )
        else:
            assert self.card_feature_projections is not None
            projected = torch.stack(
                [
                    self.card_feature_projections[name](
                        self.card_feature_table
                    )
                    for name in CARD_REGION_NAMES
                ],
                dim=0,
            )
        # The last row is the sentinel used by every non-card bag index.
        if projected.ndim == 2:
            sentinel = projected.new_zeros((1, self.config.d_model))
            return torch.cat((projected, sentinel), dim=0)
        sentinel = projected.new_zeros(
            (projected.size(0), 1, self.config.d_model)
        )
        return torch.cat((projected, sentinel), dim=1)

    def decoder_card_regions(
        self,
        categorical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        option_types = categorical[:, 0]
        player_relation = categorical[:, OPTION_PLAYER_RELATION_INDEX]
        areas = categorical[:, OPTION_AREA_INDEX]
        in_play_areas = categorical[:, OPTION_IN_PLAY_AREA_INDEX]

        candidate_regions = self.own_area_card_regions[areas]
        candidate_regions = torch.where(
            player_relation == 2,
            self.opponent_area_card_regions[areas],
            candidate_regions,
        )
        candidate_regions = torch.where(
            option_types == OPTION_TYPE_PLAY,
            torch.full_like(
                candidate_regions,
                CARD_REGION_INDEX["own_hand"],
            ),
            candidate_regions,
        )

        target_regions = torch.full_like(
            candidate_regions,
            CARD_REGION_INDEX["unknown"],
        )
        has_in_play_target = (
            (option_types == OPTION_TYPE_ATTACH)
            | (option_types == OPTION_TYPE_EVOLVE)
        )
        target_regions = torch.where(
            has_in_play_target,
            self.own_area_card_regions[in_play_areas],
            target_regions,
        )
        target_regions = torch.where(
            option_types == OPTION_TYPE_RETREAT,
            torch.full_like(
                target_regions,
                CARD_REGION_INDEX["own_active"],
            ),
            target_regions,
        )
        return candidate_regions, target_regions

    def project_attack_features(self) -> torch.Tensor:
        projected = self.attack_feature_projection(
            self.attack_feature_table
        )
        return torch.cat(
            [
                projected,
                projected.new_zeros((1, self.config.d_model)),
            ],
            dim=0,
        )

    def project_pokemon_dynamic(
        self, features: torch.Tensor
    ) -> torch.Tensor:
        present = (
            (features[:, 0] > 0)
            | (features[:, POKEMON_DYNAMIC_WORD_DIM] > 0)
        )
        return self.pokemon_dynamic_projection(features) * present.unsqueeze(1)

    def project_attack_dynamic(
        self, features: torch.Tensor
    ) -> torch.Tensor:
        present = features[:, 0] > 0
        return self.attack_dynamic_projection(features) * present.unsqueeze(1)

    def apply_option_token_mlp(self, token: torch.Tensor) -> torch.Tensor:
        if self.option_token_mlp is None:
            return token
        return self.option_token_mlp(token)

    def encode_options(
        self,
        categorical: torch.Tensor,
        numeric: torch.Tensor,
        pokemon_dynamic: torch.Tensor,
        attack_dynamic: torch.Tensor,
        projected_card_features: torch.Tensor,
        projected_attack_features: torch.Tensor,
    ) -> torch.Tensor:
        candidate_ids = categorical[:, 2]
        target_ids = categorical[:, 3]
        attack_ids = categorical[:, 4]
        if projected_card_features.ndim == 3:
            candidate_regions, target_regions = self.decoder_card_regions(
                categorical
            )
            candidate_static = projected_card_features[
                candidate_regions, candidate_ids
            ]
            target_static = projected_card_features[
                target_regions, target_ids
            ]
        else:
            candidate_static = projected_card_features[candidate_ids]
            target_static = projected_card_features[target_ids]
        token = (
            self.option_type_embedding(categorical[:, 0])
            + self.option_context_embedding(categorical[:, 1])
            + self.option_candidate_embedding(candidate_ids)
            + self.option_target_embedding(target_ids)
            + self.option_attack_embedding(attack_ids)
            + self.option_number_embedding(categorical[:, 5])
            + self.option_count_embedding(categorical[:, 6])
            + self.option_player_relation_embedding(categorical[:, 7])
            + self.option_area_embedding(categorical[:, 8])
            + self.option_in_play_area_embedding(categorical[:, 9])
            + self.option_special_condition_embedding(categorical[:, 10])
            + self.option_numeric_projection(numeric)
            + candidate_static
            + target_static
            + projected_attack_features[attack_ids]
            + self.project_pokemon_dynamic(pokemon_dynamic)
            + self.project_attack_dynamic(attack_dynamic)
        )
        return self.apply_option_token_mlp(token)

    def combine_actions(
        self,
        option_embeddings: torch.Tensor,
        action_option_index: torch.Tensor,
        action_option_offset: torch.Tensor,
    ) -> torch.Tensor:
        if action_option_index.numel() == 0:
            action_embeddings = option_embeddings.new_zeros(
                (action_option_offset.numel() - 1, self.config.d_model)
            )
        else:
            action_embeddings = F.embedding_bag(
                action_option_index,
                option_embeddings,
                action_option_offset,
                mode="sum",
                include_last_offset=True,
            )
        empty = action_option_offset[1:] == action_option_offset[:-1]
        return action_embeddings + empty.unsqueeze(1) * self.no_action_embedding

    @staticmethod
    def _pool_history_options(
        option_embeddings: torch.Tensor,
        option_offsets: torch.Tensor,
        d_model: int,
    ) -> torch.Tensor:
        slots = option_offsets.numel() - 1
        if option_embeddings.size(0) == 0:
            return option_embeddings.new_zeros((slots, d_model))
        indices = torch.arange(
            option_embeddings.size(0), device=option_embeddings.device
        )
        return F.embedding_bag(
            indices,
            option_embeddings,
            option_offsets,
            mode="sum",
            include_last_offset=True,
        )

    def _encode_history_option_rows(
        self,
        categorical: torch.Tensor,
        structural: torch.Tensor,
        pokemon_dynamic: torch.Tensor,
        attack_dynamic: torch.Tensor,
    ) -> torch.Tensor:
        mode = self.config.history_encoding
        token = self.history_option_type_embedding(categorical[:, 0])
        if mode == "basic":
            return token
        if mode == "structural":
            return (
                self.history_option_type_embedding(
                    structural[:, HISTORY_OPTION_TYPE_INDEX]
                )
                + self.history_source_area_embedding(
                    structural[:, HISTORY_SOURCE_AREA_INDEX]
                )
                + self.history_target_area_embedding(
                    structural[:, HISTORY_TARGET_AREA_INDEX]
                )
                + self.history_source_relation_embedding(
                    structural[:, HISTORY_SOURCE_RELATION_INDEX]
                )
                + self.history_target_relation_embedding(
                    structural[:, HISTORY_TARGET_RELATION_INDEX]
                )
                + self.history_number_embedding(
                    structural[:, HISTORY_NUMBER_INDEX]
                )
                + self.history_count_embedding(
                    structural[:, HISTORY_COUNT_INDEX]
                )
                + self.history_special_condition_embedding(
                    structural[:, HISTORY_SPECIAL_CONDITION_INDEX]
                )
            )

        candidate_ids = categorical[:, 2]
        target_ids = categorical[:, 3]
        attack_ids = categorical[:, 4]
        if self.history_candidate_static_projection is None:
            candidate_static = token.new_zeros(token.shape)
            target_static = token.new_zeros(token.shape)
        else:
            candidate_table = self.history_candidate_static_projection(
                self.card_feature_table
            )
            target_table = self.history_target_static_projection(
                self.card_feature_table
            )
            zero_card = candidate_table.new_zeros(
                (1, self.config.d_model)
            )
            candidate_static = torch.cat(
                (candidate_table, zero_card), dim=0
            )[candidate_ids]
            target_static = torch.cat(
                (target_table, zero_card), dim=0
            )[target_ids]
        attack_table = self.history_attack_static_projection(
            self.attack_feature_table
        )
        attack_static = torch.cat(
            (attack_table, attack_table.new_zeros((1, self.config.d_model))),
            dim=0,
        )[attack_ids]
        pokemon_present = (
            (pokemon_dynamic[:, 0] > 0)
            | (pokemon_dynamic[:, POKEMON_DYNAMIC_WORD_DIM] > 0)
        )
        pokemon_token = self.history_pokemon_dynamic_projection(
            pokemon_dynamic
        ) * pokemon_present.unsqueeze(1)
        attack_present = attack_dynamic[:, 0] > 0
        attack_token = self.history_attack_dynamic_projection(
            attack_dynamic
        ) * attack_present.unsqueeze(1)
        return (
            token
            + self.history_candidate_embedding(candidate_ids)
            + self.history_target_embedding(target_ids)
            + self.history_attack_embedding(attack_ids)
            + self.history_number_embedding(categorical[:, 5])
            + self.history_count_embedding(categorical[:, 6])
            + self.history_player_relation_embedding(categorical[:, 7])
            + self.history_area_embedding(categorical[:, 8])
            + self.history_in_play_area_embedding(categorical[:, 9])
            + self.history_special_condition_embedding(categorical[:, 10])
            + candidate_static
            + target_static
            + attack_static
            + pokemon_token
            + attack_token
        )

    def encode_history(
        self,
        select_type: torch.Tensor,
        select_context: torch.Tensor,
        valid: torch.Tensor,
        categorical: torch.Tensor,
        structural: torch.Tensor,
        pokemon_dynamic: torch.Tensor,
        attack_dynamic: torch.Tensor,
        option_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Map three chronological historical actions to one token."""
        if self.config.history_encoding == "off":
            raise RuntimeError("history encoder is disabled")
        batch_size = valid.size(0)
        expected_slots = batch_size * HISTORY_STEPS
        if option_offsets.numel() != expected_slots + 1:
            raise ValueError("history option offsets do not match the batch")
        if categorical.size(0) != structural.size(0):
            raise ValueError("history categorical and structural rows differ")
        option_tokens = self._encode_history_option_rows(
            categorical,
            structural,
            pokemon_dynamic,
            attack_dynamic,
        )
        actions = self._pool_history_options(
            option_tokens,
            option_offsets,
            self.config.d_model,
        )
        flat_valid = valid.reshape(-1).to(dtype=torch.bool)
        actions = actions + self.history_select_type_embedding(
            select_type.reshape(-1)
        )
        actions = actions + self.history_context_embedding(
            select_context.reshape(-1)
        )
        empty = option_offsets[1:] == option_offsets[:-1]
        actions = actions + (
            empty & flat_valid
        ).unsqueeze(1) * self.history_no_action_embedding
        if self.history_action_mlp is not None:
            actions = self.history_action_mlp(actions)
        actions = actions * flat_valid.unsqueeze(1)
        actions = actions.reshape(
            batch_size, HISTORY_STEPS * self.config.d_model
        )
        history_token = self.history_sequence_mlp(actions)
        return history_token * valid.any(dim=1, keepdim=True)

    @staticmethod
    def _encoder_padding_mask(
        own_summary: torch.Tensor,
        opponent_summary: torch.Tensor,
        history_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        slots = torch.arange(BENCH_SLOTS, device=own_summary.device)
        own_count = torch.round(
            own_summary[:, PLAYER_BENCH_COUNT_INDEX] * BENCH_SLOTS
        ).to(torch.long).clamp(0, BENCH_SLOTS)
        opponent_count = torch.round(
            opponent_summary[:, PLAYER_BENCH_COUNT_INDEX] * BENCH_SLOTS
        ).to(torch.long).clamp(0, BENCH_SLOTS)
        own_padding = slots.unsqueeze(0) >= own_count.unsqueeze(1)
        opponent_padding = (
            slots.unsqueeze(0) >= opponent_count.unsqueeze(1)
        )
        fixed_tokens = torch.zeros(
            (own_summary.size(0), ENCODER_TOKENS - 2 * BENCH_SLOTS),
            dtype=torch.bool,
            device=own_summary.device,
        )
        result = torch.cat(
            (own_padding, opponent_padding, fixed_tokens), dim=1
        )
        if history_valid is not None:
            result = torch.cat(
                (result, ~history_valid.to(torch.bool).any(dim=1, keepdim=True)),
                dim=1,
            )
        return result

    def forward(
        self,
        index_encoder,
        value_encoder,
        offset_encoder,
        pokemon_appear,
        own_summary,
        opponent_summary,
        global_summary,
        history_select_type,
        history_select_context,
        history_valid,
        history_option_categorical,
        history_structural,
        history_pokemon_dynamic,
        history_attack_dynamic,
        history_option_offset,
        option_categorical,
        option_numeric,
        pokemon_dynamic,
        attack_dynamic,
        action_option_index,
        action_option_offset,
    ):
        cfg = self.config
        projected_card_features = self.project_card_features()
        projected_attack_features = self.project_attack_features()
        encoded = self.encoder_bag(
            index_encoder,
            offset_encoder,
            per_sample_weights=value_encoder,
            projected_card_features=projected_card_features,
        )
        batch_size = own_summary.size(0)
        encoded = encoded.reshape(
            batch_size,
            ENCODER_TOKENS,
            cfg.d_model,
        )
        if self.pokemon_appear_embedding is not None:
            expected = (batch_size, POKEMON_ENCODER_TOKENS)
            if tuple(pokemon_appear.shape) != expected:
                raise ValueError(
                    "pokemon_appear must have shape "
                    f"{expected}, found {tuple(pokemon_appear.shape)}"
                )
            pokemon_tokens = (
                encoded[:, :POKEMON_ENCODER_TOKENS]
                + self.pokemon_appear_embedding(pokemon_appear)
            )
            encoded = torch.cat(
                (pokemon_tokens, encoded[:, POKEMON_ENCODER_TOKENS:]),
                dim=1,
            )
        encoded = torch.cat(
            (
                encoded[:, :18],
                self.own_summary_projection(own_summary).unsqueeze(1),
                self.opponent_summary_projection(opponent_summary).unsqueeze(1),
                encoded[:, 20:26],
                self.global_summary_projection(global_summary).unsqueeze(1),
            ),
            dim=1,
        )
        encoded = self.apply_region_token_mlps(encoded)
        if cfg.history_encoding != "off":
            history_token = self.encode_history(
                history_select_type,
                history_select_context,
                history_valid,
                history_option_categorical,
                history_structural,
                history_pokemon_dynamic,
                history_attack_dynamic,
                history_option_offset,
            )
            encoded = torch.cat((encoded, history_token.unsqueeze(1)), dim=1)
        if self.encoder_input_norm is not None:
            encoded = self.embedding_dropout(
                self.encoder_input_norm(encoded)
            )
        encoded = encoded.transpose(0, 1)
        encoder_padding_mask = self._encoder_padding_mask(
            own_summary,
            opponent_summary,
            history_valid if cfg.history_encoding != "off" else None,
        )
        encoder_out = self.encoder(
            encoded,
            src_key_padding_mask=encoder_padding_mask,
        )
        option_embeddings = self.encode_options(
            option_categorical,
            option_numeric,
            pokemon_dynamic,
            attack_dynamic,
            projected_card_features,
            projected_attack_features,
        )
        policy = self.combine_actions(
            option_embeddings,
            action_option_index,
            action_option_offset,
        )
        if self.action_input_norm is not None:
            policy = self.embedding_dropout(
                self.action_input_norm(policy)
            )
        policy = policy.reshape(batch_size, -1, cfg.d_model).transpose(0, 1)
        # Every decoder layer cross-attends to the same encoder output. There
        # is deliberately no self-attention between candidate actions.
        for layer in self.decoder:
            policy = layer(policy, encoder_out, encoder_padding_mask)
        # Return raw logits. Cross entropy applies log-softmax internally, and
        # argmax(logits) is identical to argmax(softmax(logits)) at inference.
        return self.decoder_fc(policy).transpose(0, 1).reshape(batch_size, -1)
