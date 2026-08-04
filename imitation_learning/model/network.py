"""Transformer architecture from the source notebook."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import torch
import torch.nn.functional as F

from model.attack_features import ATTACK_FEATURE_DIM
from model.card_features import CARD_FEATURE_DIM
from model.features import (
    COMPONENT_APPEAR,
    COMPONENT_AREA_CARD,
    COMPONENT_ENERGY_CARD,
    COMPONENT_HP,
    COMPONENT_KIND_COUNT,
    COMPONENT_POKEMON_CARD,
    COMPONENT_TOOL_CARD,
)


ENCODER_TOKENS = 26
BENCH_SLOTS = 8
PLAYER_BENCH_COUNT_INDEX = 10
OWN_SUMMARY_DIM = 69
OPPONENT_SUMMARY_DIM = 71
GLOBAL_SUMMARY_DIM = 73
OPTION_TYPE_COUNT = 17
OPTION_CONTEXT_COUNT = 49
OPTION_NUMERIC_DIM = 76

OPTION_PLAYER_OFFSET = 16
OPTION_AREA_OFFSET = 19
OPTION_IN_PLAY_AREA_OFFSET = 32

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
ENCODER_CARD_REGION_NAMES = (
    "own_bench",
    "opponent_bench",
    "own_active",
    "opponent_active",
    "own_discard",
    "opponent_discard",
    "own_hand",
    "own_deck",
    "stadium",
)

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
    encoder_size: int = 22_000
    d_model: int = 128
    num_heads: int = 2
    d_feedforward: int = 256
    encoder_layers: int = 1
    decoder_layers: int = 1
    norm_mode: str = "postnorm"
    summary_mlp_layers: int = 1
    card_mlp_layers: int = 1
    option_numeric_mlp_layers: int = 1
    card_mlp_scope: str = "shared"
    pokemon_appear_embedding: bool = False
    bench_region_encoder_layers: int = 0
    active_region_encoder_layers: int = 0
    discard_region_encoder_layers: int = 0
    hand_region_encoder_layers: int = 0
    deck_region_encoder_layers: int = 0
    option_encoder_layers: int = 0

    def __post_init__(self) -> None:
        if self.norm_mode not in {"prenorm", "postnorm"}:
            raise ValueError("norm_mode must be prenorm or postnorm")
        if self.summary_mlp_layers < 1:
            raise ValueError("summary_mlp_layers must be >= 1")
        if self.card_mlp_layers < 0:
            raise ValueError("card_mlp_layers must be >= 0")
        if self.card_mlp_scope not in {"shared", "region"}:
            raise ValueError("card_mlp_scope must be shared or region")
        if self.option_numeric_mlp_layers < 1:
            raise ValueError("option_numeric_mlp_layers must be >= 1")
        if type(self.pokemon_appear_embedding) is not bool:
            raise ValueError("pokemon_appear_embedding must be a boolean")
        for name in (
            "bench_region_encoder_layers",
            "active_region_encoder_layers",
            "discard_region_encoder_layers",
            "hand_region_encoder_layers",
            "deck_region_encoder_layers",
            "option_encoder_layers",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be an integer >= 0")

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


class MaskedSetEncoder(torch.nn.Module):
    """Encode an unordered padded token set and return its first token."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_feedforward: int,
        layers: int,
        norm_mode: str,
        *,
        use_cls: bool,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("MaskedSetEncoder requires at least one layer")
        self.cls = (
            torch.nn.Parameter(torch.zeros(d_model)) if use_cls else None
        )
        prenorm = norm_mode == "prenorm"
        layer = torch.nn.TransformerEncoderLayer(
            d_model,
            num_heads,
            d_feedforward,
            dropout=0,
            batch_first=True,
            norm_first=prenorm,
        )
        final_norm = torch.nn.LayerNorm(d_model) if prenorm else None
        self.encoder = torch.nn.TransformerEncoder(
            layer,
            layers,
            norm=final_norm,
            enable_nested_tensor=False,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.cls is not None:
            cls = self.cls.view(1, 1, -1).expand(tokens.size(0), 1, -1)
            tokens = torch.cat((cls, tokens), dim=1)
            cls_mask = torch.zeros(
                (padding_mask.size(0), 1),
                dtype=torch.bool,
                device=padding_mask.device,
            )
            padding_mask = torch.cat((cls_mask, padding_mask), dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        return encoded[:, 0]


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
            x = x + attended
            return x + self.fc2(
                torch.nn.functional.relu(self.fc1(self.norm2(x)))
            )
        y, _ = self.attention(
            x,
            encoder_out,
            encoder_out,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
        )
        residual = self.norm1(x + y)
        y = self.fc2(torch.nn.functional.relu(self.fc1(residual)))
        return self.norm2(residual + y)


class PTCGTransformer(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        card_feature_table: torch.Tensor,
        attack_feature_table: torch.Tensor,
    ):
        super().__init__()
        self.config = config
        self.encoder_token_count = ENCODER_TOKENS
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
        self.encoder_card_embeddings = torch.nn.ModuleDict(
            {
                name: torch.nn.Embedding(
                    config.card_count + 1,
                    config.d_model,
                    padding_idx=config.card_count,
                )
                for name in ENCODER_CARD_REGION_NAMES
            }
        )
        self.component_kind_embedding = torch.nn.Embedding(
            COMPONENT_KIND_COUNT, config.d_model
        )
        self.hp_component_embedding = torch.nn.Parameter(
            torch.zeros(config.d_model)
        )
        self.pokemon_appear_embedding = (
            torch.nn.Embedding(3, config.d_model, padding_idx=0)
            if config.pokemon_appear_embedding
            else None
        )
        self.own_bench_region_encoder = self._make_region_encoder(
            config.bench_region_encoder_layers, use_cls=False
        )
        self.opponent_bench_region_encoder = self._make_region_encoder(
            config.bench_region_encoder_layers, use_cls=False
        )
        self.own_active_region_encoder = self._make_region_encoder(
            config.active_region_encoder_layers, use_cls=False
        )
        self.opponent_active_region_encoder = self._make_region_encoder(
            config.active_region_encoder_layers, use_cls=False
        )
        self.own_discard_region_encoder = self._make_region_encoder(
            config.discard_region_encoder_layers, use_cls=True
        )
        self.opponent_discard_region_encoder = self._make_region_encoder(
            config.discard_region_encoder_layers, use_cls=True
        )
        self.own_hand_region_encoder = self._make_region_encoder(
            config.hand_region_encoder_layers, use_cls=True
        )
        self.own_deck_region_encoder = self._make_region_encoder(
            config.deck_region_encoder_layers, use_cls=True
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
        self.option_numeric_projection = _projection_mlp(
            OPTION_NUMERIC_DIM,
            config.d_model,
            config.option_numeric_mlp_layers,
        )
        self.attack_feature_projection = torch.nn.Linear(
            ATTACK_FEATURE_DIM, config.d_model
        )
        self.no_action_embedding = torch.nn.Parameter(
            torch.zeros(config.d_model)
        )
        self.option_region_encoder = self._make_region_encoder(
            config.option_encoder_layers, use_cls=True
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

    def _make_region_encoder(
        self,
        layers: int,
        *,
        use_cls: bool,
    ) -> MaskedSetEncoder | None:
        if layers == 0:
            return None
        return MaskedSetEncoder(
            self.config.d_model,
            self.config.num_heads,
            self.config.d_feedforward,
            layers,
            self.config.norm_mode,
            use_cls=use_cls,
        )

    @staticmethod
    def _aggregate_region(
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        encoder: MaskedSetEncoder | None,
    ) -> torch.Tensor:
        empty = padding_mask.all(dim=1)
        if encoder is None:
            return tokens.masked_fill(padding_mask.unsqueeze(-1), 0).sum(dim=1)
        safe_tokens = tokens
        safe_mask = padding_mask
        if encoder.cls is None and torch.any(empty):
            safe_tokens = tokens.clone()
            safe_mask = padding_mask.clone()
            safe_tokens[empty, 0] = 0
            safe_mask[empty, 0] = False
        output = encoder(safe_tokens, safe_mask)
        return output.masked_fill(empty.unsqueeze(1), 0)

    def _embed_region_components(
        self,
        kinds: torch.Tensor,
        entity_ids: torch.Tensor,
        values: torch.Tensor,
        padding_mask: torch.Tensor,
        region_name: str,
        projected_card_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        region_index = CARD_REGION_INDEX[region_name]
        card_kinds = (
            (kinds == COMPONENT_POKEMON_CARD)
            | (kinds == COMPONENT_TOOL_CARD)
            | (kinds == COMPONENT_ENERGY_CARD)
            | (kinds == COMPONENT_AREA_CARD)
        )
        card_ids = entity_ids.clamp(0, self.config.card_count)
        learned_cards = self.encoder_card_embeddings[region_name](card_ids)
        if projected_card_features.ndim == 3:
            static_cards = projected_card_features[region_index, card_ids]
        else:
            static_cards = projected_card_features[card_ids]
        kind_tokens = self.component_kind_embedding(kinds)
        tokens = torch.where(
            card_kinds.unsqueeze(-1),
            learned_cards + static_cards + kind_tokens,
            torch.zeros_like(learned_cards),
        )
        hp_tokens = (
            kind_tokens
            + values.unsqueeze(-1) * self.hp_component_embedding
        )
        tokens = torch.where(
            (kinds == COMPONENT_HP).unsqueeze(-1), hp_tokens, tokens
        )
        if self.pokemon_appear_embedding is not None:
            appear_tokens = (
                kind_tokens
                + self.pokemon_appear_embedding(entity_ids.clamp(0, 2))
            )
            tokens = torch.where(
                (kinds == COMPONENT_APPEAR).unsqueeze(-1),
                appear_tokens,
                tokens,
            )
        else:
            padding_mask = padding_mask | (kinds == COMPONENT_APPEAR)
        return tokens, padding_mask

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
        numeric: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        option_types = categorical[:, 0]
        player_relation = numeric[
            :, OPTION_PLAYER_OFFSET:OPTION_PLAYER_OFFSET + 3
        ].argmax(dim=1)
        areas = numeric[
            :, OPTION_AREA_OFFSET:OPTION_AREA_OFFSET + 13
        ].argmax(dim=1)
        in_play_areas = numeric[
            :, OPTION_IN_PLAY_AREA_OFFSET:OPTION_IN_PLAY_AREA_OFFSET + 13
        ].argmax(dim=1)

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

    def encode_options(
        self,
        categorical: torch.Tensor,
        numeric: torch.Tensor,
        projected_card_features: torch.Tensor,
        projected_attack_features: torch.Tensor,
    ) -> torch.Tensor:
        candidate_ids = categorical[:, 2]
        target_ids = categorical[:, 3]
        attack_ids = categorical[:, 4]
        if projected_card_features.ndim == 3:
            candidate_regions, target_regions = self.decoder_card_regions(
                categorical, numeric
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
        components = torch.stack(
            (
                self.option_type_embedding(categorical[:, 0]),
                self.option_context_embedding(categorical[:, 1]),
                self.option_candidate_embedding(candidate_ids)
                + candidate_static,
                self.option_target_embedding(target_ids) + target_static,
                self.option_attack_embedding(attack_ids)
                + projected_attack_features[attack_ids],
                self.option_numeric_projection(numeric),
            ),
            dim=1,
        )
        component_mask = torch.zeros(
            components.shape[:2],
            dtype=torch.bool,
            device=components.device,
        )
        component_mask[:, 2] = candidate_ids == self.config.card_count
        component_mask[:, 3] = target_ids == self.config.card_count
        component_mask[:, 4] = attack_ids == self.config.attack_count
        return self._aggregate_region(
            components, component_mask, self.option_region_encoder
        )

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

    def _encode_outer_group(
        self,
        kinds: torch.Tensor,
        entity_ids: torch.Tensor,
        values: torch.Tensor,
        component_mask: torch.Tensor,
        start: int,
        end: int,
        region_name: str,
        encoder: MaskedSetEncoder | None,
        projected_card_features: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, width = kinds.shape
        group_size = end - start
        group_kinds = kinds[:, start:end].reshape(-1, width)
        group_ids = entity_ids[:, start:end].reshape(-1, width)
        group_values = values[:, start:end].reshape(-1, width)
        group_mask = component_mask[:, start:end].reshape(-1, width)
        tokens, group_mask = self._embed_region_components(
            group_kinds,
            group_ids,
            group_values,
            group_mask,
            region_name,
            projected_card_features,
        )
        aggregated = self._aggregate_region(tokens, group_mask, encoder)
        return aggregated.reshape(batch_size, group_size, -1)

    def build_outer_tokens(
        self,
        component_kind: torch.Tensor,
        component_id: torch.Tensor,
        component_value: torch.Tensor,
        component_mask: torch.Tensor,
        own_summary: torch.Tensor,
        opponent_summary: torch.Tensor,
        global_summary: torch.Tensor,
        projected_card_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = component_kind.size(0)
        outer = component_value.new_zeros(
            (batch_size, ENCODER_TOKENS, self.config.d_model)
        )
        groups = (
            (0, 8, "own_bench", self.own_bench_region_encoder),
            (8, 16, "opponent_bench", self.opponent_bench_region_encoder),
            (16, 17, "own_active", self.own_active_region_encoder),
            (17, 18, "opponent_active", self.opponent_active_region_encoder),
            (20, 21, "own_discard", self.own_discard_region_encoder),
            (21, 22, "opponent_discard", self.opponent_discard_region_encoder),
            (22, 23, "own_hand", self.own_hand_region_encoder),
            (23, 24, "own_deck", self.own_deck_region_encoder),
            (24, 25, "stadium", None),
        )
        for start, end, region_name, encoder in groups:
            outer[:, start:end] = self._encode_outer_group(
                component_kind,
                component_id,
                component_value,
                component_mask,
                start,
                end,
                region_name,
                encoder,
                projected_card_features,
            )
        outer[:, 18] = self.own_summary_projection(own_summary)
        outer[:, 19] = self.opponent_summary_projection(opponent_summary)
        outer[:, 25] = self.global_summary_projection(global_summary)
        outer_mask = component_mask.all(dim=2)
        outer_mask[:, 18] = False
        outer_mask[:, 19] = False
        outer_mask[:, 25] = False
        return outer, outer_mask

    def forward(
        self,
        encoder_component_kind,
        encoder_component_id,
        encoder_component_value,
        encoder_component_mask,
        own_summary,
        opponent_summary,
        global_summary,
        option_categorical,
        option_numeric,
        action_option_index,
        action_option_offset,
    ):
        cfg = self.config
        projected_card_features = self.project_card_features()
        projected_attack_features = self.project_attack_features()
        batch_size = own_summary.size(0)
        encoded, encoder_padding_mask = self.build_outer_tokens(
            encoder_component_kind,
            encoder_component_id,
            encoder_component_value,
            encoder_component_mask,
            own_summary,
            opponent_summary,
            global_summary,
            projected_card_features,
        )
        encoded = encoded.transpose(0, 1)
        encoder_out = self.encoder(
            encoded,
            src_key_padding_mask=encoder_padding_mask,
        )
        option_embeddings = self.encode_options(
            option_categorical,
            option_numeric,
            projected_card_features,
            projected_attack_features,
        )
        policy = self.combine_actions(
            option_embeddings,
            action_option_index,
            action_option_offset,
        )
        policy = policy.reshape(batch_size, -1, cfg.d_model).transpose(0, 1)
        # Every decoder layer cross-attends to the same encoder output. There
        # is deliberately no self-attention between candidate actions.
        for layer in self.decoder:
            policy = layer(policy, encoder_out, encoder_padding_mask)
        # Return raw logits. Cross entropy applies log-softmax internally, and
        # argmax(logits) is identical to argmax(softmax(logits)) at inference.
        return self.decoder_fc(policy).transpose(0, 1).reshape(batch_size, -1)
