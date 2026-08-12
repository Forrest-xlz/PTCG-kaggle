"""Packed, mmap-backed feature storage for large behavior-cloning datasets."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import AbstractSet, Iterable, Mapping

import numpy as np


CACHE_SCHEMA_VERSION = 16
ENCODER_WORDS = 26
POKEMON_ENCODER_TOKENS = 18
OWN_SUMMARY_DIM = 69
OPPONENT_SUMMARY_DIM = 71
GLOBAL_SUMMARY_DIM = 73
OPTION_CATEGORICAL_DIM = 11
OPTION_NUMERIC_DIM = 5
POKEMON_DYNAMIC_DIM = 46
ATTACK_DYNAMIC_DIM = 6
HISTORY_STEPS = 3
HISTORY_STRUCTURAL_DIM = 8
MAX_ACTIONS = 64
ALIGNMENT = 64
PLAYER_RESULT_WIN = 1
PLAYER_RESULT_LOSS = 2
PLAYER_RESULT_DRAW = 3
PLAYER_RESULTS = frozenset(
    {PLAYER_RESULT_WIN, PLAYER_RESULT_LOSS, PLAYER_RESULT_DRAW}
)

SECTION_DTYPES = {
    "encoder_index": np.dtype("<u2"),
    "encoder_value": np.dtype("<f2"),
    "encoder_ptr": np.dtype("<u4"),
    "encoder_offset": np.dtype("<u2"),
    "encoder_pokemon_appear": np.dtype("u1"),
    "own_summary": np.dtype("<f2"),
    "opponent_summary": np.dtype("<f2"),
    "global_summary": np.dtype("<f2"),
    "option_categorical": np.dtype("<u2"),
    "option_numeric": np.dtype("<f2"),
    "pokemon_dynamic": np.dtype("<f2"),
    "attack_dynamic": np.dtype("<f2"),
    "option_ptr": np.dtype("<u4"),
    "history_select_type": np.dtype("u1"),
    "history_select_context": np.dtype("u1"),
    "history_valid": np.dtype("u1"),
    "history_option_categorical": np.dtype("<u2"),
    "history_structural": np.dtype("<u2"),
    "history_pokemon_dynamic": np.dtype("<f2"),
    "history_attack_dynamic": np.dtype("<f2"),
    "history_option_ptr": np.dtype("<u4"),
    "action_option_index": np.dtype("<u2"),
    "action_option_ptr": np.dtype("<u4"),
    "action_option_offset": np.dtype("<u2"),
    "action_option_offset_ptr": np.dtype("<u4"),
    "target": np.dtype("u1"),
    "action_count": np.dtype("u1"),
    "episode_key": np.dtype("<u4"),
    "deck_key": np.dtype("<u8"),
    "player_result": np.dtype("u1"),
}


@dataclass(slots=True)
class FeatureRecord:
    encoder_index: list[int]
    encoder_value: list[float]
    encoder_offset: list[int]
    encoder_pokemon_appear: list[int]
    own_summary: list[float]
    opponent_summary: list[float]
    global_summary: list[float]
    option_categorical: list[int]
    option_numeric: list[float]
    pokemon_dynamic: list[float]
    attack_dynamic: list[float]
    action_option_index: list[int]
    action_option_offset: list[int]
    target: int
    action_count: int
    episode_key: int
    deck_key: int
    history_select_type: list[int]
    history_select_context: list[int]
    history_valid: list[int]
    history_option_categorical: list[int]
    history_structural: list[int]
    history_pokemon_dynamic: list[float]
    history_attack_dynamic: list[float]
    history_option_offset: list[int]
    player_result: int = PLAYER_RESULT_WIN


@dataclass(frozen=True, slots=True)
class FeatureView:
    encoder_index: np.ndarray
    encoder_value: np.ndarray
    encoder_offset: np.ndarray
    encoder_pokemon_appear: np.ndarray
    own_summary: np.ndarray
    opponent_summary: np.ndarray
    global_summary: np.ndarray
    option_categorical: np.ndarray
    option_numeric: np.ndarray
    pokemon_dynamic: np.ndarray
    attack_dynamic: np.ndarray
    action_option_index: np.ndarray
    action_option_offset: np.ndarray
    target: int
    action_count: int
    episode_key: int
    deck_key: int
    player_result: int
    history_select_type: np.ndarray
    history_select_context: np.ndarray
    history_valid: np.ndarray
    history_option_categorical: np.ndarray
    history_structural: np.ndarray
    history_pokemon_dynamic: np.ndarray
    history_attack_dynamic: np.ndarray
    history_option_offset: np.ndarray


@dataclass(frozen=True, slots=True)
class IndexBatch:
    global_ids: np.ndarray


@dataclass(frozen=True, slots=True)
class CachedBatch:
    encoder_index: np.ndarray
    encoder_value: np.ndarray
    encoder_offset: np.ndarray
    encoder_pokemon_appear: np.ndarray
    own_summary: np.ndarray
    opponent_summary: np.ndarray
    global_summary: np.ndarray
    option_categorical: np.ndarray
    option_numeric: np.ndarray
    pokemon_dynamic: np.ndarray
    attack_dynamic: np.ndarray
    action_option_index: np.ndarray
    action_option_offset: np.ndarray
    target: np.ndarray
    action_count: np.ndarray
    history_select_type: np.ndarray
    history_select_context: np.ndarray
    history_valid: np.ndarray
    history_option_categorical: np.ndarray
    history_structural: np.ndarray
    history_pokemon_dynamic: np.ndarray
    history_attack_dynamic: np.ndarray
    history_option_offset: np.ndarray

    def __len__(self) -> int:
        return int(self.target.size)


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    train: np.ndarray
    isolation: np.ndarray
    isolation_masks: dict[str, np.ndarray]
    isolation_sample_counts: dict[str, int]
    isolation_union_replays: int
    in_distribution: np.ndarray
    in_distribution_expert_mask: np.ndarray
    in_distribution_top_deck_mask: np.ndarray
    in_distribution_expert_top_deck_mask: np.ndarray
    in_distribution_top_deck_masks: tuple[np.ndarray, ...]
    in_distribution_expert_top_deck_masks: tuple[np.ndarray, ...]
    latest: np.ndarray
    latest_expert_mask: np.ndarray
    latest_top_deck_mask: np.ndarray
    latest_expert_top_deck_mask: np.ndarray
    latest_top_deck_masks: tuple[np.ndarray, ...]
    latest_expert_top_deck_masks: tuple[np.ndarray, ...]
    latest_date: tuple[int, int]
    eligible_train_samples: int
    eligible_train_replays: int
    selected_train_replays: int
    loser_augmentation_counts: dict[
        tuple[int, int], "LoserAugmentationCounts"
    ]
    loser_augmentation_replays: int
    loser_augmentation_samples: int
    loser_fraction_in_train: float


@dataclass(frozen=True, slots=True)
class LoserAugmentationCounts:
    score_eligible_episodes: int
    after_validation_episodes: int
    selected_train_episodes: int
    loser_samples: int


def stable_episode_key(episode_id: object) -> int:
    payload = str(episode_id).encode("utf-8")
    return int.from_bytes(
        hashlib.blake2s(payload, digest_size=4).digest(), "little"
    )


def stable_deck_key(deck: Iterable[int]) -> int:
    raw_card_ids = list(deck)
    if any(
        isinstance(card_id, bool)
        or not isinstance(card_id, (int, np.integer))
        for card_id in raw_card_ids
    ):
        raise ValueError("deck card IDs must be integers")
    card_ids = sorted(int(card_id) for card_id in raw_card_ids)
    if len(card_ids) != 60:
        raise ValueError(f"deck must contain exactly 60 cards, found {len(card_ids)}")
    uint32_max = np.iinfo(np.uint32).max
    if any(card_id < 0 or card_id > uint32_max for card_id in card_ids):
        raise ValueError("deck card IDs must fit uint32")
    payload = np.asarray(card_ids, dtype="<u4").tobytes()
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(), "little"
    )


def parse_source_date(name: str) -> tuple[int, int]:
    match = re.search(r"(?<!\d)(\d{1,2})\.(\d{1,2})(?!\d)", str(name))
    if match is None:
        raise ValueError(f"cannot parse month.day from cache source: {name}")
    month, day = map(int, match.groups())
    if not 1 <= month <= 12 or not 1 <= day <= 31:
        raise ValueError(f"invalid month.day cache source: {name}")
    return month, day


def _mix_episode_keys(keys: np.ndarray, seed: int) -> np.ndarray:
    mixed = np.asarray(keys, dtype=np.uint32).copy()
    mixed ^= np.uint32(int(seed) & 0xFFFFFFFF)
    mixed ^= mixed >> np.uint32(16)
    mixed *= np.uint32(0x7FEB352D)
    mixed ^= mixed >> np.uint32(15)
    mixed *= np.uint32(0x846CA68B)
    mixed ^= mixed >> np.uint32(16)
    return mixed


def _validate_unsigned(name: str, values: Iterable[int], maximum: int) -> None:
    for value in values:
        if int(value) < 0 or int(value) > maximum:
            raise ValueError(f"{name} value {value} is outside [0, {maximum}]")


def _validate_dense_summary(name: str, values: list[float], width: int) -> None:
    if len(values) != width:
        raise ValueError(f"{name} must contain {width} values")
    full_precision = np.asarray(values, dtype=np.float32)
    narrowed = full_precision.astype(np.float16)
    if not np.all(np.isfinite(full_precision)) or not np.all(np.isfinite(narrowed)):
        raise ValueError(f"{name} contains a non-finite or float16-overflow value")


class PackedShardWriter:
    """Stream feature records into one atomic packed cache directory."""

    def __init__(
        self,
        destination: Path,
        signature: dict,
        source: dict,
        flush_samples: int = 2048,
    ):
        self.destination = Path(destination)
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.signature = dict(signature)
        self.source = dict(source)
        self.flush_samples = max(1, int(flush_samples))
        self._temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{self.destination.name}.",
                dir=self.destination.parent,
            )
        )
        self._component_dir = self._temporary / "components"
        self._component_dir.mkdir()
        self._handles = {
            name: (self._component_dir / f"{name}.bin").open("wb")
            for name in SECTION_DTYPES
        }
        self._buffers: dict[str, list] = {name: [] for name in SECTION_DTYPES}
        self._counts = {name: 0 for name in SECTION_DTYPES}
        self._samples = 0
        self._buffered_samples = 0
        self._encoder_nnz = 0
        self._option_count = 0
        self._history_option_count = 0
        self._action_option_nnz = 0
        self._action_offset_count = 0
        self._closed = False

        self._append("encoder_ptr", 0)
        self._append("option_ptr", 0)
        self._append("history_option_ptr", 0)
        self._append("action_option_ptr", 0)
        self._append("action_option_offset_ptr", 0)

    def _append(self, name: str, value) -> None:
        self._buffers[name].append(value)

    def add(self, record: FeatureRecord) -> None:
        if self._closed:
            raise RuntimeError("cannot add to a closed cache writer")
        if len(record.encoder_index) != len(record.encoder_value):
            raise ValueError("encoder_index and encoder_value lengths differ")
        if len(record.encoder_offset) != ENCODER_WORDS:
            raise ValueError(
                f"encoder_offset must contain {ENCODER_WORDS} words"
            )
        if len(record.encoder_pokemon_appear) != POKEMON_ENCODER_TOKENS:
            raise ValueError(
                "encoder_pokemon_appear must contain 18 values"
            )
        if any(
            value < 0 or value > 2
            for value in record.encoder_pokemon_appear
        ):
            raise ValueError(
                "encoder_pokemon_appear values must be in [0, 2]"
            )
        _validate_dense_summary("own_summary", record.own_summary, OWN_SUMMARY_DIM)
        _validate_dense_summary(
            "opponent_summary", record.opponent_summary, OPPONENT_SUMMARY_DIM
        )
        _validate_dense_summary(
            "global_summary", record.global_summary, GLOBAL_SUMMARY_DIM
        )
        if not 1 <= int(record.action_count) <= MAX_ACTIONS:
            raise ValueError(f"action_count must be in [1, {MAX_ACTIONS}]")
        if len(record.option_categorical) % OPTION_CATEGORICAL_DIM:
            raise ValueError("option_categorical has an invalid width")
        option_count = len(record.option_categorical) // OPTION_CATEGORICAL_DIM
        dynamic_widths = {
            "option_numeric": OPTION_NUMERIC_DIM,
            "pokemon_dynamic": POKEMON_DYNAMIC_DIM,
            "attack_dynamic": ATTACK_DYNAMIC_DIM,
        }
        for name, width in dynamic_widths.items():
            if len(getattr(record, name)) != option_count * width:
                raise ValueError(f"{name} does not align with options")
        for name in (
            "history_select_type",
            "history_select_context",
            "history_valid",
        ):
            if len(getattr(record, name)) != HISTORY_STEPS:
                raise ValueError(f"{name} must contain {HISTORY_STEPS} values")
        if any(value not in (0, 1) for value in record.history_valid):
            raise ValueError("history_valid values must be 0 or 1")
        if len(record.history_option_offset) != HISTORY_STEPS + 1:
            raise ValueError(
                "history_option_offset must contain four boundaries"
            )
        if len(record.history_option_categorical) % OPTION_CATEGORICAL_DIM:
            raise ValueError("history_option_categorical has an invalid width")
        history_option_count = (
            len(record.history_option_categorical) // OPTION_CATEGORICAL_DIM
        )
        history_offsets = np.asarray(
            record.history_option_offset, dtype=np.int64
        )
        if (
            history_offsets[0] != 0
            or history_offsets[-1] != history_option_count
            or np.any(history_offsets[1:] < history_offsets[:-1])
        ):
            raise ValueError("history_option_offset boundaries are invalid")
        history_widths = {
            "history_structural": HISTORY_STRUCTURAL_DIM,
            "history_pokemon_dynamic": POKEMON_DYNAMIC_DIM,
            "history_attack_dynamic": ATTACK_DYNAMIC_DIM,
        }
        for name, width in history_widths.items():
            if len(getattr(record, name)) != history_option_count * width:
                raise ValueError(f"{name} does not align with history options")
        for slot, valid in enumerate(record.history_valid):
            if not valid and history_offsets[slot + 1] != history_offsets[slot]:
                raise ValueError("padded history slot contains options")
        if len(record.action_option_offset) != int(record.action_count) + 1:
            raise ValueError(
                "action_option_offset length must equal action_count + 1"
            )
        action_offsets = np.asarray(
            record.action_option_offset, dtype=np.int64
        )
        if (
            action_offsets[0] != 0
            or action_offsets[-1] != len(record.action_option_index)
            or np.any(action_offsets[1:] < action_offsets[:-1])
        ):
            raise ValueError("action_option_offset boundaries are invalid")
        if any(
            index < 0 or index >= option_count
            for index in record.action_option_index
        ):
            raise ValueError("action_option_index contains an invalid option")
        if not 0 <= int(record.target) < int(record.action_count):
            raise ValueError("target must be smaller than action_count")
        if not 0 <= int(record.episode_key) <= np.iinfo(np.uint32).max:
            raise ValueError("episode_key must fit uint32")
        if not 0 <= int(record.deck_key) <= np.iinfo(np.uint64).max:
            raise ValueError("deck_key must fit uint64")
        if int(record.player_result) not in PLAYER_RESULTS:
            raise ValueError("player_result must be win, loss, or draw")

        _validate_unsigned("encoder_index", record.encoder_index, np.iinfo(np.uint16).max)
        _validate_unsigned("encoder_offset", record.encoder_offset, np.iinfo(np.uint16).max)
        _validate_unsigned(
            "option_categorical",
            record.option_categorical,
            np.iinfo(np.uint16).max,
        )
        _validate_unsigned(
            "history_option_categorical",
            record.history_option_categorical,
            np.iinfo(np.uint16).max,
        )
        _validate_unsigned(
            "history_structural",
            record.history_structural,
            np.iinfo(np.uint16).max,
        )
        _validate_unsigned(
            "action_option_index",
            record.action_option_index,
            np.iinfo(np.uint16).max,
        )
        _validate_unsigned(
            "action_option_offset",
            record.action_option_offset,
            np.iinfo(np.uint16).max,
        )

        encoder_values = np.asarray(record.encoder_value, dtype=np.float32)
        narrowed = encoder_values.astype(np.float16)
        if not np.all(np.isfinite(encoder_values)) or not np.all(np.isfinite(narrowed)):
            raise ValueError("encoder_value contains a non-finite or float16-overflow value")
        for name in (
            "option_numeric",
            "pokemon_dynamic",
            "attack_dynamic",
            "history_pokemon_dynamic",
            "history_attack_dynamic",
        ):
            full_precision = np.asarray(getattr(record, name), dtype=np.float32)
            narrowed = full_precision.astype(np.float16)
            if not np.all(np.isfinite(full_precision)) or not np.all(
                np.isfinite(narrowed)
            ):
                raise ValueError(
                    f"{name} contains a non-finite or float16-overflow value"
                )

        next_encoder_nnz = self._encoder_nnz + len(record.encoder_index)
        next_option_count = self._option_count + option_count
        next_history_option_count = (
            self._history_option_count + history_option_count
        )
        next_action_option_nnz = (
            self._action_option_nnz + len(record.action_option_index)
        )
        next_action_offset_count = (
            self._action_offset_count + len(record.action_option_offset)
        )
        uint32_max = np.iinfo(np.uint32).max
        if max(
            next_encoder_nnz,
            next_option_count,
            next_history_option_count,
            next_action_option_nnz,
            next_action_offset_count,
        ) > uint32_max:
            raise ValueError("cache shard pointer exceeds uint32 range")

        self._buffers["encoder_index"].extend(record.encoder_index)
        self._buffers["encoder_value"].extend(record.encoder_value)
        self._buffers["encoder_offset"].extend(record.encoder_offset)
        self._buffers["encoder_pokemon_appear"].extend(
            record.encoder_pokemon_appear
        )
        self._buffers["own_summary"].extend(record.own_summary)
        self._buffers["opponent_summary"].extend(record.opponent_summary)
        self._buffers["global_summary"].extend(record.global_summary)
        self._buffers["option_categorical"].extend(
            record.option_categorical
        )
        self._buffers["option_numeric"].extend(record.option_numeric)
        self._buffers["pokemon_dynamic"].extend(record.pokemon_dynamic)
        self._buffers["attack_dynamic"].extend(record.attack_dynamic)
        self._buffers["history_select_type"].extend(
            record.history_select_type
        )
        self._buffers["history_select_context"].extend(
            record.history_select_context
        )
        self._buffers["history_valid"].extend(record.history_valid)
        self._buffers["history_option_categorical"].extend(
            record.history_option_categorical
        )
        self._buffers["history_structural"].extend(
            record.history_structural
        )
        self._buffers["history_pokemon_dynamic"].extend(
            record.history_pokemon_dynamic
        )
        self._buffers["history_attack_dynamic"].extend(
            record.history_attack_dynamic
        )
        self._buffers["action_option_index"].extend(
            record.action_option_index
        )
        self._buffers["action_option_offset"].extend(
            record.action_option_offset
        )

        self._encoder_nnz = next_encoder_nnz
        self._option_count = next_option_count
        for boundary in record.history_option_offset[1:]:
            self._append(
                "history_option_ptr",
                self._history_option_count + int(boundary),
            )
        self._history_option_count = next_history_option_count
        self._action_option_nnz = next_action_option_nnz
        self._action_offset_count = next_action_offset_count
        self._append("encoder_ptr", self._encoder_nnz)
        self._append("option_ptr", self._option_count)
        self._append("action_option_ptr", self._action_option_nnz)
        self._append(
            "action_option_offset_ptr", self._action_offset_count
        )
        self._append("target", int(record.target))
        self._append("action_count", int(record.action_count))
        self._append("episode_key", int(record.episode_key))
        self._append("deck_key", int(record.deck_key))
        self._append("player_result", int(record.player_result))
        self._samples += 1
        self._buffered_samples += 1
        if self._buffered_samples >= self.flush_samples:
            self._flush()

    def _flush(self) -> None:
        for name, values in self._buffers.items():
            if not values:
                continue
            array = np.asarray(values, dtype=SECTION_DTYPES[name])
            self._handles[name].write(array.tobytes(order="C"))
            self._counts[name] += int(array.size)
            values.clear()
        self._buffered_samples = 0

    def _close_components(self) -> None:
        for handle in self._handles.values():
            if not handle.closed:
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()

    def finalize(self) -> dict:
        if self._closed:
            raise RuntimeError("cache writer is already closed")
        if self._samples < 1:
            self.abort()
            raise ValueError("cannot finalize an empty cache shard")
        self._flush()
        self._close_components()

        sections = {}
        data_path = self._temporary / "data.bin"
        with data_path.open("wb") as destination:
            for name, dtype in SECTION_DTYPES.items():
                padding = (-destination.tell()) % ALIGNMENT
                if padding:
                    destination.write(b"\0" * padding)
                byte_offset = destination.tell()
                component = self._component_dir / f"{name}.bin"
                with component.open("rb") as source:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
                byte_length = destination.tell() - byte_offset
                expected = self._counts[name] * dtype.itemsize
                if byte_length != expected:
                    raise RuntimeError(
                        f"{name} byte length {byte_length} != expected {expected}"
                    )
                sections[name] = {
                    "dtype": dtype.str,
                    "count": self._counts[name],
                    "byte_offset": byte_offset,
                    "byte_length": byte_length,
                }
            destination.flush()
            os.fsync(destination.fileno())

        shutil.rmtree(self._component_dir)
        metadata = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "samples": self._samples,
            "signature": self.signature,
            "source": self.source,
            "sections": sections,
            "data_bytes": data_path.stat().st_size,
        }
        meta_path = self._temporary / "meta.json"
        meta_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if self.destination.exists():
            raise FileExistsError(f"cache destination already exists: {self.destination}")
        os.replace(self._temporary, self.destination)
        self._closed = True
        return metadata

    def abort(self) -> None:
        if self._closed:
            return
        self._close_components()
        shutil.rmtree(self._temporary, ignore_errors=True)
        self._closed = True

    def __del__(self):
        if not getattr(self, "_closed", True):
            self.abort()


class PackedShard:
    """Read-only mmap view over one packed cache shard."""

    def __init__(self, path: Path, expected_signature: dict | None = None):
        self.path = Path(path)
        meta_path = self.path / "meta.json"
        data_path = self.path / "data.bin"
        if not meta_path.exists() or not data_path.exists():
            raise ValueError(f"incomplete cache shard: {self.path}")
        self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if self.metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError(f"unsupported cache schema in {self.path}")
        if expected_signature is not None:
            actual = self.metadata.get("signature", {})
            for key, value in expected_signature.items():
                if actual.get(key) != value:
                    raise ValueError(
                        f"cache signature mismatch for {key}: "
                        f"{actual.get(key)!r} != {value!r}; rebuild the cache"
                    )
        if data_path.stat().st_size != int(self.metadata.get("data_bytes", -1)):
            raise ValueError(f"cache data size mismatch: {self.path}")

        self._buffer = np.memmap(data_path, dtype=np.uint8, mode="r")
        self.arrays = {}
        for name, expected_dtype in SECTION_DTYPES.items():
            section = self.metadata.get("sections", {}).get(name)
            if section is None:
                raise ValueError(f"cache section {name} is missing in {self.path}")
            dtype = np.dtype(section["dtype"])
            if dtype != expected_dtype:
                raise ValueError(f"cache section {name} has dtype {dtype}, expected {expected_dtype}")
            offset = int(section["byte_offset"])
            count = int(section["count"])
            length = int(section["byte_length"])
            if offset < 0 or length != count * dtype.itemsize:
                raise ValueError(f"invalid cache section metadata for {name}")
            if offset + length > self._buffer.size:
                raise ValueError(f"cache section {name} exceeds data.bin")
            self.arrays[name] = np.ndarray(
                shape=(count,),
                dtype=dtype,
                buffer=self._buffer,
                offset=offset,
            )

        self.samples = int(self.metadata["samples"])
        if self.arrays["encoder_ptr"].size != self.samples + 1:
            raise ValueError("encoder_ptr length does not match sample count")
        for name in (
            "option_ptr",
            "action_option_ptr",
            "action_option_offset_ptr",
        ):
            if self.arrays[name].size != self.samples + 1:
                raise ValueError(
                    f"{name} length does not match sample count"
                )
        if (
            self.arrays["history_option_ptr"].size
            != self.samples * HISTORY_STEPS + 1
        ):
            raise ValueError(
                "history_option_ptr length does not match sample count"
            )
        if self.arrays["encoder_offset"].size != self.samples * ENCODER_WORDS:
            raise ValueError("encoder_offset length does not match sample count")
        if (
            self.arrays["encoder_pokemon_appear"].size
            != self.samples * POKEMON_ENCODER_TOKENS
        ):
            raise ValueError(
                "encoder_pokemon_appear length does not match sample count"
            )
        if np.any(self.arrays["encoder_pokemon_appear"] > 2):
            raise ValueError(
                "encoder_pokemon_appear contains an invalid value"
            )
        dense_widths = {
            "own_summary": OWN_SUMMARY_DIM,
            "opponent_summary": OPPONENT_SUMMARY_DIM,
            "global_summary": GLOBAL_SUMMARY_DIM,
        }
        for name, width in dense_widths.items():
            if self.arrays[name].size != self.samples * width:
                raise ValueError(f"{name} length does not match sample count")
        for name in (
            "history_select_type",
            "history_select_context",
            "history_valid",
        ):
            if self.arrays[name].size != self.samples * HISTORY_STEPS:
                raise ValueError(f"{name} length does not match sample count")
        if np.any(self.arrays["history_valid"] > 1):
            raise ValueError("history_valid contains an invalid value")
        if self.arrays["target"].size != self.samples:
            raise ValueError("target length does not match sample count")
        if self.arrays["action_count"].size != self.samples:
            raise ValueError("action_count length does not match sample count")
        if self.arrays["episode_key"].size != self.samples:
            raise ValueError("episode_key length does not match sample count")
        if self.arrays["deck_key"].size != self.samples:
            raise ValueError("deck_key length does not match sample count")
        if self.arrays["player_result"].size != self.samples:
            raise ValueError("player_result length does not match sample count")
        if not np.isin(
            self.arrays["player_result"], tuple(PLAYER_RESULTS)
        ).all():
            raise ValueError("player_result contains an invalid value")
        option_count = int(self.arrays["option_ptr"][-1])
        if (
            self.arrays["option_categorical"].size
            != option_count * OPTION_CATEGORICAL_DIM
            or self.arrays["option_numeric"].size
            != option_count * OPTION_NUMERIC_DIM
            or self.arrays["pokemon_dynamic"].size
            != option_count * POKEMON_DYNAMIC_DIM
            or self.arrays["attack_dynamic"].size
            != option_count * ATTACK_DYNAMIC_DIM
        ):
            raise ValueError("option feature arrays do not align with option_ptr")
        history_option_count = int(self.arrays["history_option_ptr"][-1])
        if (
            self.arrays["history_option_categorical"].size
            != history_option_count * OPTION_CATEGORICAL_DIM
            or self.arrays["history_structural"].size
            != history_option_count * HISTORY_STRUCTURAL_DIM
            or self.arrays["history_pokemon_dynamic"].size
            != history_option_count * POKEMON_DYNAMIC_DIM
            or self.arrays["history_attack_dynamic"].size
            != history_option_count * ATTACK_DYNAMIC_DIM
        ):
            raise ValueError(
                "history feature arrays do not align with history_option_ptr"
            )
        pointer_targets = {
            "encoder_ptr": self.arrays["encoder_index"].size,
            "option_ptr": option_count,
            "history_option_ptr": history_option_count,
            "action_option_ptr": self.arrays["action_option_index"].size,
            "action_option_offset_ptr": self.arrays[
                "action_option_offset"
            ].size,
        }
        for pointer_name, final_count in pointer_targets.items():
            pointer = self.arrays[pointer_name]
            if int(pointer[0]) != 0 or int(pointer[-1]) != final_count:
                raise ValueError(f"{pointer_name} boundaries are inconsistent")
            if np.any(pointer[1:] < pointer[:-1]):
                raise ValueError(f"{pointer_name} is not monotonic")
        if np.any(self.arrays["action_count"] < 1) or np.any(
            self.arrays["action_count"] > MAX_ACTIONS
        ):
            raise ValueError("action_count contains an invalid value")
        if np.any(self.arrays["target"] >= self.arrays["action_count"]):
            raise ValueError("target must be smaller than action_count")
        expected_offsets = int(self.arrays["action_count"].sum()) + self.samples
        if self.arrays["action_option_offset"].size != expected_offsets:
            raise ValueError(
                "action_option_offset length does not match action counts"
            )

    def __len__(self) -> int:
        return self.samples

    def sample(self, local_id: int) -> FeatureView:
        local_id = int(local_id)
        if not 0 <= local_id < self.samples:
            raise IndexError(local_id)
        encoder_start = int(self.arrays["encoder_ptr"][local_id])
        encoder_end = int(self.arrays["encoder_ptr"][local_id + 1])
        option_start = int(self.arrays["option_ptr"][local_id])
        option_end = int(self.arrays["option_ptr"][local_id + 1])
        history_pointer_start = local_id * HISTORY_STEPS
        history_boundaries = self.arrays["history_option_ptr"][
            history_pointer_start:
            history_pointer_start + HISTORY_STEPS + 1
        ]
        history_start = int(history_boundaries[0])
        history_end = int(history_boundaries[-1])
        action_start = int(self.arrays["action_option_ptr"][local_id])
        action_end = int(self.arrays["action_option_ptr"][local_id + 1])
        offset_start = int(
            self.arrays["action_option_offset_ptr"][local_id]
        )
        offset_end = int(
            self.arrays["action_option_offset_ptr"][local_id + 1]
        )
        encoder_word_start = local_id * ENCODER_WORDS
        pokemon_start = local_id * POKEMON_ENCODER_TOKENS
        own_start = local_id * OWN_SUMMARY_DIM
        opponent_start = local_id * OPPONENT_SUMMARY_DIM
        global_start = local_id * GLOBAL_SUMMARY_DIM
        history_fixed_start = local_id * HISTORY_STEPS
        return FeatureView(
            encoder_index=self.arrays["encoder_index"][encoder_start:encoder_end],
            encoder_value=self.arrays["encoder_value"][encoder_start:encoder_end],
            encoder_offset=self.arrays["encoder_offset"][
                encoder_word_start : encoder_word_start + ENCODER_WORDS
            ],
            encoder_pokemon_appear=self.arrays[
                "encoder_pokemon_appear"
            ][
                pokemon_start:
                pokemon_start + POKEMON_ENCODER_TOKENS
            ],
            own_summary=self.arrays["own_summary"][
                own_start : own_start + OWN_SUMMARY_DIM
            ],
            opponent_summary=self.arrays["opponent_summary"][
                opponent_start : opponent_start + OPPONENT_SUMMARY_DIM
            ],
            global_summary=self.arrays["global_summary"][
                global_start : global_start + GLOBAL_SUMMARY_DIM
            ],
            option_categorical=self.arrays["option_categorical"][
                option_start * OPTION_CATEGORICAL_DIM:
                option_end * OPTION_CATEGORICAL_DIM
            ].reshape(-1, OPTION_CATEGORICAL_DIM),
            option_numeric=self.arrays["option_numeric"][
                option_start * OPTION_NUMERIC_DIM:
                option_end * OPTION_NUMERIC_DIM
            ].reshape(-1, OPTION_NUMERIC_DIM),
            pokemon_dynamic=self.arrays["pokemon_dynamic"][
                option_start * POKEMON_DYNAMIC_DIM:
                option_end * POKEMON_DYNAMIC_DIM
            ].reshape(-1, POKEMON_DYNAMIC_DIM),
            attack_dynamic=self.arrays["attack_dynamic"][
                option_start * ATTACK_DYNAMIC_DIM:
                option_end * ATTACK_DYNAMIC_DIM
            ].reshape(-1, ATTACK_DYNAMIC_DIM),
            action_option_index=self.arrays["action_option_index"][
                action_start:action_end
            ],
            action_option_offset=self.arrays["action_option_offset"][
                offset_start:offset_end
            ],
            target=int(self.arrays["target"][local_id]),
            action_count=int(self.arrays["action_count"][local_id]),
            episode_key=int(self.arrays["episode_key"][local_id]),
            deck_key=int(self.arrays["deck_key"][local_id]),
            player_result=int(self.arrays["player_result"][local_id]),
            history_select_type=self.arrays["history_select_type"][
                history_fixed_start:history_fixed_start + HISTORY_STEPS
            ],
            history_select_context=self.arrays["history_select_context"][
                history_fixed_start:history_fixed_start + HISTORY_STEPS
            ],
            history_valid=self.arrays["history_valid"][
                history_fixed_start:history_fixed_start + HISTORY_STEPS
            ],
            history_option_categorical=self.arrays[
                "history_option_categorical"
            ][
                history_start * OPTION_CATEGORICAL_DIM:
                history_end * OPTION_CATEGORICAL_DIM
            ].reshape(-1, OPTION_CATEGORICAL_DIM),
            history_structural=self.arrays["history_structural"][
                history_start * HISTORY_STRUCTURAL_DIM:
                history_end * HISTORY_STRUCTURAL_DIM
            ].reshape(-1, HISTORY_STRUCTURAL_DIM),
            history_pokemon_dynamic=self.arrays[
                "history_pokemon_dynamic"
            ][
                history_start * POKEMON_DYNAMIC_DIM:
                history_end * POKEMON_DYNAMIC_DIM
            ].reshape(-1, POKEMON_DYNAMIC_DIM),
            history_attack_dynamic=self.arrays[
                "history_attack_dynamic"
            ][
                history_start * ATTACK_DYNAMIC_DIM:
                history_end * ATTACK_DYNAMIC_DIM
            ].reshape(-1, ATTACK_DYNAMIC_DIM),
            history_option_offset=(
                history_boundaries.astype(np.int64) - history_start
            ),
        )

    def close(self) -> None:
        buffer = getattr(self, "_buffer", None)
        if buffer is not None:
            self.arrays.clear()
            mmap = getattr(buffer, "_mmap", None)
            if mmap is not None:
                try:
                    mmap.close()
                except BufferError:
                    # A short-lived FeatureView may still own an ndarray slice.
                    # Its base keeps the mapping alive until that view is freed.
                    pass
            self._buffer = None


class MmapFeatureDataset:
    """Collection of packed shards with bounded-memory epoch permutations."""

    def __init__(self, root: Path, expected_signature: dict | None = None):
        self.root = Path(root)
        index_path = self.root / "cache.meta.json"
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            names = index.get("part_names")
            if not isinstance(names, list):
                raise ValueError(
                    f"cache index has no part_names list: {index_path}"
                )
            paths = [self.root / str(name) for name in names]
        else:
            paths = sorted(
                path for path in self.root.rglob("*.cache") if path.is_dir()
            )
        if not paths:
            raise FileNotFoundError(f"no *.cache shards found under {self.root}")
        self.shards = []
        try:
            for path in paths:
                self.shards.append(
                    PackedShard(path, expected_signature=expected_signature)
                )
        except Exception:
            self.close()
            raise
        counts = np.asarray([len(shard) for shard in self.shards], dtype=np.int64)
        self.starts = np.empty(len(self.shards), dtype=np.int64)
        self.starts[0] = 0
        if len(self.shards) > 1:
            np.cumsum(counts[:-1], out=self.starts[1:])
        self.ends = self.starts + counts
        self.total_samples = int(counts.sum())
        self.shard_dates = [
            parse_source_date(shard.metadata.get("source", {}).get("name", ""))
            for shard in self.shards
        ]

    def __len__(self) -> int:
        return self.total_samples

    def dates_for_indices(self, indices: np.ndarray) -> np.ndarray:
        """Return month/day pairs aligned with arbitrary global sample IDs."""
        indices = np.asarray(indices)
        if indices.ndim != 1:
            raise ValueError("sample indices must be one-dimensional")
        if np.any(indices < 0) or np.any(indices >= self.total_samples):
            raise IndexError("sample index is outside the dataset")
        shard_ids = np.searchsorted(
            self.ends, indices.astype(np.int64, copy=False), side="right"
        )
        return np.asarray(self.shard_dates, dtype=np.int16)[shard_ids]

    def iter_index_batches(
        self,
        indices: np.ndarray,
        batch_size: int,
        seed: int,
        shuffle: bool,
        max_samples: int | None = None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        order = np.asarray(indices)
        if order.ndim != 1:
            raise ValueError("indices must be one-dimensional")
        limit = order.size
        if max_samples is not None:
            limit = min(limit, int(max_samples))
        if shuffle:
            order = order.copy()
            np.random.default_rng(seed).shuffle(order)
        for start in range(0, limit, batch_size):
            yield IndexBatch(order[start : min(start + batch_size, limit)])

    def build_splits(
        self,
        validation_ratio: float,
        validation_seed: int,
        expert_episode_keys: dict[tuple[int, int], set[int] | frozenset[int]]
        | None = None,
        top_deck_keys: Iterable[int] | None = None,
        train_replay_ratio: float = 1.0,
        train_replay_seed: int = 0,
        isolation_episode_keys: Mapping[
            str,
            Mapping[tuple[int, int], AbstractSet[int]],
        ]
        | None = None,
        loser_episode_keys: Mapping[
            tuple[int, int], AbstractSet[int]
        ]
        | None = None,
    ) -> DatasetSplits:
        if not 0 < validation_ratio < 1:
            raise ValueError("validation_ratio must be strictly between 0 and 1")
        if not 0 < train_replay_ratio <= 1:
            raise ValueError("train_replay_ratio must be in (0, 1]")
        latest_date = max(self.shard_dates)
        dtype = (
            np.uint32
            if self.total_samples <= np.iinfo(np.uint32).max
            else np.uint64
        )
        threshold = int(validation_ratio * (1 << 32))
        train_threshold = int(train_replay_ratio * (1 << 32))
        ordered_top_deck_keys = (
            None
            if top_deck_keys is None
            else tuple(int(key) for key in top_deck_keys)
        )
        train_parts = []
        isolation_parts = []
        eligible_train_samples = 0
        eligible_train_key_parts = []
        selected_train_key_parts = []
        loser_count_parts: dict[
            tuple[int, int],
            list[tuple[np.ndarray, np.ndarray, np.ndarray, int]],
        ] = {
            date: [] for date in (loser_episode_keys or {})
        }
        in_distribution_parts = []
        in_distribution_expert_parts = []
        in_distribution_top_deck_parts = []
        in_distribution_per_deck_parts = [
            [] for _ in ordered_top_deck_keys or ()
        ]
        latest_parts = []
        latest_expert_parts = []
        latest_top_deck_parts = []
        latest_per_deck_parts = [
            [] for _ in ordered_top_deck_keys or ()
        ]
        isolation_names = sorted(
            map(str, (isolation_episode_keys or {}).keys())
        )
        isolation_mask_parts: dict[str, list[np.ndarray]] = {
            name: [] for name in isolation_names
        }
        isolation_union_replay_ids: set[tuple[tuple[int, int], int]] = set()
        required_dates = set(self.shard_dates)
        unknown_loser_dates = set(loser_count_parts) - required_dates
        if unknown_loser_dates:
            labels = ", ".join(
                f"{month}.{day}" for month, day in sorted(unknown_loser_dates)
            )
            raise ValueError(
                f"loser augmentation dates are absent from cache: {labels}"
            )
        if isolation_episode_keys is not None:
            if not isolation_names:
                raise ValueError(
                    "isolation_episode_keys must contain at least one namespace"
                )
            for name in isolation_names:
                date_sets = isolation_episode_keys[name]
                missing_dates = required_dates - set(date_sets)
                if missing_dates:
                    labels = ", ".join(
                        f"{month}.{day}"
                        for month, day in sorted(missing_dates)
                    )
                    raise ValueError(
                        f"{name} isolation episode keys missing dates: {labels}"
                    )
                replay_ids = {
                    (date, int(key))
                    for date in required_dates
                    for key in date_sets[date]
                }
                if not replay_ids:
                    raise ValueError(
                        f"{name} isolation validation has no replay keys"
                    )
                isolation_union_replay_ids.update(replay_ids)
        if expert_episode_keys is not None:
            missing = set(self.shard_dates) - set(expert_episode_keys)
            if missing:
                labels = ", ".join(
                    f"{month}.{day}" for month, day in sorted(missing)
                )
                raise ValueError(f"expert episode keys missing for dates: {labels}")
        for shard_id, shard in enumerate(self.shards):
            global_ids = np.arange(
                self.starts[shard_id],
                self.ends[shard_id],
                dtype=dtype,
            )
            date = self.shard_dates[shard_id]
            player_results = shard.arrays["player_result"]
            winner_mask = player_results == PLAYER_RESULT_WIN
            loss_mask = player_results == PLAYER_RESULT_LOSS
            if expert_episode_keys is None:
                expert_mask = np.zeros(len(shard), dtype=np.bool_)
            else:
                keys = np.fromiter(
                    expert_episode_keys[date], dtype=np.uint32
                )
                expert_mask = np.isin(
                    shard.arrays["episode_key"], keys, assume_unique=False
                )
            if ordered_top_deck_keys is None:
                per_deck_masks: tuple[np.ndarray, ...] = ()
                top_deck_mask = np.zeros(len(shard), dtype=np.bool_)
            else:
                per_deck_masks = tuple(
                    shard.arrays["deck_key"] == deck_key
                    for deck_key in ordered_top_deck_keys
                )
                top_deck_mask = np.logical_or.reduce(per_deck_masks)
            namespace_masks: dict[str, np.ndarray] = {}
            for name in isolation_names:
                keys = np.fromiter(
                    isolation_episode_keys[name][date],
                    dtype=np.uint32,
                )
                namespace_masks[name] = np.isin(
                    shard.arrays["episode_key"],
                    keys,
                    assume_unique=False,
                )
            if namespace_masks:
                isolation_replay_mask = np.logical_or.reduce(
                    tuple(namespace_masks.values())
                )
                isolation_validation_mask = (
                    isolation_replay_mask & winner_mask
                )
                isolation_parts.append(global_ids[isolation_validation_mask])
                for name, mask in namespace_masks.items():
                    isolation_mask_parts[name].append(
                        mask[isolation_validation_mask]
                    )
            else:
                isolation_replay_mask = np.zeros(len(shard), dtype=np.bool_)
            if date == latest_date:
                latest_mask = ~isolation_replay_mask & winner_mask
                if np.any(isolation_replay_mask & latest_mask):
                    raise RuntimeError(
                        "isolation validation overlaps latest-date samples"
                    )
                latest_parts.append(global_ids[latest_mask])
                latest_expert_parts.append(expert_mask[latest_mask])
                latest_top_deck_parts.append(top_deck_mask[latest_mask])
                for parts, deck_mask in zip(
                    latest_per_deck_parts, per_deck_masks
                ):
                    parts.append(deck_mask[latest_mask])
                continue
            mixed = _mix_episode_keys(
                shard.arrays["episode_key"], validation_seed
            )
            validation_replay_mask = (
                ~isolation_replay_mask
                & (mixed.astype(np.uint64) < threshold)
            )
            validation_mask = validation_replay_mask & winner_mask
            in_distribution_parts.append(global_ids[validation_mask])
            in_distribution_expert_parts.append(
                expert_mask[validation_mask]
            )
            in_distribution_top_deck_parts.append(
                top_deck_mask[validation_mask]
            )
            for parts, deck_mask in zip(
                in_distribution_per_deck_parts, per_deck_masks
            ):
                parts.append(deck_mask[validation_mask])
            eligible_replay_mask = (
                ~isolation_replay_mask & ~validation_replay_mask
            )
            train_selection = (
                _mix_episode_keys(
                    shard.arrays["episode_key"],
                    train_replay_seed ^ 0x9E3779B9,
                ).astype(np.uint64)
                < train_threshold
            )
            if date in loser_count_parts:
                loser_keys = np.fromiter(
                    loser_episode_keys[date], dtype=np.uint32
                )
                score_eligible_loss_mask = loss_mask & np.isin(
                    shard.arrays["episode_key"],
                    loser_keys,
                    assume_unique=False,
                )
            else:
                score_eligible_loss_mask = np.zeros(
                    len(shard), dtype=np.bool_
                )
            train_eligible_sample_mask = (
                winner_mask | score_eligible_loss_mask
            )
            selected_mask = (
                eligible_replay_mask
                & train_selection
                & train_eligible_sample_mask
            )
            if np.any(
                isolation_replay_mask & (validation_mask | selected_mask)
            ):
                raise RuntimeError(
                    "isolation validation overlaps later split samples"
                )
            train_parts.append(global_ids[selected_mask])
            eligible_sample_mask = (
                eligible_replay_mask & train_eligible_sample_mask
            )
            eligible_train_samples += int(eligible_sample_mask.sum())
            eligible_train_key_parts.append(
                np.unique(shard.arrays["episode_key"][eligible_sample_mask])
            )
            selected_train_key_parts.append(
                np.unique(shard.arrays["episode_key"][selected_mask])
            )
            if date in loser_count_parts:
                after_validation_mask = (
                    eligible_replay_mask & score_eligible_loss_mask
                )
                selected_loser_mask = after_validation_mask & train_selection
                loser_count_parts[date].append(
                    (
                        np.unique(
                            shard.arrays["episode_key"][
                                score_eligible_loss_mask
                            ]
                        ),
                        np.unique(
                            shard.arrays["episode_key"][after_validation_mask]
                        ),
                        np.unique(
                            shard.arrays["episode_key"][selected_loser_mask]
                        ),
                        int(selected_loser_mask.sum()),
                    )
                )

        def combine(parts: list[np.ndarray], name: str) -> np.ndarray:
            nonempty = [part for part in parts if part.size]
            if not nonempty:
                raise ValueError(f"{name} split is empty")
            return np.concatenate(nonempty).astype(dtype, copy=False)

        in_distribution = combine(
            in_distribution_parts, "in-distribution validation"
        )
        latest = combine(latest_parts, "latest-date validation")
        if isolation_names:
            isolation = combine(
                isolation_parts,
                "isolation validation union",
            )
            isolation_masks = {
                name: np.concatenate(isolation_mask_parts[name]).astype(
                    np.bool_,
                    copy=False,
                )
                for name in isolation_names
            }
            empty_isolation = [
                name
                for name, mask in isolation_masks.items()
                if not np.any(mask)
            ]
            if empty_isolation:
                raise ValueError(
                    "isolation validation subsets are empty in cache: "
                    f"{empty_isolation}"
                )
        else:
            isolation = np.empty(0, dtype=dtype)
            isolation_masks = {}
        in_distribution_expert_mask = np.concatenate(
            in_distribution_expert_parts
        ).astype(np.bool_, copy=False)
        latest_expert_mask = np.concatenate(latest_expert_parts).astype(
            np.bool_, copy=False
        )
        in_distribution_top_deck_mask = np.concatenate(
            in_distribution_top_deck_parts
        ).astype(np.bool_, copy=False)
        latest_top_deck_mask = np.concatenate(latest_top_deck_parts).astype(
            np.bool_, copy=False
        )
        latest_expert_top_deck_mask = (
            latest_expert_mask & latest_top_deck_mask
        )
        in_distribution_expert_top_deck_mask = (
            in_distribution_expert_mask & in_distribution_top_deck_mask
        )
        in_distribution_top_deck_masks = tuple(
            np.concatenate(parts).astype(np.bool_, copy=False)
            for parts in in_distribution_per_deck_parts
        )
        latest_top_deck_masks = tuple(
            np.concatenate(parts).astype(np.bool_, copy=False)
            for parts in latest_per_deck_parts
        )
        in_distribution_expert_top_deck_masks = tuple(
            in_distribution_expert_mask & mask
            for mask in in_distribution_top_deck_masks
        )
        latest_expert_top_deck_masks = tuple(
            latest_expert_mask & mask for mask in latest_top_deck_masks
        )
        if expert_episode_keys is not None:
            if not np.any(in_distribution_expert_mask):
                raise ValueError("in-distribution expert validation is empty")
            if not np.any(latest_expert_mask):
                raise ValueError("latest-date expert validation is empty")
        if ordered_top_deck_keys is not None:
            if not np.any(in_distribution_top_deck_mask):
                raise ValueError("in-distribution top-deck validation is empty")
            if not np.any(latest_top_deck_mask):
                raise ValueError("latest-date top-deck validation is empty")
            if not np.any(latest_expert_top_deck_mask):
                raise ValueError(
                    "latest-date expert top-deck validation is empty"
                )
            named_masks = {
                **{
                    f"in-distribution deck{index}": mask
                    for index, mask in enumerate(
                        in_distribution_top_deck_masks, start=1
                    )
                },
                **{
                    f"in-distribution expert deck{index}": mask
                    for index, mask in enumerate(
                        in_distribution_expert_top_deck_masks, start=1
                    )
                },
                **{
                    f"latest-date deck{index}": mask
                    for index, mask in enumerate(
                        latest_top_deck_masks, start=1
                    )
                },
                **{
                    f"latest-date expert deck{index}": mask
                    for index, mask in enumerate(
                        latest_expert_top_deck_masks, start=1
                    )
                },
            }
            empty_masks = [
                name for name, mask in named_masks.items() if not np.any(mask)
            ]
            if empty_masks:
                raise ValueError(
                    "configured top-deck validation subsets are empty: "
                    f"{empty_masks}"
                )
        train = combine(train_parts, "train")

        def unique_count(parts: list[np.ndarray]) -> int:
            nonempty = [part for part in parts if part.size]
            if not nonempty:
                return 0
            return int(np.unique(np.concatenate(nonempty)).size)

        loser_augmentation_counts = {}
        for date, parts in loser_count_parts.items():
            score_parts = [score for score, _, _, _ in parts]
            after_parts = [after for _, after, _, _ in parts]
            selected_parts = [selected for _, _, selected, _ in parts]
            loser_augmentation_counts[date] = LoserAugmentationCounts(
                score_eligible_episodes=unique_count(score_parts),
                after_validation_episodes=unique_count(after_parts),
                selected_train_episodes=unique_count(selected_parts),
                loser_samples=sum(samples for _, _, _, samples in parts),
            )
        loser_augmentation_replays = sum(
            counts.selected_train_episodes
            for counts in loser_augmentation_counts.values()
        )
        loser_augmentation_samples = sum(
            counts.loser_samples
            for counts in loser_augmentation_counts.values()
        )

        return DatasetSplits(
            train=train,
            isolation=isolation,
            isolation_masks=isolation_masks,
            isolation_sample_counts={
                name: int(mask.sum())
                for name, mask in isolation_masks.items()
            },
            isolation_union_replays=len(isolation_union_replay_ids),
            in_distribution=in_distribution,
            in_distribution_expert_mask=in_distribution_expert_mask,
            in_distribution_top_deck_mask=in_distribution_top_deck_mask,
            in_distribution_expert_top_deck_mask=(
                in_distribution_expert_top_deck_mask
            ),
            in_distribution_top_deck_masks=in_distribution_top_deck_masks,
            in_distribution_expert_top_deck_masks=(
                in_distribution_expert_top_deck_masks
            ),
            latest=latest,
            latest_expert_mask=latest_expert_mask,
            latest_top_deck_mask=latest_top_deck_mask,
            latest_expert_top_deck_mask=latest_expert_top_deck_mask,
            latest_top_deck_masks=latest_top_deck_masks,
            latest_expert_top_deck_masks=latest_expert_top_deck_masks,
            latest_date=latest_date,
            eligible_train_samples=eligible_train_samples,
            eligible_train_replays=unique_count(eligible_train_key_parts),
            selected_train_replays=unique_count(selected_train_key_parts),
            loser_augmentation_counts=loser_augmentation_counts,
            loser_augmentation_replays=loser_augmentation_replays,
            loser_augmentation_samples=loser_augmentation_samples,
            loser_fraction_in_train=(
                loser_augmentation_samples / len(train)
                if len(train)
                else 0.0
            ),
        )

    def collate(self, index_batch: IndexBatch) -> CachedBatch:
        global_ids = np.asarray(index_batch.global_ids, dtype=np.int64)
        shard_ids = np.searchsorted(self.ends, global_ids, side="right")
        encoder_indices = []
        encoder_values = []
        option_categorical = []
        option_numeric = []
        pokemon_dynamic = []
        attack_dynamic = []
        history_option_categorical = []
        history_structural = []
        history_pokemon_dynamic = []
        history_attack_dynamic = []
        action_option_indices = []
        encoder_offsets = np.empty(global_ids.size * ENCODER_WORDS, dtype=np.int32)
        encoder_pokemon_appear = np.empty(
            (global_ids.size, POKEMON_ENCODER_TOKENS),
            dtype=np.uint8,
        )
        own_summaries = np.empty(
            (global_ids.size, OWN_SUMMARY_DIM), dtype=np.float16
        )
        opponent_summaries = np.empty(
            (global_ids.size, OPPONENT_SUMMARY_DIM), dtype=np.float16
        )
        global_summaries = np.empty(
            (global_ids.size, GLOBAL_SUMMARY_DIM), dtype=np.float16
        )
        history_select_type = np.empty(
            (global_ids.size, HISTORY_STEPS), dtype=np.uint8
        )
        history_select_context = np.empty(
            (global_ids.size, HISTORY_STEPS), dtype=np.uint8
        )
        history_valid = np.empty(
            (global_ids.size, HISTORY_STEPS), dtype=np.uint8
        )
        history_option_offsets = np.empty(
            global_ids.size * HISTORY_STEPS + 1, dtype=np.int32
        )
        history_option_offsets[0] = 0
        action_option_offsets = np.empty(
            global_ids.size * MAX_ACTIONS + 1, dtype=np.int32
        )
        targets = np.empty(global_ids.size, dtype=np.int64)
        action_counts = np.empty(global_ids.size, dtype=np.int64)
        encoder_base = 0
        option_base = 0
        action_option_base = 0
        history_option_base = 0

        for row, (global_id, shard_id) in enumerate(zip(global_ids, shard_ids)):
            local_id = int(global_id - self.starts[int(shard_id)])
            sample = self.shards[int(shard_id)].sample(local_id)
            encoder_indices.append(sample.encoder_index)
            encoder_values.append(sample.encoder_value)
            encoder_pokemon_appear[row] = sample.encoder_pokemon_appear
            option_categorical.append(sample.option_categorical)
            option_numeric.append(sample.option_numeric)
            pokemon_dynamic.append(sample.pokemon_dynamic)
            attack_dynamic.append(sample.attack_dynamic)
            history_option_categorical.append(
                sample.history_option_categorical
            )
            history_structural.append(sample.history_structural)
            history_pokemon_dynamic.append(sample.history_pokemon_dynamic)
            history_attack_dynamic.append(sample.history_attack_dynamic)
            history_select_type[row] = sample.history_select_type
            history_select_context[row] = sample.history_select_context
            history_valid[row] = sample.history_valid
            action_option_indices.append(
                sample.action_option_index.astype(np.int32) + option_base
            )
            own_summaries[row] = sample.own_summary
            opponent_summaries[row] = sample.opponent_summary
            global_summaries[row] = sample.global_summary

            enc_slice = slice(row * ENCODER_WORDS, (row + 1) * ENCODER_WORDS)
            encoder_offsets[enc_slice] = (
                sample.encoder_offset.astype(np.int32) + encoder_base
            )
            action_start = row * MAX_ACTIONS
            action_end = action_start + sample.action_count + 1
            action_option_offsets[action_start:action_end] = (
                sample.action_option_offset.astype(np.int32)
                + action_option_base
            )
            action_option_offsets[
                action_end:(row + 1) * MAX_ACTIONS + 1
            ] = (
                action_option_base + sample.action_option_index.size
            )
            history_offset_start = row * HISTORY_STEPS
            history_option_offsets[
                history_offset_start + 1:
                history_offset_start + HISTORY_STEPS + 1
            ] = (
                sample.history_option_offset[1:].astype(np.int32)
                + history_option_base
            )
            targets[row] = sample.target
            action_counts[row] = sample.action_count
            encoder_base += int(sample.encoder_index.size)
            option_base += int(sample.option_categorical.shape[0])
            action_option_base += int(sample.action_option_index.size)
            history_option_base += int(
                sample.history_option_categorical.shape[0]
            )

        return CachedBatch(
            encoder_index=np.concatenate(encoder_indices).astype(np.int32, copy=False),
            encoder_value=np.concatenate(encoder_values).astype(np.float16, copy=False),
            encoder_offset=encoder_offsets,
            encoder_pokemon_appear=encoder_pokemon_appear,
            own_summary=own_summaries,
            opponent_summary=opponent_summaries,
            global_summary=global_summaries,
            option_categorical=np.concatenate(option_categorical).astype(
                np.int64, copy=False
            ),
            option_numeric=np.concatenate(option_numeric).astype(
                np.float16, copy=False
            ),
            pokemon_dynamic=np.concatenate(pokemon_dynamic).astype(
                np.float16, copy=False
            ),
            attack_dynamic=np.concatenate(attack_dynamic).astype(
                np.float16, copy=False
            ),
            action_option_index=np.concatenate(action_option_indices).astype(
                np.int64, copy=False
            ),
            action_option_offset=action_option_offsets,
            target=targets,
            action_count=action_counts,
            history_select_type=history_select_type,
            history_select_context=history_select_context,
            history_valid=history_valid,
            history_option_categorical=np.concatenate(
                history_option_categorical
            ).astype(np.int64, copy=False),
            history_structural=np.concatenate(history_structural).astype(
                np.int64, copy=False
            ),
            history_pokemon_dynamic=np.concatenate(
                history_pokemon_dynamic
            ).astype(np.float16, copy=False),
            history_attack_dynamic=np.concatenate(
                history_attack_dynamic
            ).astype(np.float16, copy=False),
            history_option_offset=history_option_offsets,
        )

    def iter_batches(
        self,
        indices: np.ndarray,
        batch_size: int,
        seed: int,
        shuffle: bool,
        max_samples: int | None = None,
    ):
        for index_batch in self.iter_index_batches(
            indices=indices,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            max_samples=max_samples,
        ):
            yield self.collate(index_batch)

    def close(self) -> None:
        for shard in self.shards:
            shard.close()
