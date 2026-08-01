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


CACHE_SCHEMA_VERSION = 6
ENCODER_WORDS = 20
OWN_SUMMARY_DIM = 60
OPPONENT_SUMMARY_DIM = 62
GLOBAL_SUMMARY_DIM = 73
OPTION_CATEGORICAL_DIM = 5
OPTION_NUMERIC_DIM = 16
MAX_ACTIONS = 64
ALIGNMENT = 64

SECTION_DTYPES = {
    "encoder_index": np.dtype("<u2"),
    "encoder_value": np.dtype("<f2"),
    "encoder_ptr": np.dtype("<u4"),
    "encoder_offset": np.dtype("<u2"),
    "own_summary": np.dtype("<f2"),
    "opponent_summary": np.dtype("<f2"),
    "global_summary": np.dtype("<f2"),
    "option_categorical": np.dtype("<u2"),
    "option_numeric": np.dtype("<f2"),
    "option_ptr": np.dtype("<u4"),
    "action_option_index": np.dtype("<u2"),
    "action_option_ptr": np.dtype("<u4"),
    "action_option_offset": np.dtype("<u2"),
    "action_option_offset_ptr": np.dtype("<u4"),
    "target": np.dtype("u1"),
    "action_count": np.dtype("u1"),
    "episode_key": np.dtype("<u4"),
    "deck_key": np.dtype("<u8"),
}


@dataclass(slots=True)
class FeatureRecord:
    encoder_index: list[int]
    encoder_value: list[float]
    encoder_offset: list[int]
    own_summary: list[float]
    opponent_summary: list[float]
    global_summary: list[float]
    option_categorical: list[int]
    option_numeric: list[float]
    action_option_index: list[int]
    action_option_offset: list[int]
    target: int
    action_count: int
    episode_key: int
    deck_key: int


@dataclass(frozen=True, slots=True)
class FeatureView:
    encoder_index: np.ndarray
    encoder_value: np.ndarray
    encoder_offset: np.ndarray
    own_summary: np.ndarray
    opponent_summary: np.ndarray
    global_summary: np.ndarray
    option_categorical: np.ndarray
    option_numeric: np.ndarray
    action_option_index: np.ndarray
    action_option_offset: np.ndarray
    target: int
    action_count: int
    episode_key: int
    deck_key: int


@dataclass(frozen=True, slots=True)
class IndexBatch:
    global_ids: np.ndarray


@dataclass(frozen=True, slots=True)
class CachedBatch:
    encoder_index: np.ndarray
    encoder_value: np.ndarray
    encoder_offset: np.ndarray
    own_summary: np.ndarray
    opponent_summary: np.ndarray
    global_summary: np.ndarray
    option_categorical: np.ndarray
    option_numeric: np.ndarray
    action_option_index: np.ndarray
    action_option_offset: np.ndarray
    target: np.ndarray
    action_count: np.ndarray

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
    latest: np.ndarray
    latest_expert_mask: np.ndarray
    latest_top_deck_mask: np.ndarray
    latest_expert_top_deck_mask: np.ndarray
    latest_date: tuple[int, int]
    eligible_train_samples: int
    eligible_train_replays: int
    selected_train_replays: int


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
        self._action_option_nnz = 0
        self._action_offset_count = 0
        self._closed = False

        self._append("encoder_ptr", 0)
        self._append("option_ptr", 0)
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
        if len(record.option_numeric) != option_count * OPTION_NUMERIC_DIM:
            raise ValueError("option_numeric does not align with options")
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

        _validate_unsigned("encoder_index", record.encoder_index, np.iinfo(np.uint16).max)
        _validate_unsigned("encoder_offset", record.encoder_offset, np.iinfo(np.uint16).max)
        _validate_unsigned(
            "option_categorical",
            record.option_categorical,
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
        option_numeric = np.asarray(record.option_numeric, dtype=np.float32)
        narrowed_numeric = option_numeric.astype(np.float16)
        if not np.all(np.isfinite(option_numeric)) or not np.all(
            np.isfinite(narrowed_numeric)
        ):
            raise ValueError(
                "option_numeric contains a non-finite or float16-overflow value"
            )

        next_encoder_nnz = self._encoder_nnz + len(record.encoder_index)
        next_option_count = self._option_count + option_count
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
            next_action_option_nnz,
            next_action_offset_count,
        ) > uint32_max:
            raise ValueError("cache shard pointer exceeds uint32 range")

        self._buffers["encoder_index"].extend(record.encoder_index)
        self._buffers["encoder_value"].extend(record.encoder_value)
        self._buffers["encoder_offset"].extend(record.encoder_offset)
        self._buffers["own_summary"].extend(record.own_summary)
        self._buffers["opponent_summary"].extend(record.opponent_summary)
        self._buffers["global_summary"].extend(record.global_summary)
        self._buffers["option_categorical"].extend(
            record.option_categorical
        )
        self._buffers["option_numeric"].extend(record.option_numeric)
        self._buffers["action_option_index"].extend(
            record.action_option_index
        )
        self._buffers["action_option_offset"].extend(
            record.action_option_offset
        )

        self._encoder_nnz = next_encoder_nnz
        self._option_count = next_option_count
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
        if self.arrays["encoder_offset"].size != self.samples * ENCODER_WORDS:
            raise ValueError("encoder_offset length does not match sample count")
        dense_widths = {
            "own_summary": OWN_SUMMARY_DIM,
            "opponent_summary": OPPONENT_SUMMARY_DIM,
            "global_summary": GLOBAL_SUMMARY_DIM,
        }
        for name, width in dense_widths.items():
            if self.arrays[name].size != self.samples * width:
                raise ValueError(f"{name} length does not match sample count")
        if self.arrays["target"].size != self.samples:
            raise ValueError("target length does not match sample count")
        if self.arrays["action_count"].size != self.samples:
            raise ValueError("action_count length does not match sample count")
        if self.arrays["episode_key"].size != self.samples:
            raise ValueError("episode_key length does not match sample count")
        if self.arrays["deck_key"].size != self.samples:
            raise ValueError("deck_key length does not match sample count")
        option_count = int(self.arrays["option_ptr"][-1])
        if (
            self.arrays["option_categorical"].size
            != option_count * OPTION_CATEGORICAL_DIM
            or self.arrays["option_numeric"].size
            != option_count * OPTION_NUMERIC_DIM
        ):
            raise ValueError("option feature arrays do not align with option_ptr")
        pointer_targets = {
            "encoder_ptr": self.arrays["encoder_index"].size,
            "option_ptr": option_count,
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
        action_start = int(self.arrays["action_option_ptr"][local_id])
        action_end = int(self.arrays["action_option_ptr"][local_id + 1])
        offset_start = int(
            self.arrays["action_option_offset_ptr"][local_id]
        )
        offset_end = int(
            self.arrays["action_option_offset_ptr"][local_id + 1]
        )
        encoder_word_start = local_id * ENCODER_WORDS
        own_start = local_id * OWN_SUMMARY_DIM
        opponent_start = local_id * OPPONENT_SUMMARY_DIM
        global_start = local_id * GLOBAL_SUMMARY_DIM
        return FeatureView(
            encoder_index=self.arrays["encoder_index"][encoder_start:encoder_end],
            encoder_value=self.arrays["encoder_value"][encoder_start:encoder_end],
            encoder_offset=self.arrays["encoder_offset"][
                encoder_word_start : encoder_word_start + ENCODER_WORDS
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
        top_deck_keys: set[int] | frozenset[int] | None = None,
        train_replay_ratio: float = 1.0,
        train_replay_seed: int = 0,
        isolation_episode_keys: Mapping[
            str,
            Mapping[tuple[int, int], AbstractSet[int]],
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
        train_parts = []
        isolation_parts = []
        eligible_train_samples = 0
        eligible_train_key_parts = []
        selected_train_key_parts = []
        in_distribution_parts = []
        in_distribution_expert_parts = []
        in_distribution_top_deck_parts = []
        latest_parts = []
        latest_expert_parts = []
        latest_top_deck_parts = []
        isolation_names = sorted(
            map(str, (isolation_episode_keys or {}).keys())
        )
        isolation_mask_parts: dict[str, list[np.ndarray]] = {
            name: [] for name in isolation_names
        }
        isolation_union_replay_ids: set[tuple[tuple[int, int], int]] = set()
        required_dates = set(self.shard_dates)
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
            if expert_episode_keys is None:
                expert_mask = np.zeros(len(shard), dtype=np.bool_)
            else:
                keys = np.fromiter(
                    expert_episode_keys[date], dtype=np.uint32
                )
                expert_mask = np.isin(
                    shard.arrays["episode_key"], keys, assume_unique=False
                )
            if top_deck_keys is None:
                top_deck_mask = np.zeros(len(shard), dtype=np.bool_)
            else:
                deck_keys = np.fromiter(top_deck_keys, dtype=np.uint64)
                top_deck_mask = np.isin(
                    shard.arrays["deck_key"],
                    deck_keys,
                    assume_unique=False,
                )
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
                isolation_mask = np.logical_or.reduce(
                    tuple(namespace_masks.values())
                )
                isolation_parts.append(global_ids[isolation_mask])
                for name, mask in namespace_masks.items():
                    isolation_mask_parts[name].append(mask[isolation_mask])
            else:
                isolation_mask = np.zeros(len(shard), dtype=np.bool_)
            if date == latest_date:
                latest_mask = ~isolation_mask
                if np.any(isolation_mask & latest_mask):
                    raise RuntimeError(
                        "isolation validation overlaps latest-date samples"
                    )
                latest_parts.append(global_ids[latest_mask])
                latest_expert_parts.append(expert_mask[latest_mask])
                latest_top_deck_parts.append(top_deck_mask[latest_mask])
                continue
            mixed = _mix_episode_keys(
                shard.arrays["episode_key"], validation_seed
            )
            validation_mask = (
                ~isolation_mask
                & (mixed.astype(np.uint64) < threshold)
            )
            in_distribution_parts.append(global_ids[validation_mask])
            in_distribution_expert_parts.append(
                expert_mask[validation_mask]
            )
            in_distribution_top_deck_parts.append(
                top_deck_mask[validation_mask]
            )
            eligible_mask = ~isolation_mask & ~validation_mask
            train_selection = (
                _mix_episode_keys(
                    shard.arrays["episode_key"],
                    train_replay_seed ^ 0x9E3779B9,
                ).astype(np.uint64)
                < train_threshold
            )
            selected_mask = eligible_mask & train_selection
            if np.any(
                isolation_mask & (validation_mask | selected_mask)
            ):
                raise RuntimeError(
                    "isolation validation overlaps later split samples"
                )
            train_parts.append(global_ids[selected_mask])
            eligible_train_samples += int(eligible_mask.sum())
            eligible_train_key_parts.append(
                np.unique(shard.arrays["episode_key"][eligible_mask])
            )
            selected_train_key_parts.append(
                np.unique(shard.arrays["episode_key"][selected_mask])
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
        if expert_episode_keys is not None:
            if not np.any(in_distribution_expert_mask):
                raise ValueError("in-distribution expert validation is empty")
            if not np.any(latest_expert_mask):
                raise ValueError("latest-date expert validation is empty")
        if top_deck_keys is not None:
            if not np.any(in_distribution_top_deck_mask):
                raise ValueError("in-distribution top-deck validation is empty")
            if not np.any(latest_top_deck_mask):
                raise ValueError("latest-date top-deck validation is empty")
            if not np.any(latest_expert_top_deck_mask):
                raise ValueError(
                    "latest-date expert top-deck validation is empty"
                )
        train = combine(train_parts, "train")

        def unique_count(parts: list[np.ndarray]) -> int:
            nonempty = [part for part in parts if part.size]
            if not nonempty:
                return 0
            return int(np.unique(np.concatenate(nonempty)).size)

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
            latest=latest,
            latest_expert_mask=latest_expert_mask,
            latest_top_deck_mask=latest_top_deck_mask,
            latest_expert_top_deck_mask=latest_expert_top_deck_mask,
            latest_date=latest_date,
            eligible_train_samples=eligible_train_samples,
            eligible_train_replays=unique_count(eligible_train_key_parts),
            selected_train_replays=unique_count(selected_train_key_parts),
        )

    def collate(self, index_batch: IndexBatch) -> CachedBatch:
        global_ids = np.asarray(index_batch.global_ids, dtype=np.int64)
        shard_ids = np.searchsorted(self.ends, global_ids, side="right")
        encoder_indices = []
        encoder_values = []
        option_categorical = []
        option_numeric = []
        action_option_indices = []
        encoder_offsets = np.empty(global_ids.size * ENCODER_WORDS, dtype=np.int32)
        own_summaries = np.empty(
            (global_ids.size, OWN_SUMMARY_DIM), dtype=np.float16
        )
        opponent_summaries = np.empty(
            (global_ids.size, OPPONENT_SUMMARY_DIM), dtype=np.float16
        )
        global_summaries = np.empty(
            (global_ids.size, GLOBAL_SUMMARY_DIM), dtype=np.float16
        )
        action_option_offsets = np.empty(
            global_ids.size * MAX_ACTIONS + 1, dtype=np.int32
        )
        targets = np.empty(global_ids.size, dtype=np.int64)
        action_counts = np.empty(global_ids.size, dtype=np.int64)
        encoder_base = 0
        option_base = 0
        action_option_base = 0

        for row, (global_id, shard_id) in enumerate(zip(global_ids, shard_ids)):
            local_id = int(global_id - self.starts[int(shard_id)])
            sample = self.shards[int(shard_id)].sample(local_id)
            encoder_indices.append(sample.encoder_index)
            encoder_values.append(sample.encoder_value)
            option_categorical.append(sample.option_categorical)
            option_numeric.append(sample.option_numeric)
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
            targets[row] = sample.target
            action_counts[row] = sample.action_count
            encoder_base += int(sample.encoder_index.size)
            option_base += int(sample.option_categorical.shape[0])
            action_option_base += int(sample.action_option_index.size)

        return CachedBatch(
            encoder_index=np.concatenate(encoder_indices).astype(np.int32, copy=False),
            encoder_value=np.concatenate(encoder_values).astype(np.float16, copy=False),
            encoder_offset=encoder_offsets,
            own_summary=own_summaries,
            opponent_summary=opponent_summaries,
            global_summary=global_summaries,
            option_categorical=np.concatenate(option_categorical).astype(
                np.int64, copy=False
            ),
            option_numeric=np.concatenate(option_numeric).astype(
                np.float16, copy=False
            ),
            action_option_index=np.concatenate(action_option_indices).astype(
                np.int64, copy=False
            ),
            action_option_offset=action_option_offsets,
            target=targets,
            action_count=action_counts,
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
