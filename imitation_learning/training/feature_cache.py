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
from typing import Iterable

import numpy as np


CACHE_SCHEMA_VERSION = 3
ENCODER_WORDS = 24
MAX_ACTIONS = 64
ALIGNMENT = 64

SECTION_DTYPES = {
    "encoder_index": np.dtype("<u2"),
    "encoder_value": np.dtype("<f2"),
    "encoder_ptr": np.dtype("<u4"),
    "encoder_offset": np.dtype("<u2"),
    "decoder_index": np.dtype("<u4"),
    "decoder_ptr": np.dtype("<u4"),
    "decoder_offset": np.dtype("<u2"),
    "decoder_offset_ptr": np.dtype("<u4"),
    "target": np.dtype("u1"),
    "action_count": np.dtype("u1"),
    "episode_key": np.dtype("<u4"),
}


@dataclass(slots=True)
class FeatureRecord:
    encoder_index: list[int]
    encoder_value: list[float]
    encoder_offset: list[int]
    decoder_index: list[int]
    decoder_offset: list[int]
    target: int
    action_count: int
    episode_key: int


@dataclass(frozen=True, slots=True)
class FeatureView:
    encoder_index: np.ndarray
    encoder_value: np.ndarray
    encoder_offset: np.ndarray
    decoder_index: np.ndarray
    decoder_offset: np.ndarray
    target: int
    action_count: int
    episode_key: int


@dataclass(frozen=True, slots=True)
class IndexBatch:
    global_ids: np.ndarray


@dataclass(frozen=True, slots=True)
class CachedBatch:
    encoder_index: np.ndarray
    encoder_value: np.ndarray
    encoder_offset: np.ndarray
    decoder_index: np.ndarray
    decoder_offset: np.ndarray
    target: np.ndarray
    action_count: np.ndarray

    def __len__(self) -> int:
        return int(self.target.size)


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    train: np.ndarray
    in_distribution: np.ndarray
    latest: np.ndarray
    latest_date: tuple[int, int]


def stable_episode_key(episode_id: object) -> int:
    payload = str(episode_id).encode("utf-8")
    return int.from_bytes(
        hashlib.blake2s(payload, digest_size=4).digest(), "little"
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
        self._decoder_nnz = 0
        self._decoder_words = 0
        self._closed = False

        self._append("encoder_ptr", 0)
        self._append("decoder_ptr", 0)
        self._append("decoder_offset_ptr", 0)

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
        if not 1 <= int(record.action_count) <= MAX_ACTIONS:
            raise ValueError(f"action_count must be in [1, {MAX_ACTIONS}]")
        if len(record.decoder_offset) != int(record.action_count):
            raise ValueError("decoder_offset length must equal action_count")
        if not 0 <= int(record.target) < int(record.action_count):
            raise ValueError("target must be smaller than action_count")
        if not 0 <= int(record.episode_key) <= np.iinfo(np.uint32).max:
            raise ValueError("episode_key must fit uint32")

        _validate_unsigned("encoder_index", record.encoder_index, np.iinfo(np.uint16).max)
        _validate_unsigned("decoder_index", record.decoder_index, np.iinfo(np.uint32).max)
        _validate_unsigned("encoder_offset", record.encoder_offset, np.iinfo(np.uint16).max)
        _validate_unsigned("decoder_offset", record.decoder_offset, np.iinfo(np.uint16).max)

        encoder_values = np.asarray(record.encoder_value, dtype=np.float32)
        narrowed = encoder_values.astype(np.float16)
        if not np.all(np.isfinite(encoder_values)) or not np.all(np.isfinite(narrowed)):
            raise ValueError("encoder_value contains a non-finite or float16-overflow value")

        next_encoder_nnz = self._encoder_nnz + len(record.encoder_index)
        next_decoder_nnz = self._decoder_nnz + len(record.decoder_index)
        next_decoder_words = self._decoder_words + len(record.decoder_offset)
        uint32_max = np.iinfo(np.uint32).max
        if max(next_encoder_nnz, next_decoder_nnz, next_decoder_words) > uint32_max:
            raise ValueError("cache shard pointer exceeds uint32 range")

        self._buffers["encoder_index"].extend(record.encoder_index)
        self._buffers["encoder_value"].extend(record.encoder_value)
        self._buffers["encoder_offset"].extend(record.encoder_offset)
        self._buffers["decoder_index"].extend(record.decoder_index)
        self._buffers["decoder_offset"].extend(record.decoder_offset)

        self._encoder_nnz = next_encoder_nnz
        self._decoder_nnz = next_decoder_nnz
        self._decoder_words = next_decoder_words
        self._append("encoder_ptr", self._encoder_nnz)
        self._append("decoder_ptr", self._decoder_nnz)
        self._append("decoder_offset_ptr", self._decoder_words)
        self._append("target", int(record.target))
        self._append("action_count", int(record.action_count))
        self._append("episode_key", int(record.episode_key))
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
        if self.arrays["decoder_ptr"].size != self.samples + 1:
            raise ValueError("decoder_ptr length does not match sample count")
        if self.arrays["decoder_offset_ptr"].size != self.samples + 1:
            raise ValueError("decoder_offset_ptr length does not match sample count")
        if self.arrays["encoder_offset"].size != self.samples * ENCODER_WORDS:
            raise ValueError("encoder_offset length does not match sample count")
        if self.arrays["target"].size != self.samples:
            raise ValueError("target length does not match sample count")
        if self.arrays["action_count"].size != self.samples:
            raise ValueError("action_count length does not match sample count")
        if self.arrays["episode_key"].size != self.samples:
            raise ValueError("episode_key length does not match sample count")
        pointer_targets = {
            "encoder_ptr": self.arrays["encoder_index"].size,
            "decoder_ptr": self.arrays["decoder_index"].size,
            "decoder_offset_ptr": self.arrays["decoder_offset"].size,
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

    def __len__(self) -> int:
        return self.samples

    def sample(self, local_id: int) -> FeatureView:
        local_id = int(local_id)
        if not 0 <= local_id < self.samples:
            raise IndexError(local_id)
        encoder_start = int(self.arrays["encoder_ptr"][local_id])
        encoder_end = int(self.arrays["encoder_ptr"][local_id + 1])
        decoder_start = int(self.arrays["decoder_ptr"][local_id])
        decoder_end = int(self.arrays["decoder_ptr"][local_id + 1])
        offset_start = int(self.arrays["decoder_offset_ptr"][local_id])
        offset_end = int(self.arrays["decoder_offset_ptr"][local_id + 1])
        encoder_word_start = local_id * ENCODER_WORDS
        return FeatureView(
            encoder_index=self.arrays["encoder_index"][encoder_start:encoder_end],
            encoder_value=self.arrays["encoder_value"][encoder_start:encoder_end],
            encoder_offset=self.arrays["encoder_offset"][
                encoder_word_start : encoder_word_start + ENCODER_WORDS
            ],
            decoder_index=self.arrays["decoder_index"][decoder_start:decoder_end],
            decoder_offset=self.arrays["decoder_offset"][offset_start:offset_end],
            target=int(self.arrays["target"][local_id]),
            action_count=int(self.arrays["action_count"][local_id]),
            episode_key=int(self.arrays["episode_key"][local_id]),
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
    ) -> DatasetSplits:
        if not 0 < validation_ratio < 1:
            raise ValueError("validation_ratio must be strictly between 0 and 1")
        latest_date = max(self.shard_dates)
        dtype = (
            np.uint32
            if self.total_samples <= np.iinfo(np.uint32).max
            else np.uint64
        )
        threshold = int(validation_ratio * (1 << 32))
        train_parts = []
        in_distribution_parts = []
        latest_parts = []
        for shard_id, shard in enumerate(self.shards):
            global_ids = np.arange(
                self.starts[shard_id],
                self.ends[shard_id],
                dtype=dtype,
            )
            if self.shard_dates[shard_id] == latest_date:
                latest_parts.append(global_ids)
                continue
            mixed = _mix_episode_keys(
                shard.arrays["episode_key"], validation_seed
            )
            validation_mask = mixed.astype(np.uint64) < threshold
            in_distribution_parts.append(global_ids[validation_mask])
            train_parts.append(global_ids[~validation_mask])

        def combine(parts: list[np.ndarray], name: str) -> np.ndarray:
            nonempty = [part for part in parts if part.size]
            if not nonempty:
                raise ValueError(f"{name} split is empty")
            return np.concatenate(nonempty).astype(dtype, copy=False)

        return DatasetSplits(
            train=combine(train_parts, "train"),
            in_distribution=combine(
                in_distribution_parts, "in-distribution validation"
            ),
            latest=combine(latest_parts, "latest-date validation"),
            latest_date=latest_date,
        )

    def collate(self, index_batch: IndexBatch) -> CachedBatch:
        global_ids = np.asarray(index_batch.global_ids, dtype=np.int64)
        shard_ids = np.searchsorted(self.ends, global_ids, side="right")
        encoder_indices = []
        encoder_values = []
        decoder_indices = []
        encoder_offsets = np.empty(global_ids.size * ENCODER_WORDS, dtype=np.int32)
        decoder_offsets = np.empty(global_ids.size * MAX_ACTIONS, dtype=np.int32)
        targets = np.empty(global_ids.size, dtype=np.int64)
        action_counts = np.empty(global_ids.size, dtype=np.int64)
        encoder_base = 0
        decoder_base = 0

        for row, (global_id, shard_id) in enumerate(zip(global_ids, shard_ids)):
            local_id = int(global_id - self.starts[int(shard_id)])
            sample = self.shards[int(shard_id)].sample(local_id)
            encoder_indices.append(sample.encoder_index)
            encoder_values.append(sample.encoder_value)
            decoder_indices.append(sample.decoder_index)

            enc_slice = slice(row * ENCODER_WORDS, (row + 1) * ENCODER_WORDS)
            encoder_offsets[enc_slice] = (
                sample.encoder_offset.astype(np.int32) + encoder_base
            )
            dec_start = row * MAX_ACTIONS
            dec_end = dec_start + sample.action_count
            decoder_offsets[dec_start:dec_end] = (
                sample.decoder_offset.astype(np.int32) + decoder_base
            )
            decoder_offsets[dec_end : dec_start + MAX_ACTIONS] = (
                decoder_base + sample.decoder_index.size
            )
            targets[row] = sample.target
            action_counts[row] = sample.action_count
            encoder_base += int(sample.encoder_index.size)
            decoder_base += int(sample.decoder_index.size)

        return CachedBatch(
            encoder_index=np.concatenate(encoder_indices).astype(np.int32, copy=False),
            encoder_value=np.concatenate(encoder_values).astype(np.float16, copy=False),
            encoder_offset=encoder_offsets,
            decoder_index=np.concatenate(decoder_indices).astype(np.int32, copy=False),
            decoder_offset=decoder_offsets,
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
