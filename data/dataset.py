"""
Dataset factory for fibo_edit training scripts.

Two modes (trimmed from the shared original's three -- DPO mode, used only by the
separate train_fibo_edit_dpo.py, is unreachable from train_fibo_edit_standard.py's
TrainConfig/setup_dataloader and was removed; see README.md in this directory):
- STANDARD: single latent + context
- DREAMBOOTH: raw image pairs with on-the-fly VAE encoding
"""

import csv
import glob
import json
import os
import random
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

import torch
import webdataset as wds
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import DataLoader
from datasets import IterableDataset
from torchvision import transforms
from utils.accelerator import load_dataset_from_tars


class DatasetMode(Enum):
    STANDARD = "standard"
    DREAMBOOTH = "dreambooth"


RESOLUTIONS_1K = {
    0.67: (832, 1248),
    0.778: (896, 1152),
    0.883: (960, 1088),
    1.000: (1024, 1024),
    1.133: (1088, 960),
    1.286: (1152, 896),
    1.462: (1216, 832),
    1.600: (1280, 800),
    1.750: (1344, 768),
}


def find_closest_resolution(image_width, image_height):
    image_aspect = image_width / image_height
    closest_ratio = min(RESOLUTIONS_1K.keys(), key=lambda x: abs(x - image_aspect))
    return RESOLUTIONS_1K[closest_ratio]


class DreamBoothDataset(torch.utils.data.Dataset):
    """Dataset for fine-tuning with paired source/target raw images and edit prompts.

    Images are dynamically resized and center-cropped to the closest 1K aspect ratio.
    Supports loading from local CSV (metadata.csv) or HuggingFace datasets.
    """

    def __init__(
        self,
        instance_data_dir: Optional[str] = None,
        dataset_name: Optional[str] = None,
        dataset_config_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        image_column: str = "image",
        context_image_column: str = "context_image",
        caption_column: str = "caption",
        raw_caption: bool = False,
        edit_instruction_only_prob: float = 0.0,
    ):
        self.raw_caption = raw_caption
        self.edit_instruction_only_prob = edit_instruction_only_prob

        if dataset_name is not None:
            self.target_images, self.source_images, self.prompts = self._load_from_huggingface(
                dataset_name, dataset_config_name, cache_dir,
                image_column, context_image_column, caption_column,
            )
        elif instance_data_dir is not None:
            self.target_images, self.source_images, self.prompts = self._load_from_csv(instance_data_dir)
        else:
            raise ValueError("Either dataset_name or instance_data_dir must be provided")

        self.to_tensor_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.target_images)

    def _load_from_huggingface(self, dataset_name, config_name, cache_dir,
                                image_col, context_col, caption_col):
        from datasets import load_dataset

        dataset = load_dataset(dataset_name, config_name, cache_dir=cache_dir)
        column_names = dataset["train"].column_names

        for col, label in [(image_col, "image"), (context_col, "context_image"), (caption_col, "caption")]:
            if col not in column_names:
                raise ValueError(f"Column '{col}' not found in dataset. Available: {column_names}")

        targets = list(dataset["train"][image_col])
        sources = list(dataset["train"][context_col])
        prompts = [self._parse_caption(c) for c in dataset["train"][caption_col]]
        return targets, sources, prompts

    def _load_from_csv(self, instance_data_dir):
        data_root = Path(instance_data_dir)
        metadata_path = data_root / "metadata.csv"
        if not metadata_path.exists():
            raise ValueError(f"metadata.csv not found in {data_root}")

        targets, sources, prompts = [], [], []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                target_path = data_root / row["output_file_name"]
                source_path = data_root / row["input_file_name"]
                if not target_path.exists():
                    raise ValueError(f"Target image not found: {target_path}")
                if not source_path.exists():
                    raise ValueError(f"Source image not found: {source_path}")
                targets.append(target_path)
                sources.append(source_path)
                prompts.append(self._parse_caption(row["caption"]))
        return targets, sources, prompts

    def _parse_caption(self, caption):
        if self.raw_caption:
            return caption.strip()
        try:
            parsed = json.loads(caption)
            return json.dumps(parsed)
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Caption must be valid JSON: {e}. Got: {str(caption)[:100]}")

    def _load_image(self, image_or_path):
        if isinstance(image_or_path, (str, Path)):
            return Image.open(image_or_path)
        return image_or_path

    def _process_image(self, image, target_width, target_height):
        image = exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        img_w, img_h = image.size
        target_aspect = target_width / target_height
        if (img_w / img_h) > target_aspect:
            scale = target_height / img_h
        else:
            scale = target_width / img_w

        new_size = (int(img_h * scale), int(img_w * scale))
        image = transforms.Resize(new_size, interpolation=transforms.InterpolationMode.BILINEAR)(image)
        image = transforms.CenterCrop((target_height, target_width))(image)
        return self.to_tensor_normalize(image)

    def _get_caption(self, index):
        caption = self.prompts[index]
        if not self.raw_caption and self.edit_instruction_only_prob > 0 and random.random() < self.edit_instruction_only_prob:
            try:
                return json.loads(caption)["edit_instruction"]
            except (json.JSONDecodeError, KeyError):
                return caption
        return caption

    def __getitem__(self, index):
        target_image = self._load_image(self.target_images[index])
        source_image = self._load_image(self.source_images[index])

        target_image = exif_transpose(target_image)
        if target_image.mode != "RGB":
            target_image = target_image.convert("RGB")
        target_width, target_height = find_closest_resolution(*target_image.size)

        return {
            "instance_images": self._process_image(target_image, target_width, target_height),
            "context_images": self._process_image(source_image, target_width, target_height),
            "caption": self._get_caption(index),
        }


@dataclass
class DatasetConfig:
    mode: DatasetMode
    batch_size: int
    training_dirs: List[str] = field(default_factory=list)
    random_latents: bool = False
    random_latents_resolution: int = 256  # Only used when random_latents=True

    # STANDARD mode options
    edit_instruction_only_prob: float = 0.0

    # VAE config for random latents
    vae_latent_channels: int = 48
    vae_compression_rate: int = 16

    # Interleaved multi-resolution loading
    interleave_resolutions: bool = False

    # DataLoader
    num_workers: int = 0
    prefetch_factor: Optional[int] = 4
    max_sequence_length: int = 3000

    # DREAMBOOTH mode options
    instance_data_dir: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    cache_dir: Optional[str] = None
    image_column: str = "image"
    context_image_column: str = "context_image"
    caption_column: str = "caption"
    raw_caption: bool = False
    vae_model: Optional[object] = None  # AutoencoderKLWan, set at runtime
    vae_device: Optional[object] = None  # torch.device, set at runtime
    vae_weight_dtype: Optional[object] = None  # torch.dtype, set at runtime


# Context-latent tar keys: indexed (multi-ref) context_latents_{i}, or legacy
# single-context context_latents_{w}_{h}. The two are mutually exclusive per record.
_CTX_IDX_RE = re.compile(r"^context_latents_(\d+)$")
_CTX_LEGACY_RE = re.compile(r"^context_latents_\d+_\d+$")


def preprocess_standard(record, config):
    """Extract single latent + context."""
    record = record.get("pickle", record)
    example = {}

    # Caption (with optional edit_instruction extraction)
    if config.edit_instruction_only_prob > 0 and random.random() < config.edit_instruction_only_prob:
        try:
            example["caption"] = json.loads(record["caption"])["edit_instruction"]
        except (json.JSONDecodeError, KeyError):
            example["caption"] = record["caption"]
    else:
        example["caption"] = record["caption"]

    # Target latent.
    for key in record:
        if key.startswith("latents_") and not key.startswith("context_"):
            example["pixel_values"] = record[key]
    if "pixel_values" not in example:
        raise KeyError(f"No latent key found. Available: {list(record.keys())}")

    # Context latents -> ordered list. New multi-ref tars use index-only keys
    # context_latents_{i}; existing single-context tars use the legacy
    # context_latents_{w}_{h} key, mapped to a single context at index 0.
    ctx_idx = sorted(
        ((int(m.group(1)), record[k]) for k in record if (m := _CTX_IDX_RE.match(k))),
        key=lambda t: t[0],
    )
    if ctx_idx:
        example["context_latents"] = [v for _, v in ctx_idx]
    else:
        legacy = [record[k] for k in record if _CTX_LEGACY_RE.match(k)]
        if legacy:
            example["context_latents"] = [legacy[0]]
    # else: no context key -> T2I sample (context = None), leave unset

    return example


def collate_standard(examples):
    """Collate for STANDARD mode -> (pixel_values, context_latents, captions, image_dims)

    Always returns lists (not stacked tensors) to support variable resolutions.
    This unified format works for both single-resolution and multi-resolution training.

    Returns:
        pixel_values: List[Tensor] - each [C, H_i, W_i]
        context_latents: None (T2I) | List[List[Tensor]] - outer = batch, inner = ordered
            contexts per sample (length >= 1; refs may differ in size from target/each other)
        captions: List[str]
        image_dims: List[Tuple[int, int]] - [(H_1, W_1), (H_2, W_2), ...]
    """
    captions = [ex["caption"] for ex in examples]

    # Return list of tensors, not stacked
    pixel_values = [
        ex["pixel_values"].to(memory_format=torch.contiguous_format).float()
        for ex in examples
    ]

    # Ordered list of context latents per sample. No shape-equality check vs the target:
    # references may differ in size from the target and from each other (multi-ref).
    context_latents = None
    if "context_latents" in examples[0]:
        context_latents = [
            [c.to(memory_format=torch.contiguous_format).float() for c in ex["context_latents"]]
            for ex in examples
        ]

    # Extract per-sample dimensions from tensor shapes
    # pixel_values[i] is [C, H_i, W_i]
    image_dims = [(pv.shape[1], pv.shape[2]) for pv in pixel_values]  # List[(H, W)]

    return pixel_values, context_latents, captions, image_dims


def collate_dreambooth(examples, vae_model, device, weight_dtype):
    """Collate for DREAMBOOTH mode -> (pixel_values, context_latents, captions, image_dims)

    Encodes raw images with VAE on-the-fly and returns latents in the same
    list-of-tensors format as collate_standard.

    The returned latents are raw VAE output (NOT shifted/scaled) because the
    training script's prepare_training_latents applies shift/scale itself.
    """
    captions = [ex["caption"] for ex in examples]

    pixel_values = []
    context_latents = []
    image_dims = []

    with torch.no_grad():
        for ex in examples:
            target_img = ex["instance_images"]
            context_img = ex["context_images"]

            target_latent = vae_model.encode(
                target_img.unsqueeze(0).unsqueeze(2).to(device, dtype=weight_dtype)
            ).latent_dist.mean[:, :, 0].float().squeeze(0)

            context_latent = vae_model.encode(
                context_img.unsqueeze(0).unsqueeze(2).to(device, dtype=weight_dtype)
            ).latent_dist.mean[:, :, 0].float().squeeze(0)

            pixel_values.append(target_latent)
            context_latents.append(context_latent)
            image_dims.append((target_latent.shape[1], target_latent.shape[2]))

    return pixel_values, context_latents, captions, image_dims


class DatasetBuilder:
    """Builds dataloaders for training."""

    def __init__(self, config: DatasetConfig, rank: int, world_size: int, seed: int):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.rng = random.Random(seed + rank)

    def _get_preprocess_fn(self):
        config = self.config
        return lambda r: preprocess_standard(r, config)

    def _get_collate_fn(self):
        return collate_standard

    def build(self):
        """Returns a get_dataloader() function that creates fresh dataloaders."""
        config = self.config

        if config.mode == DatasetMode.DREAMBOOTH:
            return self._build_dreambooth()

        preprocess = self._get_preprocess_fn()
        collate = self._get_collate_fn()

        def get_dataloader():
            seed = self.rng.randint(0, 10**6)

            if config.random_latents:
                dataset = self._random_dataset(seed)
                dataset = dataset.map(preprocess)
            elif config.interleave_resolutions and len(config.training_dirs) > 1:
                # RandomMix doesn't support .map(), so preprocess is applied per-stream
                dataset = self._interleaved_dataset(seed, preprocess)
            else:
                dataset = load_dataset_from_tars(
                    training_dirs=config.training_dirs,
                    rank=self.rank,
                    world_size=self.world_size,
                    seed=seed,
                    slice=False,
                )
                dataset = dataset.shuffle(10000, rng=random.Random(seed))
                dataset = dataset.map(preprocess)

            return DataLoader(
                dataset,
                collate_fn=collate,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                drop_last=True,
                prefetch_factor=config.prefetch_factor,
            )

        return get_dataloader

    def _interleaved_dataset(self, seed, preprocess):
        """One wds.WebDataset per resolution dir, combined with wds.RandomMix.

        Each stream uses resampled=True so shards are sampled infinitely with
        replacement. This prevents RandomMix from permanently dropping shorter
        resolution streams when they exhaust (the default longest=True behavior).
        The training loop runs to max_train_steps so epoch boundaries don't matter.

        Mix probabilities are tar-count-proportional, so each AR dir is sampled
        in proportion to its data size (each tar ≈ chunk_size rows, so tar count
        is a near-exact proxy for row count). This ensures rows-proportional
        sampling across ARs — every row has equal probability of being seen per
        step, regardless of which AR bucket it lives in.
        """
        streams = []
        tar_counts = []
        for training_dir in self.config.training_dirs:
            ds = load_dataset_from_tars([training_dir], self.rank, self.world_size, seed, slice=False, resampled=True)
            ds = ds.shuffle(10000, rng=random.Random(seed))
            ds = ds.map(preprocess)
            streams.append(ds)
            tar_counts.append(len(glob.glob(f"{training_dir}/*.tar")))

        total = sum(tar_counts)
        if total > 0:
            probs = [c / total for c in tar_counts]
        else:
            probs = None  # fall back to uniform if globbing returned 0 everywhere
        return wds.RandomMix(streams, probs=probs, longest=True)

    def _random_dataset(self, seed):
        """Synthetic dataset for debugging / bs-capacity probing.

        Per-sample latent shapes are sampled from the resolution family's real AR buckets
        (measured image dims / 16), not a fixed square. Emits N indexed context latents
        (context_latents_0..N-1) so the multi-ref forward is exercised for any N. Caption length
        is controllable so text tokens match the real (per-source) distribution. Knobs via env:
          RL_NUM_CONTEXTS (int, default 1)  -- number of context latents (0 = T2I).
          RL_TEXT_LEN     (int, default max_sequence_length) -- approx caption tokens.
        """
        config = self.config
        res = config.random_latents_resolution
        C = config.vae_latent_channels
        rng = random.Random(seed)
        # Latent (H,W) buckets per resolution family = real image <W>x<H> dirs / 16.
        _AR = {
            256:  [(19, 13), (18, 14), (16, 16), (14, 18), (13, 19), (12, 20), (12, 21)],
            512:  [(39, 26), (37, 27), (35, 29), (32, 32), (30, 34), (28, 36),
                   (27, 37), (26, 38), (26, 39), (25, 40), (24, 42), (23, 44)],
            1024: [(64, 64), (57, 71), (55, 74), (53, 76), (52, 78), (51, 79),
                   (50, 81), (48, 85), (78, 52), (74, 55), (71, 57)],
        }
        buckets = _AR.get(res)
        if buckets is None:
            s = res // config.vae_compression_rate
            buckets = [(s, s)]
        n_ctx = int(os.environ.get("RL_NUM_CONTEXTS", "1"))
        text_len = int(os.environ.get("RL_TEXT_LEN", str(config.max_sequence_length)))
        caption = " ".join(["the"] * max(1, text_len))  # ~text_len tokens; content irrelevant, length matters

        def gen():
            while True:
                data = {"caption": caption}
                th, tw = rng.choice(buckets)
                if config.mode == DatasetMode.STANDARD:
                    data[f"latents_{th}_{tw}"] = torch.randn(C, th, tw)
                else:
                    data[f"good_latents_{th}_{tw}"] = torch.randn(C, th, tw)
                    data[f"bad_latents_{th}_{tw}"] = torch.randn(C, th, tw)
                for i in range(n_ctx):
                    ch, cw = rng.choice(buckets)
                    data[f"context_latents_{i}"] = torch.randn(C, ch, cw)
                yield data

        return IterableDataset.from_generator(gen)

    def _build_dreambooth(self):
        config = self.config

        dataset = DreamBoothDataset(
            instance_data_dir=config.instance_data_dir,
            dataset_name=config.dataset_name,
            dataset_config_name=config.dataset_config_name,
            cache_dir=config.cache_dir,
            image_column=config.image_column,
            context_image_column=config.context_image_column,
            caption_column=config.caption_column,
            raw_caption=config.raw_caption,
            edit_instruction_only_prob=config.edit_instruction_only_prob,
        )

        vae_model = config.vae_model
        vae_device = config.vae_device
        vae_weight_dtype = config.vae_weight_dtype

        def collate_fn(examples):
            return collate_dreambooth(examples, vae_model, vae_device, vae_weight_dtype)

        rng = self.rng

        def get_dataloader():
            current_seed = rng.randint(0, 10**6)
            generator = torch.Generator().manual_seed(current_seed)
            return DataLoader(
                dataset,
                collate_fn=collate_fn,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                drop_last=True,
                shuffle=True,
                generator=generator,
                prefetch_factor=config.prefetch_factor,
            )

        return get_dataloader


def calculate_total_batch_size(
    json_data_input_path: str,
    train_batch_size: int,
    world_size: int,
    single_data_dir: bool = False,
    debug: bool = False,
    random_latents: bool = False,
) -> int:
    """Calculate total batch size across all processes (without accumulation).

    Note: single_data_dir refers to the CLI flag that overrides ALL GPUs to use
    one directory. JSON-based "mixed" markers still use JSON for batch calculation.
    """
    if debug or random_latents or single_data_dir:
        return train_batch_size * world_size

    with open(json_data_input_path) as f:
        data_config = json.load(f)

    return sum(v["batch_size"] * sum(v["gpu_allocation_map"]) for v in data_config.values())
