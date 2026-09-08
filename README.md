# Self-contained `train_fibo_edit_standard.py` copy

A trimmed, standalone copy of everything `train_fibo_edit_standard.py` (the
fibo_edit_next training entrypoint, originally at
`training/fibo_edit/train_fibo_edit_standard.py` in Bria's internal
`foundation-training` monorepo) actually needs to run locally, with every
unused import, dead function, and non-training file removed. Verified with
`pyflakes` (0 warnings) and an import smoke test — see
[Verification](#verification).

10,138 → 6,268 lines (~38% smaller) across the retained files, for **zero**
behavior change on the actual training path (STANDARD + DREAMBOOTH data
modes, LoRA/full fine-tune, flash + varlen attention, FSDP, smolLM text
encoder, EMA, resume/reinit — everything `TrainConfig` still exposes).

## Files

| File | What it is |
|---|---|
| `train_fibo_edit_standard.py` | Entry point (`python train_fibo_edit_standard.py --config ...`) |
| `train_common.py` | Shared trainer plumbing (data-config parsing, latent prep, CFG dropout, varlen packing) |
| `dataset_factory.py` | Dataloader (`DatasetBuilder`) — STANDARD (packed tars) + DREAMBOOTH (raw images) modes |
| `bria_utils.py` | **Merged** from the source's `bria_utils.py` + `bria4_utils.py` (see [Merged files](#merged-files) below) — RoPE (`FluxPosEmbed`), LR scheduler, smolLM text-encoder init, prompt embedding, wandb init, timestep samplers |
| `init_handler.py` | Transformer construction, weight loading, LoRA setup, FSDP prepare |
| `checkpoint_loader.py` / `checkpoint_saver.py` | Checkpoint resume/save (unmodified from source) |
| `transformer_bria_repa.py` | The model (`Bria4Transformer2DModel`) |
| `transformer_bria_varlen.py` | Varlen-attention transformer block variants |
| `lora_utils.py` | `add_lora` / `load_lora` (unmodified from source) |
| `latent_packing.py` | **New** — 5 pure functions extracted from `pipeline_bria_wan.py`'s `BriaPipeline` (see below) |
| `utils/torch_utils.py` | Accelerator/FSDP setup, WebDataset tar loading, torch.compile helper |
| `vae_wan.json.out`, `bria_transformer.json.out`, `bria_transformer_debug.json.out` | Model/VAE config JSON loaded at runtime relative to `__file__` (renamed from the source's `flux_transformer*.json.out` — the config's own `_class_name` field said `FluxTransformer2DModel`, but the code never reads that field to pick a class; it always calls `Bria4Transformer2DModel.from_config(...)` explicitly. Renamed the files and fixed `_class_name` to match what's actually loaded; left `_name_or_path`, an accurate provenance note pointing at the original FLUX.1-dev snapshot this config was derived from) |
| `requirements.txt` | **New** — full pinned dependency list for running this copy standalone (the source repo's own `requirements.txt` is 3 lines and assumes the `briatorch:pytorch-2.10.0-aws` training image already has everything else baked in); versions match what was actually smoke-tested — see below |

Not copied at all — genuinely unrelated to running this training script:
`sky/`, `launch_scripts/`, `launcher/`, `tests/`, `reframe/`, `expand_bg/`,
`edit_t2i_aesthetics/`, `merge_loras_and_fuse.py`, `split_buckets.py`,
`validate_dataset.py`, `probe_bs.py`, the sibling `train_fibo*.py` scripts
(unified / cfg_distillation / dpo — different training scripts sharing some
of these modules), `dockers/`.

## The big cut: the inference pipeline was never used by training

`train_common.py` and `train_fibo_edit_standard.py` only ever called 5
**static** methods directly on the `Bria4EditPipelineWan` class object
(`Bria4EditPipelineWan._pack_latents(...)`, never on an instance) — pure
tensor reshape/patchify math with no `self` dependency. Those 5 methods
actually live on `Bria4EditPipelineWan`'s parent, `BriaPipeline`
(`pipeline_bria_wan.py`). Grepped the entire training call graph: the
pipeline classes themselves are **never instantiated** — no
`Bria4EditPipelineWan(...)` / `BriaPipeline(...)` constructor call exists
anywhere in the training path. Training runs entirely on precomputed
latents from packed tars; it never encodes or decodes an image.

That means the full generation/inference machinery — `encode_prompt`,
the denoising sampling loop, VAE decode, classifier-free/adaptive-projected
guidance — was dead weight in the training process's import graph, dragging
in three more files nothing in training actually used:
- `pipeline_bria4_edit_wan.py` / `pipeline_bria_wan.py` — the full
  `Bria4EditPipelineWan(BriaPipeline(FluxPipeline))` inference pipelines
- `transformer_bria.py` — imported by `pipeline_bria_wan.py` **only** for
  a constructor type annotation (`transformer: BriaTransformer2DModel`)
  that nothing in the training path ever satisfies (the real transformer
  passed around is `Bria4Transformer2DModel` from `transformer_bria_repa.py`)
- `vae2_2.py` (`Wan2_2_VAE`) — only reachable through the pipelines' own
  `__call__`/decode path; training never touches a VAE
- `apg_utils.py` — guidance helpers, only used inside the pipelines' own
  `__call__`

`latent_packing.py` is the replacement: the 5 static methods, copied
verbatim as plain module-level functions (`pack_latents`,
`pack_latents_no_patch`, `unpack_latents`, `unpack_latents_no_patch`,
`prepare_latent_image_ids`), needing only `torch`. `train_common.py` and
`train_fibo_edit_standard.py` import from it instead of from
`pipeline_bria4_edit_wan`.

## Merged files

The source repo also has a separate `checkpoint_loader.py` / `checkpoint_saver.py`
pair — kept as two files here, unmodified: they're genuinely complementary
(load vs. save), zero overlapping symbol names, both directly imported by
`train_fibo_edit_standard.py` at two different call sites. Nothing to merge.

`bria_utils.py` / `bria4_utils.py` were a different story. In the source
repo the split makes some sense (`bria_utils.py` is older/shared
infrastructure used by several training scripts; `bria4_utils.py` holds
bria4-generation-specific additions), but after trimming both down to only
what `train_fibo_edit_standard.py` actually uses, `bria4_utils.py`'s *only*
reason to be a separate file — `get_env_prefix` — turned out to already
exist in `bria_utils.py` too, **defined identically** (byte-for-byte
identical body, just single- vs double-quote style):

```python
# bria_utils.py                      # bria4_utils.py
def get_env_prefix():                def get_env_prefix():
    env = os.environ.get(...)            env = os.environ.get(...)
    ...                                   ...
```

Not a naming coincidence — a genuine duplicate. Every other symbol in each
file was only ever used by call sites that already knew which file to import
it from, so there was no real functional split left to preserve. Merged
everything into one `bria_utils.py`, keeping a single `get_env_prefix`, and
updated `train_fibo_edit_standard.py`'s two import blocks into one. Deleted
`bria4_utils.py`.

## Other removals

- **Dead imports** (verified via `pyflakes`, each individually confirmed
  unreferenced): `is_torch_version` (`transformer_bria_repa.py`), bare
  `import diffusers` (was only in the now-removed `pipeline_bria_wan.py`),
  `flex_attention` (`transformer_bria_varlen.py` — imported but the varlen
  path only ever calls `varlen_attn`), `get_logger`/`shutil`/`threading`
  (`utils/torch_utils.py`, only used by the also-removed `AsyncCheckpointSaver`).
- **`interleave_for_attention` / `split_from_attention`**
  (`transformer_bria_varlen.py`) — explicitly marked in their own comment as
  "Legacy functions for backward compatibility," superseded by
  `build_interleave_indices`/`interleave_with_indices`/`split_with_indices`.
  Zero call sites anywhere.
- **`AsyncCheckpointSaver`** (`utils/torch_utils.py`) — fully self-contained
  class, defined and never instantiated anywhere (`checkpoint_saver.py` uses
  a different mechanism). `get_params` (`utils/torch_utils.py`) — same, zero
  call sites.
- **`T5`/`Llama` text-encoder paths** (`init_text_encoder`, plus the
  `get_t5_prompt_embeds`/`get_llama_prompt_embeds` helpers they called) —
  `train_fibo_edit_standard.py`'s own `validate_config()` hard-rejects any
  `text_encoder_type` other than `"smolLM"` (`raise ValueError(...)` if
  not), so these branches were provably unreachable, not just unused.
  `init_text_encoder` is now smolLM-only.
- **`DatasetMode.DPO`** (`dataset_factory.py`) — `preprocess_dpo`,
  `collate_dpo`, `_stack_tensors` (only DPO's own helper). No code path in
  `TrainConfig`/`setup_dataloader` can ever construct a `DatasetConfig` with
  `mode=DPO` — that's `train_fibo_edit_dpo.py`'s data mode, a different
  training script. `DatasetMode.DREAMBOOTH` (a real, reachable
  `TrainConfig.use_dreambooth` option) was kept.
- **CFG-distillation teacher / DPO reference transformer** (`init_handler.py`)
  — `TransformerInitHandler._setup_teacher_model`, `_setup_ref_model`, and
  `_load_lora_weights` (only called from the teacher path). Removed because
  `train_fibo_edit_standard.py`'s `setup_models()` always passes
  `create_teacher=False, create_ref=False` — the `create_teacher`/
  `create_ref` params, and the `teacher_transformer`/`ref_transformer`
  fields on `TransformerInitResult`, are gone too.
- **Everything else T5/Llama/CLIP-embedding-only in the original `bria_utils.py`**:
  `get_text`, `get_by_t5_prompt_embeds`, `get_t5_prompt_embeds` (a *second*,
  differently-scoped copy of that name — distinct from `bria4_utils.py`'s
  own `get_t5_prompt_embeds`, which was itself removed above — only ever
  imported by the now-removed `pipeline_bria_wan.py`), `get_original_sigmas`,
  `is_ng_none`, `CudaTimerContext` (only imported by `train_fibo.py`, a
  different script), `compute_density_for_timestep_sampling`,
  `compute_loss_weighting_for_sd3`, `initialize_distributed`,
  `get_clip_prompt_embeds`.
- **Inference-only helpers in the original `bria4_utils.py`**: `load_checkpoint`
  (explicitly marked `DEPRECATED` in its own docstring — superseded by
  `checkpoint_loader.py`'s version, which is what's actually imported),
  `init_inference_scheduler` and `InferenceSchedule` (both explicitly
  eval/sampling-schedule-only per their own docstrings — used by the
  discarded pipelines, never by training), `init_data_config` (unused,
  legacy channel-config builder), `get_DINO_encoding` (unused, pulls in a
  `timm` dependency for nothing), `create_attention_matrix`, and the unused
  `SAMPLERS` lookup dict.

## Verification

```
python3 -m pyflakes *.py utils/*.py   # 0 warnings
python3 -m py_compile *.py utils/*.py # all files compile
```

Every file also import-checked successfully in this dev environment except
`transformer_bria_varlen.py` / `utils/torch_utils.py` / `dataset_factory.py`,
which fail with `ModuleNotFoundError: torch.nn.attention.varlen` — confirmed
**pre-existing**, not caused by this trim (the untouched original
`transformer_bria_varlen.py` in the source monorepo fails the identical
import the same way on this box's torch 2.9.0; the real training image
`briatorch:pytorch-2.10.0-aws` has this submodule).

## Actually run: local smoke test (train → checkpoint → resume)

Beyond the static checks above, this copy was run for real on a single GPU
(no S3, no real training data — `random_latents: true`, `transformer_architecture: a1-t`,
`debug: 0` with hand-picked tiny/fast settings instead of `debug: 1`, since
`debug: 1`'s `set_debug_env()` unconditionally forces `lora_rank = 128`,
which would have made a true `lora_rank: 0` test impossible). Two full
train → save-checkpoint → reload-checkpoint → resume-training cycles:

| | full fine-tune (`lora_rank: 0`) | LoRA (`lora_rank: 64`) |
|---|---|---|
| Fresh train, 5 steps | ✅ finite loss (~2.3), checkpoint saved at step 4 | ✅ finite loss (~2.3-2.4), checkpoint saved at step 4 |
| Checkpoint format | `model.safetensors` + `optimizer.bin`/`scheduler.bin`/`random_states_*.pkl` (accelerate full state), 553M | PEFT adapter format (`adapter_model.safetensors` + `adapter_config.json`) + `optimizer.pt`/`scheduler.pt`, 105M — correctly smaller |
| Resume + 2 more steps (→7) | ✅ all 5 components loaded ("model/optimizer/scheduler/dataloader sampler/random states loaded successfully"), `global_step` continued at 4→7, not reset | ✅ adapter + optimizer + scheduler loaded, `global_step` continued 4→7 |
| Weights actually changed between the two checkpoints | ✅ 264/270 tensors changed, max |Δ| ≈ 2e-4 | ✅ 210/210 (100%) adapter tensors changed, max |Δ| ≈ 2e-4 |
| Second checkpoint saves correctly mid-resume | ✅ (`checkpoint_000006`) | ✅ (`checkpoint_000006`) |

**One genuine bug found — not in this repo, an environment dependency
mismatch**: the LoRA resume initially failed with
`ImportError: cannot import name 'EmbeddingParallel' from 'transformers.integrations.tensor_parallel'`,
raised from inside `peft==0.19.1`'s `load_adapter()` → `_maybe_shard_state_dict_for_tp()`
(this dev box's pre-installed `peft` expects a `transformers` internal class
`transformers==4.56.0` here doesn't have). `checkpoint_loader.py`'s
`_load_lora_weights()` — the code that calls `load_adapter()` — is
unmodified/verbatim from the source repo; the bug is purely a `peft`/`transformers`
version pairing issue in this environment. Fixed by `pip install -U peft==0.20.0`;
re-ran and it passed cleanly. Worth checking this same `peft`/`transformers`
pairing in the real training image before assuming LoRA-resume works there.

**Test-harness-only workarounds used (none touch the files above)**: a
`sitecustomize.py` stub providing a dummy `torch.nn.attention.varlen.varlen_attn`
(only needed because `utils/torch_utils.py`/`transformer_bria_varlen.py`
import it unconditionally at module load even when `use_varlen_attention: false`
— both smoke tests used flash attention, so the stub is never actually
called), and a patch forcing `wandb.init(mode="disabled")` (`init_wandb()`
in `bria_utils.py` hardcodes `mode="online"`, which ignores `WANDB_MODE` and
tries a real network call otherwise).

### Follow-up round: FSDP, multi-ref, T2I, EMA

Four more train → checkpoint → resume cycles, each isolating one previously-untested
code path (still single-GPU, flash attention, `lora_rank: 0`, same `a1-t`/
`random_latents` setup as above unless noted):

| | result |
|---|---|
| **`use_fsdp: true`** (`fsdp_sharding_strategy: full`) | ✅. PyTorch auto-degrades `FULL_SHARD`→`NO_SHARD` at world_size=1 (expected, not a bug), but genuinely exercises accelerate's FSDP-specific save/load path — distinct checkpoint format (`pytorch_model_fsdp.bin`, FSDP-specific optimizer save/load logging) from the plain (non-FSDP) `model.safetensors` path tested earlier. This is the path every real production config actually uses (`use_fsdp: 1` + hybrid sharding), so worth having covered even in this degraded single-GPU form. Resume loaded + continued correctly. |
| **Multi-ref (`RL_NUM_CONTEXTS=3`)** | ✅. Confirmed `train_common.py::prepare_latents()`'s context-packing loop (`ctx_list` of arbitrary length) works under **flash attention**, not just varlen — this is the actual code path `multi_ref_curated`-style training uses. Finite losses, checkpoint round-trips. |
| **T2I (`RL_NUM_CONTEXTS=0`, no context latents at all)** | ✅. The opposite edge — `context_patched_latents=None` branch. Finite losses, checkpoint round-trips. |
| **`use_ema: true`** | ✅. `transformer_ema.bin` (full-gathered, for eval) and `ema_state_rank_0.pt` (per-rank shadow-params shard, for resume) both saved correctly alongside the regular checkpoint; resume logged "Loading per-rank EMA shard" and continued; second checkpoint saved both EMA files again correctly. |

**Not run**: `use_varlen_attention: true` and anything requiring it
(`train_batch_size > 1`, context dropout) — blocked by the missing
`torch.nn.attention.varlen` on this box's torch 2.9.0 (needs the real
training image or a torch upgrade to ≥2.10). `torch.compile`
(`use_torch_compile: true`) — higher time cost for a checkpoint-roundtrip
test, lower marginal value. DREAMBOOTH data mode — needs real image files +
`metadata.csv`, more setup than the rest. True multi-GPU/multi-node FSDP
sharding — only 1 GPU on this box. `reinit_optimizer`/`reinit_scheduler` on
resume. And anything beyond `a1-t`-scale/a few steps — no attempt made to
validate loss actually *decreases* over a meaningful training run, since
random synthetic latents/captions have no signal to fit and that wouldn't
have proven anything.
