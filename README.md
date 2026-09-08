# bria_edit

Standalone training code for **fibo_edit_next** — Bria's edit-conditioned
image diffusion model (a FLUX-style transformer, flow-matching objective,
Wan2.2 VAE latent space, smolLM text conditioning). This repo is a trimmed,
self-contained copy of the training entrypoint and everything it needs to
run — no unrelated inference/eval code, no dead imports. See
`README_INTERNAL.md` if you want the full "why is it structured this way"
writeup; this doc is just how to actually use it.

## Install

```bash
pip install -r requirements.txt
```

`torch>=2.10.0` is required for **varlen attention** specifically
(`torch.nn.attention.varlen` was added in 2.10). Flash attention (the
default, `use_varlen_attention: false`) works fine on 2.9.x too. See
[Troubleshooting](#troubleshooting) if you hit import errors here.

## Quickstart: run it locally in 2 minutes (no real data needed)

The fastest way to confirm your environment works is a tiny synthetic-data
run — no S3, no packed tars, no real checkpoint. `random_latents: true`
makes the dataloader generate random tensors of the right shape instead of
reading real data, and `a1-t` is the smallest model scale (~50M params).

```yaml
# smoke.yaml
debug: 0
transformer_architecture: a1-t
lora_rank: 0                    # 0 = full fine-tune, >0 = LoRA (e.g. 64)
vae: wan
text_encoder_type: smolLM
text_encoder_path: ""           # empty -> downloads HuggingFaceTB/SmolLM3-3B
                                 # from the HF Hub on first run (~6GB)

train_batch_size: 1
max_train_steps: 5
gradient_accumulation_steps: 1
learning_rate: 1.0e-4
lr_warmup_steps: 2

random_latents: true
random_latents_resolution: 256
max_sequence_length: 32

checkpointing_steps: 3
resume_from_checkpoint: "no"
checkpoint_local_path: "/tmp/bria_edit_smoke/ckpt"
output_dir: "/tmp/bria_edit_smoke/out"

use_fsdp: 0
use_varlen_attention: false     # flash attention -- no torch 2.10 requirement
```

```bash
MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 \
WANDB_MODE=disabled \
python train_fibo_edit_standard.py --config_path smoke.yaml
```

That's a real forward + backward pass + optimizer step + a real checkpoint
save on your GPU, just with meaningless synthetic data — good for
"does my environment/GPU/dependencies actually work" before committing to a
real run. To test resuming, rerun with `resume_from_checkpoint: "latest"`
and a higher `max_train_steps`.

`WANDB_MODE=disabled` avoids needing a real `WANDB_TOKEN` for this kind of
throwaway run — see [Troubleshooting](#troubleshooting), it doesn't fully
suppress wandb on its own.

## Real training

### 1. Data

Training reads **packed tars** of precomputed VAE latents + captions (not
raw images — there's no VAE encode step in the training loop itself). Each
pickle record inside a tar looks like:

```python
{
    "caption": "<json string: structured caption, with an 'edit_instruction' field>",
    "latents_{w}_{h}": <torch.Tensor [C, h, w]>,        # target/output latent
    "context_latents_0": <torch.Tensor [C, h0, w0]>,    # optional: 1..N ordered
    "context_latents_1": <torch.Tensor [C, h1, w1]>,    # reference/context latents
    ...
}
```

No context keys at all → a pure text-to-image sample. One `context_latents_i`
→ a single-reference edit. Multiple → multi-reference editing (arbitrary N,
each can be a different resolution).

Point `data_config` (a `"NAME:GPUS:BATCH_SIZE"` string, comma-separated for
multiple channels) and `data_paths` (`{NAME: mount_path}`) at your tar
directories:

```yaml
data_config: "MY_CHANNEL:1:1"     # 1 GPU, batch size 1, for a single channel
data_paths:
  MY_CHANNEL: /data/precomputed/sft_edit/my_channel
```

`interleave_resolutions: true` + `interleave_min_res`/`interleave_max_res`
lets one channel mix multiple `{W}x{H}` resolution buckets (`RandomMix`d
proportionally to tar count).

### 2. Model weights

```yaml
transformer_init_path: /path/to/pretrained/checkpoint   # empty = random init
text_encoder_path: /path/to/local/SmolLM3-3B             # empty = HF Hub download
transformer_architecture: alpha-t   # real training scale (a1-t..a7-t, alpha-t)
```

### 3. Full fine-tune vs. LoRA

```yaml
lora_rank: 0        # full fine-tune: every param trainable, saves the full model
# or
lora_rank: 64       # LoRA: only adapter params trainable, saves a small PEFT adapter
lora_init_weights: default   # "default" (B=zeros) or "gaussian"
```

A LoRA run resumed later needs the same `lora_rank` and `checkpoint_local_path`
— see [Resuming](#checkpointing--resuming).

### 4. Launch (single GPU, multi-GPU, multi-node)

Single GPU: run `python train_fibo_edit_standard.py --config_path <cfg>.yaml`
directly (with `MASTER_ADDR`/`MASTER_PORT`/`RANK`/`LOCAL_RANK`/`WORLD_SIZE`
set as in the quickstart above).

Multi-GPU / multi-node: use `torchrun`, and set `use_fsdp: true` in the
config (FSDP is how this model shards across GPUs — there's no plain DDP
path for the full model). `fsdp_sharding_strategy: hybrid` shards within a
node and replicates across nodes; `full` shards everywhere.

```bash
torchrun --nnodes $NUM_NODES --nproc_per_node $GPUS_PER_NODE \
  --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port 29500 \
  train_fibo_edit_standard.py --config_path <cfg>.yaml
```

`data_config`'s per-channel GPU counts must sum to the total GPU count
(`nnodes * nproc_per_node`).

### 5. Attention backend

- `use_varlen_attention: false` (default) — flash attention. Only supports
  `train_batch_size: 1` per GPU (no in-batch packing).
- `use_varlen_attention: true` — variable-length attention, required for
  `train_batch_size > 1` and for `context_drop_rate_cfg`/`both_drop_rate_cfg`
  (context CFG dropout). Needs `torch>=2.10`.

## Checkpointing & resuming

```yaml
checkpointing_steps: 1000
resume_from_checkpoint: "no"        # "no" | "latest" | an explicit checkpoint dir name
checkpoint_local_path: /ckpts/my_run
```

- `"latest"` picks the highest-numbered `checkpoint_NNNNNN` dir under
  `checkpoint_local_path`.
- A checkpoint saves model + optimizer + LR-scheduler + dataloader-sampler +
  RNG state — resuming continues `global_step` exactly, it doesn't restart.
- LoRA checkpoints are PEFT-adapter format (`adapter_model.safetensors`,
  much smaller) instead of a full model state dict.
- `use_ema: true` additionally saves `transformer_ema.bin` (full EMA
  weights, for eval) and a per-rank `ema_state_rank_N.pt` shard (for
  resuming EMA state specifically) at every checkpoint.
- `reinit_optimizer: 1` / `reinit_scheduler: 1` on a resumed run replace the
  loaded optimizer/scheduler state with a fresh one (scheduler is
  recomputed over the *remaining* steps, not the original absolute
  schedule) — use when changing the LR schedule mid-run.

## Config reference (selected fields)

| Field | Meaning |
|---|---|
| `debug` | `1` forces a bunch of fast-iteration overrides (tiny arch, `lora_rank=128`, `max_train_steps=50`, etc.) — good for a quick correctness check, **not** a substitute for choosing your own small config (it hardcodes `lora_rank=128`, so you can't get a `debug` full-fine-tune run). |
| `mixed_precision` | `bf16` (default), `fp16`, or `no`. |
| `text_drop_rate_cfg` | Probability of replacing the caption with an empty string per-sample (classifier-free-guidance text dropout). |
| `context_drop_rate_cfg` / `both_drop_rate_cfg` | Same, for context latents / both together. Requires `use_varlen_attention: true`. |
| `edit_instruction_only_prob` | Probability of feeding just the caption's `edit_instruction` field instead of the full structured-caption JSON. |
| `num_checkpointing_blocks` | Gradient checkpointing: `0` disables it, `N>0` checkpoints the first N transformer blocks (trades compute for memory). |
| `max_sequence_length` / `text_pad_length` | Caption tokenization cap / constant padding length (`-1`=dynamic, `0`=pad to `max_sequence_length`, `>0`=explicit). Constant padding avoids `torch.compile` recompilation. |
| `use_torch_compile` / `regional_compile` | `regional_compile: 1` compiles each transformer block in-place (keeps checkpoint keys stable) rather than wrapping the whole model. |

Full field list and defaults: `TrainConfig` in `train_fibo_edit_standard.py`.

## Troubleshooting

**`wandb.errors.errors.CommError: ... 401` on a throwaway/local run** —
`init_wandb()` hardcodes `mode="online"`, so `WANDB_MODE=disabled` alone
doesn't stop it from trying a real network call; it still needs a valid
`WANDB_TOKEN`. For a genuinely offline run, monkeypatch `wandb.init` (force
`mode="disabled"`) before importing the training script, or just export a
real `WANDB_TOKEN` if you have wandb access.

**`ValueError: ... environment variable MASTER_ADDR expected`** — Accelerate
needs `MASTER_ADDR`/`MASTER_PORT` set even for a single-process run outside
`torchrun`. Set `MASTER_ADDR=127.0.0.1 MASTER_PORT=<any free port>`.

**LoRA checkpoint resume fails with
`ImportError: cannot import name 'EmbeddingParallel' from transformers.integrations.tensor_parallel`**
— a `peft`/`transformers` version mismatch (seen with `peft==0.19.1`);
upgrade to `peft>=0.20.0`.

**`ModuleNotFoundError: No module named 'torch.nn.attention.varlen'`** —
your torch is older than 2.10. Only matters if `use_varlen_attention: true`;
otherwise ignore it or pin torch as in `requirements.txt`.

**`transformer_init_path` / `text_encoder_path` resolution** — both fall
back to `SM_CHANNEL_TRANSFORMER_INIT` / `SM_CHANNEL_TEXT_ENCODER` env vars
if left empty in the config (a SageMaker-channel-mount convention); on a
non-SageMaker box, either set the config field directly or set those env
vars yourself.
