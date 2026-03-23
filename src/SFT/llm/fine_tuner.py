#!/usr/bin/env python3
"""
Fine-tuner with PyTorch FSDP for multi-GPU distributed training.

Single-GPU:  python fine_tuner.py
Multi-GPU:   torchrun --nproc_per_node=NUM_GPUS fine_tuner.py
"""

import os
import time
import functools
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

# ── config ───────────────────────────────────────────────────────────────────
MODEL_PATH = "/kaggle/input/models/egregori/qwen2-5-0-5b/pytorch/default/1/qwen2.5-0.5b"
DATA_PATH = "/kaggle/input/datasets/egregori/little-town-log-dataset/llm_2p.jsonl"
OUTPUT_DIR = "/kaggle/working/checkpoints/llm-qwen/v1"
MAX_SEQ_LEN      = 512 + 128

NUM_EPOCHS       = 3
LR               = 2e-4
WARMUP_RATIO     = 0.05
PER_DEVICE_BATCH = 1
GRAD_ACCUM_STEPS = 8  # effective batch = PER_DEVICE_BATCH * GRAD_ACCUM_STEPS * world_size

# SHARD_GRAD_OP: shards gradients + optimizer states; keeps full weights on each GPU.
# Chosen because Qwen2.5-1.5B weights (~3 GB in bf16) fit comfortably on a single GPU.
# Switch to FULL_SHARD for 7B+ models where weights alone exceed single-GPU memory.
SHARDING_STRATEGY = ShardingStrategy.SHARD_GRAD_OP

# ── timing helpers ────────────────────────────────────────────────────────────
def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


class Timer:
    """Wall-clock timer for measuring phase durations."""
    def __init__(self):
        self._t = time.time()

    def lap(self) -> float:
        """Returns seconds since last lap (or creation) and resets the clock."""
        now = time.time()
        elapsed = now - self._t
        self._t = now
        return elapsed

# ── distributed helpers ───────────────────────────────────────────────────────
def dist_setup() -> int:
    """Initialize NCCL process group. Returns local_rank (0 for single-GPU)."""
    if "LOCAL_RANK" not in os.environ:
        return 0
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0

def world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1

# ── mixed precision ───────────────────────────────────────────────────────────
def make_mp_policy() -> MixedPrecision:
    """
    bf16 on modern GPUs (A100/H100): same exponent range as fp32, rarely overflows.
    fp16 fallback for older GPUs (V100/consumer): more fragile, requires loss scaling.
    """
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if is_main():
        print(f"Mixed precision dtype: {'bf16' if dtype == torch.bfloat16 else 'fp16'}")
    return MixedPrecision(
        param_dtype=dtype,   # store and communicate weights in half-precision
        reduce_dtype=dtype,  # all-reduce gradients across GPUs in half-precision
        buffer_dtype=dtype,  # model buffers (e.g. BN running stats) in half-precision
    )

# ── model ─────────────────────────────────────────────────────────────────────
def build_model() -> torch.nn.Module:
    """
    Load model in bf16. 4-bit BitsAndBytes quantization is intentionally omitted:
    quantized tensors cannot be sharded across GPUs, making it incompatible with FSDP.
    """
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        use_cache=False,  # must be False when gradient_checkpointing is enabled
    )

    # Gradient checkpointing: discard intermediate activations during the forward
    # pass and recompute them during backward. ~30% slower per step but ~60% less
    # activation memory — what makes training large models feasible.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # LoRA: freeze base weights, only train small adapter matrices (~0.3% of params)
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    # LoRA adapters are initialised in float32 by PEFT; cast everything to bf16
    # so FSDP sees a uniform dtype when it validates tensors before wrapping.
    model = model.to(torch.bfloat16)
    if is_main():
        model.print_trainable_parameters()
    return model


def wrap_fsdp(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """
    Wrap with FSDP for multi-GPU training; moves to device for single-GPU.

    use_orig_params=True is required for LoRA + FSDP correctness.
    Without it, FSDP flattens all parameters into contiguous shards and loses
    the distinction between trainable LoRA adapters and frozen base weights,
    causing the optimizer to allocate momentum/variance states for frozen
    parameters and completely negating LoRA's memory advantage.
    """
    if world_size() == 1:
        return model.to(device)

    # Wrap each transformer decoder layer independently so FSDP gathers and
    # discards weights one layer at a time, minimising peak memory per GPU.
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen2DecoderLayer},
    )

    return FSDP(
        model,
        sharding_strategy=SHARDING_STRATEGY,
        mixed_precision=make_mp_policy(),
        auto_wrap_policy=wrap_policy,
        device_id=device,       # move shards to this GPU after wrapping
        use_orig_params=True,   # preserve LoRA / frozen param distinction
    )

# ── data ──────────────────────────────────────────────────────────────────────
def build_dataloader(tokenizer, rank: int, ws: int) -> DataLoader:
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    sampler = (
        DistributedSampler(dataset, num_replicas=ws, rank=rank, shuffle=True)
        if ws > 1 else None
    )

    def collate(batch):
        texts = [
            tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            )
            for ex in batch
        ]
        enc = tokenizer(
            texts, padding=True, truncation=True,
            max_length=MAX_SEQ_LEN, return_tensors="pt",
        )
        enc["labels"] = enc["input_ids"].clone()
        enc["labels"][enc["attention_mask"] == 0] = -100  # mask padding from loss
        return enc

    return DataLoader(
        dataset,
        batch_size=PER_DEVICE_BATCH,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collate,
        pin_memory=True,
    )

# ── training loop ─────────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, scheduler, device, epoch: int):
    """
    One full training epoch with gradient accumulation.

    Gradient accumulation: instead of requiring batch_size examples in memory
    at once, we do GRAD_ACCUM_STEPS micro-steps, each adding to the gradient,
    then take a single optimizer step. The model sees the same total data per
    update without ever materialising the full batch in memory.
    """
    model.train()
    optimizer.zero_grad()
    running_loss = 0.0
    total_tokens = 0
    t0 = time.time()

    for step, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        # Scale loss so accumulated gradients match a full-batch backward pass
        loss = outputs.loss / GRAD_ACCUM_STEPS
        loss.backward()
        running_loss += outputs.loss.item()
        # Count non-padding tokens processed across all GPUs this step
        total_tokens += int(batch["attention_mask"].sum().item()) * world_size()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if is_main() and (step + 1) % 10 == 0:
            elapsed = time.time() - t0
            ms_per_step = elapsed / (step + 1) * 1000
            tok_per_s = total_tokens / elapsed
            eta_s = (len(loader) - step - 1) * (elapsed / (step + 1))
            print(
                f"  epoch {epoch}  step {step + 1}/{len(loader)}"
                f"  loss={running_loss / (step + 1):.4f}"
                f"  tok/s={tok_per_s:.0f}"
                f"  ms/step={ms_per_step:.0f}"
                f"  eta={fmt_time(eta_s)}"
                f"  lr={scheduler.get_last_lr()[0]:.2e}"
            )

    elapsed_total = time.time() - t0
    return running_loss / len(loader), elapsed_total / 60, total_tokens / elapsed_total

# ── save ──────────────────────────────────────────────────────────────────────
def save_adapter(model, tokenizer):
    """
    Save adapter weights. For FSDP, all ranks participate in the state_dict
    gather (each contributes its local shards); only rank 0 writes to disk.
    offload_to_cpu=True prevents OOM on rank 0 during the gather.
    """
    os.makedirs(OUTPUT_DIR + "/final-adapter", exist_ok=True)

    if isinstance(model, FSDP):
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            cpu_sd = model.state_dict()
        if is_main():
            torch.save(cpu_sd, OUTPUT_DIR + "/final-adapter/adapter_model.pt")
    else:
        model.save_pretrained(OUTPUT_DIR + "/final-adapter")

    if is_main():
        tokenizer.save_pretrained(OUTPUT_DIR + "/final-adapter")
        print("Adapter saved to", OUTPUT_DIR + "/final-adapter")

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    wall_t = time.time()
    local_rank = dist_setup()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    rank = dist.get_rank() if dist.is_initialized() else 0

    if is_main():
        print(f"Training on {world_size()} GPU(s)  |  sharding={SHARDING_STRATEGY.name}")

    t = Timer()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = build_model()
    if is_main():
        print(f"Model loaded in {fmt_time(t.lap())}")

    model = wrap_fsdp(model, device)
    if is_main() and world_size() > 1:
        print(f"FSDP wrap in {fmt_time(t.lap())}")
    else:
        t.lap()  # consume so next lap is accurate

    loader = build_dataloader(tokenizer, rank, world_size())
    if is_main():
        print(f"Dataset loaded  ({len(loader.dataset)} examples)  in {fmt_time(t.lap())}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    total_steps = (len(loader) // GRAD_ACCUM_STEPS) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    optimizer = AdamW(trainable, lr=LR, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if is_main():
        print(
            f"Steps/epoch: {len(loader)}  |  "
            f"Optimizer steps total: {total_steps}  |  "
            f"Warmup steps: {warmup_steps}"
        )
        print("─" * 70)

    stats = []
    train_t = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
        avg_loss, elapsed_min, avg_tok_s = run_epoch(
            model, loader, optimizer, scheduler, device, epoch
        )
        stats.append((epoch, avg_loss, elapsed_min, avg_tok_s))
        if is_main():
            gpu_mb = torch.cuda.max_memory_allocated(device) / 1e6
            print(
                f"Epoch {epoch} done — "
                f"loss={avg_loss:.4f}  "
                f"time={elapsed_min:.1f}m  "
                f"tok/s={avg_tok_s:.0f}  "
                f"peak_mem={gpu_mb:.0f} MB"
            )

    total_train_s = time.time() - train_t
    total_wall_s  = time.time() - wall_t

    if is_main():
        print("\n── Training summary ─────────────────────────────────────────────")
        print(f"  {'Epoch':<8} {'Loss':<10} {'Time (m)':<12} {'Tok/s':<10}")
        print(f"  {'─'*8} {'─'*9} {'─'*11} {'─'*9}")
        for ep, loss, mins, tok_s in stats:
            print(f"  {ep:<8} {loss:<10.4f} {mins:<12.1f} {tok_s:<10.0f}")
        print()
        print(f"  Total training time : {fmt_time(total_train_s)}")
        print(f"  Total wall time     : {fmt_time(total_wall_s)}")
        peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
        print(f"  Peak GPU memory     : {peak_mb:.0f} MB")
        print(f"  GPUs                : {world_size()}  |  sharding: {SHARDING_STRATEGY.name}")
        print("─" * 70)

    save_adapter(model, tokenizer)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
