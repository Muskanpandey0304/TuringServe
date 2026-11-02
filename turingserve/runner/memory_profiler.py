"""KV-cache sizing (ARCH 6.10.3). Run once at startup, after the weights are loaded."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from turingserve.runner.attention_metadata import AttentionMetadata, ModelInput
from turingserve.utils.gpu import GiB
from turingserve.utils.logging import log


@dataclass
class MemoryPlan:
    num_blocks: int
    baseline: int = 0
    peak: int = 0
    non_torch: int = 0
    graph_reserve: int = 0
    kv_budget: int = 0
    bytes_per_block: int = 0


def gpu_baseline(device) -> tuple[int, int]:
    """(bytes used by other processes + our CUDA context, total bytes). Call BEFORE loading."""
    free0, total = torch.cuda.mem_get_info(device)
    return total - free0, total


def dummy_prefill(model, sampler_rows: int, num_tokens: int, device) -> None:
    """One worst-case step: a prefill of ``num_tokens`` tokens, logits for ``sampler_rows`` rows."""
    nb = -(-num_tokens // 16) + 1
    kv = torch.zeros(model.cfg.num_layers, 2, nb, model.cfg.num_kv_heads, 16, model.cfg.head_dim,
                     dtype=model.dtype, device=device)
    T = num_tokens
    meta = AttentionMetadata(
        query_start_loc=torch.tensor([0, T], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([T], dtype=torch.int32, device=device),
        block_tables=torch.arange(1, nb, dtype=torch.int32, device=device)[None, :],
        slot_mapping=16 + torch.arange(T, dtype=torch.int64, device=device),
        num_decode=0, max_query_len=T, max_seq_len=T)
    inp = ModelInput(torch.zeros(T, dtype=torch.int64, device=device),
                     torch.arange(T, dtype=torch.int64, device=device), meta,
                     torch.arange(T - sampler_rows, T, dtype=torch.int64, device=device))
    hidden = model.forward(inp.input_ids, inp.positions, inp.meta, kv)
    logits = model.compute_logits(hidden, inp.sample_indices)
    probs = torch.softmax(logits, dim=-1)                    # the sampler's working set
    probs.sort(dim=-1, descending=True)
    torch.cuda.synchronize()


def profile_num_blocks(cfg, mcfg, model, baseline: int, total: int, extra_bytes_per_block: int = 0,
                       device="cuda") -> MemoryPlan:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    T = min(cfg.max_num_batched_tokens, cfg.max_model_len)
    dummy_prefill(model, min(cfg.max_num_seqs, T), T, device)
    peak = torch.cuda.max_memory_allocated()
    torch.cuda.empty_cache()
    free1, _ = torch.cuda.mem_get_info(device)
    non_torch = max(0, (total - free1) - baseline - torch.cuda.memory_reserved())
    graph_reserve = int(min(256 * 1024**2, 0.04 * total)) if cfg.cuda_graphs else 0
    kv_budget = int(total * cfg.gpu_memory_utilization - baseline - peak - non_torch - graph_reserve)
    bytes_per_block = mcfg.kv_bytes_per_token * 16 + extra_bytes_per_block
    num_blocks = max(0, kv_budget // bytes_per_block)
    plan = MemoryPlan(num_blocks, baseline, peak, non_torch, graph_reserve, kv_budget, bytes_per_block)
    log("memory_profile", baseline_gib=round(baseline / GiB, 3), total_gib=round(total / GiB, 3),
        weights_and_peak_gib=round(peak / GiB, 3), non_torch_gib=round(non_torch / GiB, 3),
        graph_reserve_gib=round(graph_reserve / GiB, 3), kv_budget_gib=round(kv_budget / GiB, 3),
        num_blocks=num_blocks, max_tokens=num_blocks * 16)
    if num_blocks - 1 < math.ceil(cfg.max_model_len / 16):
        raise RuntimeError(
            "Not enough GPU memory for one full-length sequence. Free GPU memory (is Ollama "
            "running?), lower --max-model-len, or use --quantization int8.")
    return plan
