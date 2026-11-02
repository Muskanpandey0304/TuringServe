"""Assembles a ready-to-run LLMEngine from an EngineConfig (startup order: ARCH 6.15)."""

from __future__ import annotations

import torch

from turingserve.config import EngineConfig
from turingserve.engine.engine import LLMEngine
from turingserve.engine.output_processor import OutputProcessor
from turingserve.kv.kv_manager import KVCacheManager
from turingserve.kv.prefix_cache import PrefixCache
from turingserve.metrics.prometheus import Metrics
from turingserve.model.loader import load_weights
from turingserve.model.qwen2 import Qwen2Model
from turingserve.model.qwen2_config import Qwen2Config
from turingserve.ops import get_ops
from turingserve.runner.model_runner import ModelRunner
from turingserve.sampling.sampler import Sampler
from turingserve.scheduler.scheduler import Scheduler
from turingserve.utils.logging import log, setup_logging


def load_tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)


def build_engine(cfg: EngineConfig, tokenizer=None) -> LLMEngine:
    setup_logging(cfg.log_level)
    mcfg = Qwen2Config.from_dir(cfg.model)
    cfg.validate(mcfg.max_position_embeddings)
    device = torch.device(cfg.device)
    dtype = cfg.torch_dtype
    cuda = device.type == "cuda"
    if cuda:
        from turingserve.runner.memory_profiler import gpu_baseline
        from turingserve.utils.gpu import check_stack

        check_stack()
        baseline, total = gpu_baseline(device)
    tokenizer = tokenizer or load_tokenizer(cfg.model)
    ops = get_ops(cfg.kernels)

    weights = load_weights(cfg.model, mcfg, device, dtype, cfg.quantization, cfg.int8_lm_head)
    model = Qwen2Model(mcfg, weights, ops, device, dtype, cfg.max_model_len)

    draft_model = dcfg = None
    if cfg.spec_method == "draft":
        dcfg = Qwen2Config.from_dir(cfg.spec_draft_model)
        if dcfg.vocab_size != mcfg.vocab_size:
            raise ValueError("--spec-draft-model must share the target's vocabulary")
        dweights = load_weights(cfg.spec_draft_model, dcfg, device, dtype)
        draft_model = Qwen2Model(dcfg, dweights, ops, device, dtype, cfg.max_model_len)

    # ---- Autopilot startup, step 1: kernel tuner (before anything is captured) ----------------
    table = None
    if cuda and cfg.kernels == "triton" and cfg.autopilot != "off":
        from turingserve.autopilot.startup import run_tuner

        table = run_tuner(cfg, mcfg)

    # ---- step 2: memory profile -> number of KV blocks -------------------------------------------
    if cfg.num_blocks is not None:
        num_blocks = cfg.num_blocks
    elif not cuda:
        num_blocks = 2048
    else:
        from turingserve.runner.memory_profiler import profile_num_blocks

        extra = dcfg.kv_bytes_per_token * 16 if dcfg else 0
        num_blocks = profile_num_blocks(cfg, mcfg, model, baseline, total, extra, device).num_blocks
    log("kv_cache", num_blocks=num_blocks, max_tokens=num_blocks * 16,
        kv_gib=round(num_blocks * mcfg.kv_bytes_per_token * 16 / 1024**3, 3))

    def alloc_kv(c):
        return torch.zeros(c.num_layers, 2, num_blocks, c.num_kv_heads, 16, c.head_dim,
                           dtype=dtype, device=device)

    kv_cache = alloc_kv(mcfg)

    # ---- step 3: CUDA graphs ------------------------------------------------------------------------
    graphs = None
    if cuda and cfg.cuda_graphs and cfg.kernels == "triton" and cfg.graph_batch_sizes:
        from turingserve.runner.cuda_graphs import CudaGraphRunner

        graphs = CudaGraphRunner(model, kv_cache, cfg.graph_batch_sizes, cfg.max_blocks_per_seq,
                                 cfg.max_model_len, device)
        graphs.capture()

    runner = ModelRunner(model, kv_cache, cfg.max_num_batched_tokens, cfg.max_num_seqs,
                         cfg.max_blocks_per_seq, device, graphs)
    kv = KVCacheManager(num_blocks)
    prefix_cache = PrefixCache(kv.allocator, enabled=cfg.prefix_caching)
    scheduler = Scheduler(kv, prefix_cache, cfg.max_num_batched_tokens, cfg.max_num_seqs,
                          cfg.watermark)
    sampler = Sampler(mcfg.eos_token_ids, cfg.seed, device)
    op = OutputProcessor(tokenizer, mcfg.eos_token_ids, cfg.max_model_len)
    engine = LLMEngine(cfg, runner, kv, prefix_cache, scheduler, sampler, op, mcfg.eos_token_ids,
                       Metrics(), device)

    # ---- speculative decoding --------------------------------------------------------------------
    if cfg.spec_method != "none":
        from turingserve.spec.spec_decoder import SpecDecoder

        if cfg.spec_method == "ngram":
            from turingserve.spec.ngram import NgramProposer

            proposer = NgramProposer(cfg.spec_ngram_max, cfg.spec_ngram_min)
        else:
            from turingserve.spec.draft import DraftProposer

            draft_runner = ModelRunner(draft_model, alloc_kv(dcfg), cfg.max_num_batched_tokens,
                                       max(cfg.spec_max_batch, 1), cfg.max_blocks_per_seq, device)
            proposer = DraftProposer(draft_runner)
        engine.spec = SpecDecoder(engine, proposer, cfg.spec_k, cfg.spec_k, cfg.spec_max_batch,
                                  cfg.spec_method)

    # ---- step 4: chunk-size sweep, then the runtime controller ---------------------------------------
    if table is not None:
        from turingserve.autopilot.startup import chunk_sweep

        chunk_sweep(engine, cfg, table)
    if cfg.autopilot == "full":
        from turingserve.autopilot.controller import Autopilot, Limits

        engine.autopilot = Autopilot(engine, Limits(
            ttft_target_s=cfg.ttft_target_ms / 1000, tpot_target_s=cfg.tpot_target_ms / 1000,
            max_chunk=cfg.max_num_batched_tokens, max_num_seqs=cfg.max_num_seqs,
            spec_k_max=cfg.spec_k, spec_max_batch=cfg.spec_max_batch,
            spec_enabled=cfg.spec_method != "none"))
    log("engine_ready", model=cfg.served_model_name, device=str(device), kernels=cfg.kernels,
        dtype=cfg.dtype, quantization=cfg.quantization, spec=cfg.spec_method,
        autopilot=cfg.autopilot)
    return engine
