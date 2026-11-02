"""EngineConfig: every flag of ARCH 4 in one dataclass, plus CLI parsing and validation."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, fields


@dataclass
class EngineConfig:
    model: str = ""
    served_model_name: str | None = None
    dtype: str = "float16"
    device: str = "cuda"
    kernels: str = "triton"
    max_model_len: int = 4096
    block_size: int = 16
    num_blocks: int | None = None
    gpu_memory_utilization: float = 0.85
    max_num_seqs: int = 128
    max_num_batched_tokens: int = 2048
    max_waiting_requests: int = 256
    watermark: float = 0.01
    cuda_graphs: bool = True
    graph_batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    prefix_caching: bool = True
    quantization: str = "none"
    int8_lm_head: bool = False
    spec_method: str = "none"
    spec_k: int = 4
    spec_ngram_max: int = 3
    spec_ngram_min: int = 1
    spec_draft_model: str | None = None
    spec_max_batch: int = 8
    autopilot: str = "startup"
    ttft_target_ms: float = 2000.0
    tpot_target_ms: float = 50.0
    tuning_dir: str = "~/.turingserve/tuning"
    retune: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    log_requests: bool = False
    debug_endpoints: bool = False
    profile_dir: str | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.served_model_name is None and self.model:
            self.served_model_name = os.path.basename(os.path.normpath(self.model))
        if self.device == "cpu":
            self.cuda_graphs = False
            self.autopilot = "off" if self.autopilot == "startup" else self.autopilot

    @property
    def torch_dtype(self):
        import torch

        return {"float16": torch.float16, "float32": torch.float32}[self.dtype]

    def validate(self, max_position_embeddings: int | None = None) -> None:
        def bad(msg: str):
            raise ValueError(msg)

        if not self.model:
            bad("--model is required")
        if self.dtype not in ("float16", "float32"):
            bad("--dtype must be float16 or float32")
        if self.device not in ("cuda", "cpu"):
            bad("--device must be cuda or cpu")
        if self.kernels not in ("triton", "reference"):
            bad("--kernels must be triton or reference")
        if self.block_size != 16:
            bad("--block-size must be 16 (the kernels assume it)")
        if max_position_embeddings is not None and self.max_model_len > max_position_embeddings:
            bad(f"--max-model-len {self.max_model_len} exceeds the model's "
                f"max_position_embeddings {max_position_embeddings}")
        if self.max_model_len > 32768:
            bad("--max-model-len must be <= 32768")
        if self.max_num_batched_tokens < 16:
            bad("--max-num-batched-tokens must be >= 16")
        if self.device == "cpu" and self.kernels != "reference":
            bad("--device cpu requires --kernels reference")
        if self.kernels == "triton" and self.device != "cuda":
            bad("--kernels triton needs CUDA")
        if self.quantization not in ("none", "int8"):
            bad("--quantization must be none or int8")
        if self.quantization == "int8" and self.kernels != "triton" and self.device == "cuda":
            bad("--quantization int8 needs --kernels triton "
                "(the reference path dequantizes; allowed for tests via --kernels reference)")
        if self.spec_method not in ("none", "ngram", "draft"):
            bad("--spec-method must be none, ngram or draft")
        if self.spec_method == "draft" and not self.spec_draft_model:
            bad("--spec-method draft needs --spec-draft-model")
        if not 1 <= self.spec_k <= 8:
            bad("--spec-k must be in 1..8")
        if self.autopilot not in ("off", "startup", "full"):
            bad("--autopilot must be off, startup or full")
        self.graph_batch_sizes = tuple(s for s in self.graph_batch_sizes if s <= self.max_num_seqs)

    @property
    def max_blocks_per_seq(self) -> int:
        return -(-self.max_model_len // self.block_size)


def _bool_pair(p: argparse.ArgumentParser, name: str, default: bool, dest: str, help_: str):
    g = p.add_mutually_exclusive_group()
    g.add_argument(f"--{name}", dest=dest, action="store_true", help=help_)
    g.add_argument(f"--no-{name}", dest=dest, action="store_false")
    p.set_defaults(**{dest: default})


def build_parser() -> argparse.ArgumentParser:
    d = EngineConfig()
    p = argparse.ArgumentParser(prog="turingserve")
    sub = p.add_subparsers(dest="command", required=True)
    for cmd in ("serve", "tune"):
        s = sub.add_parser(cmd)
        s.add_argument("--model", required=True)
        s.add_argument("--served-model-name", default=None)
        s.add_argument("--dtype", default=d.dtype, choices=["float16", "float32"])
        s.add_argument("--device", default=d.device, choices=["cuda", "cpu"])
        s.add_argument("--kernels", default=None, choices=["triton", "reference"])
        s.add_argument("--max-model-len", type=int, default=d.max_model_len)
        s.add_argument("--block-size", type=int, default=d.block_size)
        s.add_argument("--num-blocks", type=int, default=None)
        s.add_argument("--gpu-memory-utilization", type=float, default=d.gpu_memory_utilization)
        s.add_argument("--max-num-seqs", type=int, default=d.max_num_seqs)
        s.add_argument("--max-num-batched-tokens", type=int, default=d.max_num_batched_tokens)
        s.add_argument("--max-waiting-requests", type=int, default=d.max_waiting_requests)
        s.add_argument("--watermark", type=float, default=d.watermark)
        _bool_pair(s, "cuda-graphs", True, "cuda_graphs", "graph replay for pure-decode steps")
        s.add_argument("--graph-batch-sizes", default=",".join(map(str, d.graph_batch_sizes)))
        _bool_pair(s, "prefix-caching", True, "prefix_caching", "reuse KV blocks of identical prefixes")
        s.add_argument("--quantization", default=d.quantization, choices=["none", "int8"])
        s.add_argument("--int8-lm-head", action="store_true")
        s.add_argument("--spec-method", default=d.spec_method, choices=["none", "ngram", "draft"])
        s.add_argument("--spec-k", type=int, default=d.spec_k)
        s.add_argument("--spec-ngram-max", type=int, default=d.spec_ngram_max)
        s.add_argument("--spec-ngram-min", type=int, default=d.spec_ngram_min)
        s.add_argument("--spec-draft-model", default=None)
        s.add_argument("--spec-max-batch", type=int, default=d.spec_max_batch)
        s.add_argument("--autopilot", default=d.autopilot, choices=["off", "startup", "full"])
        s.add_argument("--ttft-target-ms", type=float, default=d.ttft_target_ms)
        s.add_argument("--tpot-target-ms", type=float, default=d.tpot_target_ms)
        s.add_argument("--tuning-dir", default=d.tuning_dir)
        s.add_argument("--retune", action="store_true")
        s.add_argument("--host", default=d.host)
        s.add_argument("--port", type=int, default=d.port)
        s.add_argument("--log-level", default=d.log_level, choices=["debug", "info", "warning"])
        s.add_argument("--log-requests", action="store_true")
        s.add_argument("--debug-endpoints", action="store_true")
        s.add_argument("--profile-dir", default=None)
        s.add_argument("--seed", type=int, default=d.seed)
    return p


def parse_args(argv: list[str] | None = None) -> tuple[str, EngineConfig]:
    ns = build_parser().parse_args(argv)
    kw = {f.name: getattr(ns, f.name) for f in fields(EngineConfig) if hasattr(ns, f.name)}
    kw["graph_batch_sizes"] = tuple(int(x) for x in ns.graph_batch_sizes.split(",") if x)
    if kw.get("kernels") is None:
        kw["kernels"] = "reference" if ns.device == "cpu" else "triton"
    cfg = EngineConfig(**kw)
    return ns.command, cfg
