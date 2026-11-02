"""``turingserve serve|tune`` entry point."""

from __future__ import annotations

import sys

from turingserve.config import parse_args


def main(argv: list[str] | None = None) -> int:
    command, cfg = parse_args(argv)
    if command == "serve":
        import uvicorn

        from turingserve.api.server import create_app

        uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, workers=1,
                    log_level="warning" if cfg.log_level == "warning" else "info")
        return 0
    if command == "tune":
        return tune(cfg)
    return 2


def tune(cfg) -> int:
    """Run only the kernel tuner (Phase 7) and print a table of winners."""
    from turingserve.autopilot.startup import run_tuner
    from turingserve.model.qwen2_config import Qwen2Config
    from turingserve.utils.logging import setup_logging

    setup_logging(cfg.log_level)
    mcfg = Qwen2Config.from_dir(cfg.model)
    cfg.validate(mcfg.max_position_embeddings)
    if cfg.device != "cuda":
        print("tuning needs a CUDA GPU", file=sys.stderr)
        return 1
    table = run_tuner(cfg, mcfg)
    print(f"{'kernel':<12}{'bucket':<20}config")
    for kernel, buckets in sorted(table.tables.items()):
        for bucket, c in sorted(buckets.items()):
            print(f"{kernel:<12}{bucket:<20}{c}")
    if table.m_cross:
        print("M_cross:", table.m_cross)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
