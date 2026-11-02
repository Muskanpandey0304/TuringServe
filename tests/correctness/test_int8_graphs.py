import pytest
import torch

from tests.correctness.helpers import load_ours, rand_ids


def test_int8_reference_path_agrees_with_fp32_on_tiny(tiny_model_dir):
    """Teacher-forced top-1 agreement of the dequantizing reference path with the fp model."""
    fp, _ = load_ours(tiny_model_dir)
    q8, _ = load_ours(tiny_model_dir, quant="int8")
    assert q8.w.layers[0].qkv.quantized and q8.w.embed.dtype == torch.float32   # embed stays fp
    agree = total = 0
    for seed in range(4):
        ids = rand_ids(64, seed=seed)
        with torch.inference_mode():
            a = fp.compute_logits(fp.forward_dense(torch.tensor(ids), torch.arange(64)),
                                  torch.arange(64))
            b = q8.compute_logits(q8.forward_dense(torch.tensor(ids), torch.arange(64)),
                                  torch.arange(64))
        agree += (a.argmax(-1) == b.argmax(-1)).sum().item()
        total += 64
    assert agree / total >= 0.80        # tiny random model: spread logits but 2 layers of noise


@pytest.mark.gpu
@pytest.mark.parametrize("n", [1, 3, 8, 31])
def test_cuda_graph_equals_eager(n, tiny_model_dir):
    from turingserve.engine.sampling_params import SamplingParams
    from turingserve.engine.sequence import Sequence
    from turingserve.kv.kv_manager import KVCacheManager
    from turingserve.model.loader import load_weights
    from turingserve.model.qwen2 import Qwen2Model
    from turingserve.model.qwen2_config import Qwen2Config
    from turingserve.ops import get_ops
    from turingserve.runner.cuda_graphs import CudaGraphRunner
    from turingserve.runner.input_builder import BatchItem, InputBuilder

    cfg = Qwen2Config.from_dir(tiny_model_dir)
    dev = torch.device("cuda")
    model = Qwen2Model(cfg, load_weights(tiny_model_dir, cfg, dev, torch.float16), get_ops("triton"),
                       dev, torch.float16, 512)
    nb = 400
    kv = torch.zeros(cfg.num_layers, 2, nb, cfg.num_kv_heads, 16, cfg.head_dim, dtype=torch.float16,
                     device=dev)
    graphs = CudaGraphRunner(model, kv, (1, 2, 4, 8, 16, 32), 32, 512, dev)
    graphs.capture()
    mgr = KVCacheManager(nb)
    items = []
    for i in range(n):
        L = 5 + 17 * i
        s = Sequence(str(i), rand_ids(L, seed=i), L, SamplingParams())
        mgr.allocate(s, L)
        items.append(BatchItem(s, L - 1, [s.token_ids[-1]], "last"))
    inp = InputBuilder(256, 64, 32, dev).build(items)
    with torch.inference_mode():
        eager = model.compute_logits(model.forward(inp.input_ids, inp.positions, inp.meta, kv),
                                     inp.sample_indices)
        graph = graphs.replay(inp, n).clone()
    assert torch.equal(eager, graph)
