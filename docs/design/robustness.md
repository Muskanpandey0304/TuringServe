# Robustness (ARCH 6.17)
CUDA OOM -> preempt everything, empty_cache, halve the chunk (3 in 60 s = unhealthy 503). NaN/inf logits -> that request
finishes with "error", never sampled. An engine-thread exception fails all requests and /health returns 503.
Finished/aborted requests are removed from every dict; the soak test checks `num_requests_tracked` returns to 0.
