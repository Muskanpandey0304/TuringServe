# Scheduler (turingserve/scheduler/scheduler.py)
Each step builds one flat token batch under a token budget (`max_num_batched_tokens`).
Pass 1 schedules running decodes (1 token each); if a decode cannot get a block, the NEWEST running
sequence not already in the plan is preempted (recompute). Pass 2 continues running chunked prefills and
never preempts. Pass 3 admits waiting sequences FCFS, only if nothing was preempted this step, keeping a
watermark of free blocks. A chunk "samples" only when it reaches the end of `token_ids`; decode is just n = 1.
Why recompute, not swap: generated tokens stay in `token_ids`, the prefix cache makes the redo cheap, and
PCIe swapping is slow. Invariants are property-tested with `FakeModelRunner` (exact outputs).
