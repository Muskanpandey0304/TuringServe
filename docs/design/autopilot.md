# Autopilot (turingserve/autopilot)
Startup: measure kernel configs per shape bucket (own tuner, JSON cache keyed by GPU/model/dtype/lengths/versions),
profile memory -> KV blocks, capture CUDA graphs, sweep the chunk size. Runtime: `decide(state, signals)` is pure and
called every 2 s over a 10 s window; a rule must hold for 3 consecutive windows and then cools down for 10 s.
R1 lowers the chunk when TPOT p90 > target; R2 raises it only if TPOT p90 < 0.7 x target (hysteresis band).
R0 OOM halves the chunk; R3-R5 govern speculation; R6/R7 move max_num_seqs. Rules over RL: explainable, testable, stable.
