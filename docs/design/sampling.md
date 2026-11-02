# Sampling and streaming text (turingserve/sampling, turingserve/engine)
Exponential race: token = argmax(p / E) with E ~ Exp(1) samples i with probability p_i; a seeded request draws E from
its own generator, so its stream does not depend on who else is in the batch. Speculative acceptance uses the same
processed distributions for p and q: accept with min(1, p/q); on rejection sample normalize(max(0, p - q)); an n-gram
proposal is a point mass (accept with p(d), else sample p without d). DecodeStream holds back incomplete UTF-8; stop
strings are searched only in the window that can contain a new match, and `max_stop_len - 1` characters are held back.
