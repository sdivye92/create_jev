# Cache reuse for Qwen3.5-0.8B: findings

## Result
Cache reuse is correct. There is no cache, position or DeltaNet bug.

- Recipe: `core(state, use_cache=True)`, then `deepcopy(cache)` per question, then `core(question_only_ids, past_key_values=clone, use_cache=True)`. `core` is `model.model`. No `position_ids` or `cache_position` are needed.
- fp32: max-abs-diff vs the naive baseline is 2.1e-5 (Q_A) and 1.4e-5 (Q_B). `allclose(atol=1e-2)` passes.
- bf16: max-abs-diff is 0.375 (Q_A) and 0.25 (Q_B). `allclose(atol=1e-2)` fails, and it cannot pass at this tolerance.

## Why bf16 can't hit 1e-2
The last-token final hidden state has entries up to about 30. One bf16 ULP at magnitude 16-32 is 2^-3 = 0.125, and at 32-64 it is 0.25, so 0.25-0.375 is 1-3 ULP. A 1e-2 absolute tolerance is below the bf16 resolution of those entries.

The two paths are mathematically equivalent but not bitwise equal:
- The baseline runs DeltaNet as one chunk over all tokens. The continuation uses a different chunk layout with an fp32 recurrent state carried over from the prefix.
- The matmul shapes differ (23 + q tokens vs q tokens). So the rounding differs.
- Layers 0-22 add this rounding noise up (18 DeltaNet layers).

Evidence (`experiments/investigations/cache_reuse/bf16_noise_check.py`), with an fp32 run as ground truth:

| | max-abs-diff vs fp32 (Q_A / Q_B) |
|---|---|
| bf16 baseline | 0.242 / 0.203 |
| bf16 cache-reuse | 0.231 / 0.163 |
| bf16 baseline vs bf16 cache-reuse | 0.375 / 0.250 |

The bf16 baseline is about 0.2 away from the true value itself. The cache-reuse path is no further from the truth than the baseline is. Cosine(base, cache) = 0.9999 and relative L2 = 1.4%.

## What was checked and ruled out
- The position offset is derived internally from `past_key_values.get_seq_length()` (23), so the continuation gets positions 23..23+q-1, the same as the baseline. This is why an explicit `position_ids` made no difference: it was never needed.
- Since fp32 matches to 2e-5 through the same code, the earlier "layer 0 diverges" observation was bf16 rounding showing up early and compounding. It was not a logic error.
- Baseline determinism is exactly 0.0 (a repeat of an identical call), so it is a stable reference. The gap is a systematic difference between two rounding paths.

## If you need a bf16 pass
Pick a tolerance that matches bf16, or compare in a scale-aware way:
- `atol` around 0.5 on raw states, or `rtol`/cosine (cos > 0.999, relative L2 < 2%).
- Or run the DeltaNet layers and norms in fp32 and keep the rest in bf16.
- Or use the fp32 proof in `cache_reuse_debug.py`.

I did not try making the final hidden state bitwise-identical in bf16. The chunking of the reference DeltaNet kernel would have to match.

## Files
- `cache_reuse_debug.py`: fp32 proof (asserts), plus bf16 shown for context.
- `bf16_noise_check.py`: the bf16-vs-fp32 ground-truth comparison.
