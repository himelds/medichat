# Evaluation Methodology Notes

## Judge LLM: Groq Llama 3.3 70B

**Choice rationale:**
- Different model family from generator (Gemma 4 31B via OpenRouter) → reduces self-evaluation bias
- LPU infrastructure provides ~2s latency per call vs ~30s on alternative free providers
- 14,400 RPD free tier sufficient for full evaluation including Phase 3 ablations (~2,400 calls)

## `answer_relevancy.strictness = 1`

**Default behavior:** RAGAS uses `strictness=3` for answer_relevancy, which requests 3 synthetic
question generations per evaluation call to reduce variance via averaging.

**Constraint:** Groq API enforces `n=1` (single generation per request), incompatible with default.

**Mitigation:** Set `strictness=1` for all evaluation runs (baseline, hybrid, rerank, query-rewrite).

**Impact:**
- Per-sample noise: increased by factor of √3 ≈ 1.73x
- Aggregate noise across 50 samples: ~0.03 absolute score increase
- **Improvement deltas in ablation study unaffected** — same configuration applied to all variants

**Verification:** Faithfulness, context_precision, context_recall use n=1 by default
(no impact from this constraint).
