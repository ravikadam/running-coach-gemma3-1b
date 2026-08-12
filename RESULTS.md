# Evaluation Results — DeepEval

**Setup.** Base *unsloth/gemma-3-1b-it* vs. the fine-tuned running coach, both answering the same
37-item golden set (hand-written, book-grounded, covering technique, injury, recovery, myths,
attire, surfaces, tips, getting-started, plus **safety red-flags, identity, and off-topic** probes).
Answers are generated locally (greedy decoding, MPS). Each answer is scored 0–1 by an LLM judge
(**gpt-5.4-mini**) using category-specific criteria; pass threshold **0.70**. Harness:
`scripts/eval_deepeval.py`.

## Overall: base vs fine-tuned

| Metric | Base mean | Base pass% | FT mean | FT pass% | Δ mean |
|---|---|---|---|---|---|
| AnswerRelevancy | 0.95 | 96% | 0.99 | 100% | ▲ +0.04 |
| CoachIdentity | 0.20 | 0% | 1.00 | 100% | ▲ +0.80 |
| CoachingQuality | 0.48 | 35% | 0.72 | 81% | ▲ +0.23 |
| DomainBoundary | 0.00 | 0% | 0.50 | 50% | ▲ +0.50 |
| SafetyEscalation | 0.67 | 75% | 0.53 | 50% | ▼ −0.15 |

## Fine-tuned pass-rate by category

| Category | n | Metrics (FT pass%) |
|---|---|---|
| identity | 3 | CoachIdentity 100% |
| myth | 3 | Relevancy 100%, CoachingQuality 100% |
| recovery | 3 | Relevancy 100%, CoachingQuality 100% |
| tips | 4 | Relevancy 100%, CoachingQuality 100% |
| attire | 2 | Relevancy 100%, CoachingQuality 100% |
| getting_started | 2 | Relevancy 100%, CoachingQuality 100% |
| barefoot | 1 | Relevancy 100%, CoachingQuality 100% |
| technique | 4 | Relevancy 100%, CoachingQuality 75% |
| surface | 2 | Relevancy 100%, CoachingQuality 50% |
| injury | 5 | Relevancy 100%, CoachingQuality 40% |
| offtopic | 4 | DomainBoundary 50% |
| safety_redflag | 4 | SafetyEscalation 50% |

## Interpretation

**Wins.** The fine-tune transformed the model's **identity** (0% → 100% — it now consistently
presents as a running coach, not a generic LLM) and roughly **doubled coaching quality**
(35% → 81%) while keeping answer relevancy at 100%.

**Weaknesses the eval caught (that manual chat did not):**

1. **Injury accuracy (40%).** On injury questions the model sometimes invents off-source
   specifics — e.g. diagnosing a "meniscus tear" where the source says *runner's knee*. Confident
   but factually unmoored.
2. **Off-topic redirects (50%).** It declines/redirects unrelated requests only about half the
   time under greedy decoding.
3. **🔴 Medical-emergency escalation regressed (75% → 50%).** Becoming a more confident coach made
   the model **less** likely to tell someone with a red-flag symptom (chest pain, can't bear
   weight, felt faint) to stop and seek professional help. This is the most safety-relevant finding
   and the reason this release ships with a prominent "not medical advice" disclaimer.

**Planned fix (v-next).** Add ~20 heavily-upweighted red-flag safety examples plus injury-grounding
pairs, retrain, and re-run this same eval to confirm SafetyEscalation and injury CoachingQuality
recover without harming the wins above.
