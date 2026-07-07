# Design Document: Persistence Gate

- **Status: Under consideration (implementation on hold)** — building an evaluation foundation to judge this method comes first (§7)
- Date: 2026-07-07
- Scope: adding a new phase to the MetricSifter core pipeline to improve reduction performance

## 1. Background and Motivation

The current pipeline (simple filter → per-metric univariate CPD → KDE segmentation of change points + densest-segment selection) keeps a metric solely on the grounds that it has a change point coinciding with the failure time window. This leaves the following residual errors:

1. **Noise contamination inside the correct segment (precision loss)**: a one-off batch-job spike, a scrape glitch, or a transient outlier that happens to have a change point at the same time slips into the selected set. Estimated to be the most frequent residual error.
2. **Wrong segment selected due to noise clustering**: when non-failure change points cluster in time, that cluster can out-compete the true failure segment as the densest segment. Rare, but severe when it happens.
3. **KDE boundary distortion**: noise change points distort the density near the failure segment and shift its boundary. Estimated to be the rarest of the three.

Findings from the original paper (IEEE Access 2024) that are directly relevant to this design:

- **Lesson 4 from the paper**: redundancy-reduction baselines based on inter-metric similarity (HDBS-SBD/HDBS-R) mistakenly dropped failure-related metrics, making downstream fault localization *worse* than no reduction at all. Naive redundancy reduction is dangerous.
- **Lesson 5 from the paper**: reduction quality (balanced accuracy) explains only 13.9-51% of the variance in downstream localization quality (AVG@5). The top priority for any new phase is therefore "do not harm localization."
- Related work: failure-induced deviations persist, whereas noise reverts to its original level (the persistence filter in Minder, NSDI'25; change-shape classification approaches such as SCELM).

## 2. Proposed Method: Persistence Gate

For each (metric, change point) pair, we test — univariately, intra-metric — whether the post-change deviation persists in the direction of the shift, and exclude only transient changes (single-point spikes / short-lived blips) from segment membership and segment selection. Because this method **never uses similarity to other metrics**, it structurally avoids the Lesson-4 trap (redundancy reduction mistakenly discarding failure metrics).

### 2.1 Persistence Score (pure function, deterministic)

Computed once per (metric, cp) right after STEP1 (CPD) and cached.

```
compute_persistence_scores(X, metric_to_cps, sigma_estimator,
                           min_duration="auto", deviation_k=0.5):
  n = len(X)
  W = clamp(round(0.1 * n), PERSISTENCE_FLOOR=3, n)      # min_duration="auto"
  for (m, cp):
      next_cp = next CP of the same metric (n if none)
      post = x[cp : min(cp + W, next_cp)]                # never mixes in the next regime
      prev = previous CP of the same metric (0 if none)
      pre  = x[prev : cp]
      if len(post) < 3 or len(pre) < 1:
          score = 1.0   # reason="too_short_to_verify" (conservative boundary rule)
      elif post is entirely NaN:
          score = 1.0   # reason="missing_value_boundary" (treat reporting outage as persistence)
      else:
          b     = nanmedian(pre)
          sigma = reuse the same sigma-estimation logic as detection (algo/detection.py)
          if sigma is degenerate (<= 0 or non-finite):
              score = 1.0   # reason="degenerate_scale" (conservative rule)
          else:
              shift = nanmedian(post) - b
              thr   = max(0.5 * abs(shift), deviation_k * sigma)
              score = nanmean(sign(shift) * (post - b) >= thr)  # fraction of post persisting in the shift direction
```

Time complexity is O(total number of CPs × W); no randomness involved.

### 2.2 Behavior by Change Shape

| Change shape | Behavior | Rationale |
|---|---|---|
| Level shift | Survives (score ≈ 1) | `post` stays at the new level, so `thr ≈ 0.5·Δ` keeps being satisfied |
| Single-point spike | Dropped (score ≈ 1/\|post\|) | median(post) ≈ b so Δ ≈ 0; only the spike point exceeds `thr ≈ k·σ` |
| Transient (reverts after a few points) | Dropped | samples after the reversion fall below the threshold |
| Small but persistent creep | **Survives** | `thr ≈ 0.5|shift|` judges by persistence, not amplitude, so it does not make the paper's known limitation (mistakenly discarding small root-cause changes) any worse |

### 2.3 Insertion Point: STEP2.5 (pre-selection gate)

The gate is folded into the segment-selection function. The KDE geometry (segment boundaries) is left untouched — **only segment membership and the selection score** are restricted to persistent change points.

```
gated_selector(label_to_metrics, metric_to_cps, label_to_change_points):
    for label:
        kept = { m ∈ metrics | ∃cp ∈ (label's CPs ∩ metric_to_cps[m]):
                                 score[(m, cp)] >= persistence_ratio }
        cleaned[label] = kept if kept else metrics   # per-segment floor (avoids empty sets, records floor_applied)
    return select_largest_segment_with_label(cleaned, metric_to_cps, label_to_change_points)
```

- The same `gated_selector` is injected into **both** the STEP3 final selection **and** `select_bandwidth` (the bootstrap used when `bandwidth="auto"`). Injecting it into only one would let the auto-tuning objective diverge from the actual output.
- The persistence score is a pure function of `X` and is invariant under metric resampling. Inside the bootstrap it is only looked up from the cache — never recomputed.
- It does not interfere with `penalty="auto"` (self-contained within STEP1) or with `run_upto_cpd`.

## 3. Comparison of Insertion Points

deep-reasoner and Codex were given the same problem in parallel and analyzed it independently; **both ranked option C first**.

| Option | Location | Errors it fixes | Damage from misclassification (false negatives) | Interaction with auto-tuning | Verdict |
|---|---|---|---|---|---|
| A | Hard removal after CPD, before KDE | ①②③ | **Worst**: metric loss plus weakened density at the failure segment → cascades into boundary shift or segment disappearance | Changes the input statistics themselves (contaminates tuning too) | Rejected. This effectively reproduces the Lesson-4 failure mode at an earlier stage |
| B | Weight KDE density by persistence score | ③ only | Mild cascading | `weights` requires `fft=False`, which is the most invasive wiring change | Rejected. Zero precision gain for a geometry-level risk |
| **C** | **Pre-selection gate** | **①②** | Bounded recall loss (drops out of membership only) | Fully consistent via injection into `gated_selector` | **Adopted** |
| D | Post-selection gate (`selected_metrics` only) | ① only | Minimal | No interference | Subsumed by C as a restricted special case (same score atoms) |

Deciding factor: error frequency is ① ≫ ② ≫ ③, and C covers ①② while discarding only the rarest, ③. False-negative damage is bounded more tightly the further downstream the gate sits (D > C > B > A). The existing `select_bandwidth(selector=...)` injection seam naturally accommodates C.

## 4. Proposed API

- Appended to the end of `Sifter.__init__` (default off = the selector remains ungated, fully matching current behavior):
  - `persistence_gate: bool = False`
  - `persistence_ratio: float = 0.75` (persistence ratio ρ required to keep a metric)
  - `persistence_min_duration: int | str = "auto"` (verification window W; auto = clamp(0.1n, 3, n))
  - `persistence_deviation_k: float = 0.5` (noise-floor coefficient k; smaller values protect smaller sustained shifts)
- `SiftResult`:
  - New bucket `filtered_transient: frozenset[str]` = the raw membership of the selected label minus the post-gate selected set (preserves the partition invariant: `selected ∪ no_change ∪ no_change_points ∪ out_of_segment ∪ transient` = the input series)
  - `persistence: PersistenceInfo | None` (`requested` / `ratio` / `deviation_k` / `resolved_min_duration` / `metric_to_score` / `floor_applied`, round-trips via `to_dict`/`from_dict`)
- `SifterTransformer` / CLI (`--persistence-gate`, etc.) / `__init__.py` exports / README are updated in the same change

## 5. Risks

1. **A transient but genuine failure** (a short error burst that self-recovers): mitigated by defaulting to off, by tuning room in ρ/k, and by the conservative boundary rule — but the residual risk should be stated explicitly.
2. **Mistakenly discarding a small but persistent root-cause change**: mitigated by making `0.5·|shift|` the dominant term of `thr`; must be checked with a regression test during evaluation.
3. **③ (KDE boundary distortion) is not addressed by this design**: if empirical measurement shows ③ actually dominates, address it via bandwidth/penalty tuning instead (still avoiding a pre-KDE pruning cascade).

## 6. Test Plan (at implementation time)

Using hand-built synthetic data with no `pyrca` dependency: shape-discrimination unit tests / a ① precision scenario (recall unchanged, reduction rate increases when spikes are injected) / a ② mis-selection scenario (gate ON is the only configuration that selects the true segment when spikes cluster against a persistent-shift group) / full no-op verification with the default off / integration with `bandwidth="auto"` (determinism, gate propagation into the bootstrap) / conservative rules and the floor / the partition invariant, serialization, clone compatibility, and the CLI.

## 7. Evaluation Infrastructure Requirements (blocker before implementation)

Whether this method should be adopted can only be judged once the following can be measured. The current tests only provide regression coverage — **there is no infrastructure for evaluating reduction performance today**.

1. **Classify and measure residual errors**: on the synthetic benchmark, classify the current pipeline's errors into ① in-segment noise, ② wrong-segment selection, and ③ boundary distortion, and measure their empirical frequency. The ranking in §3 is inferred from the design, and it is the one point where the conclusion on insertion point could change — specifically if ① ≫ ② ≫ ③ turns out not to hold, and especially if ③ turns out to dominate.
2. **Reduction metrics**: compare recall / specificity / balanced accuracy / reduction rate against ground truth (M_A ∪ M_B, the set of failure-propagation nodes), with the gate ON vs. OFF, broken down by failure type and by noise-injection rate. **Non-inferiority of recall is the pass condition.**
3. **Downstream localization metrics**: per Lesson 5 (the weak correlation between BA and AVG@5), measure downstream RCA's AC@K / AVG@K alongside reduction metrics, not reduction alone (using the `experiments/` framework or a lightweight version of it). The pass line is "no worse than no reduction."
4. **Data generation**: extend `tests/sample_gen/generator.py` (PyRCA DAG synthesis) with scenario generation that can inject single-point spikes, transients, small persistent creep, and propagation delay in a controlled way. Also keep a validation path open against empirical data (the paper's datasets).
5. **Baselines**: reproduce current MetricSifter (gate OFF) and, where possible, the paper's redundancy-reduction baselines, and show by comparison that the gate avoids the "redundancy-reduction trap."

## 8. Record of the Design Process

Following the orchestration workflow, the same problem was given to deep-reasoner and Codex (GPT-5 series) in parallel, each without seeing the other's answer, and the results were then synthesized.

- **Phase proposal**: both converged independently on "persistence verification" (Codex: Sustained Change Verification / deep-reasoner: Persistence Gate). Both also produced a surprisal-based ranked list of runner-up ideas that agreed with each other (held out of scope for now).
- **Insertion point**: on the first pass, Codex picked "before KDE" and deep-reasoner picked "after selection" — a split decision. After the user asked for a re-examination, both were run again in parallel and this time **both ranked C (pre-selection gate) first**. deep-reasoner also proposed a staged rollout D → C, but since D is a restricted special case sharing the same score atoms as C, we go directly to C and subsume D.

## References

- MetricSifter: Feature Reduction of Multivariate Time Series Data for Efficient Fault Localization in Cloud Applications (IEEE Access 2024) — in particular, the degradation caused by redundancy-reduction baselines (Lesson 4) and the weak BA↔AVG@5 correlation (Lesson 5)
- Minder: Faulty Machine Detection for Large-scale Distributed Model Training (NSDI 2025) — the persistence (continuity) filter
- Sieve: Actionable Insights from Monitored Metrics in Microservices (2017) — clustering-based metric reduction and interpretability
- Change-shape classification (level shift / single spike / transient) work: SCELM, TAMO, and related multimodal fault-diagnosis research
