# paper.tex audit — misinformation & required updates (2026-06-05)

Audit triggered by the Newton in-engine revalidation. Two classes: **(A) factual
errors / internal contradictions** to fix, **(B) stale results** to replace with
the new controlled in-engine comparison.

## A. Factual errors / contradictions

1. **Abstract Chrono description contradicts the Chrono section's own setup.**
   Abstract (L129) says Chrono used a "rigid flat surface with randomised low
   friction (μ∈[0.10,0.35])". The recent live validation (`chrono_validation.py`,
   RUNBOOK, `aau_*` logs) used **granular SCM Bekker–Wong** terrain with φ=10–30°.
   The paper documents the *older rigid-μ Curiosity* experiment (47.3%), not the
   one the repo now runs. → MOOT: Chrono section being dropped entirely (user call).

2. **"Emergent coordinated rocking" is not what the policy does.**
   Abstract L119, Results L1751–1755 ("Deep d≥22cm: coordinated rocking…
   alternating forward/reverse"), Discussion L2232–2233, Conclusion L2369.
   Action-trace analysis of the good seeds (in-engine): seed1 floors forward 85%
   of steps, reverse 4%, **0 mean-drive sign flips** (no rocking); seed3 70%
   forward, 13% reverse, mild rocking. BOTH are dominated by **aggressive steering**
   (66–72% of steps at full steer lock). The recovery mechanism is steering, not
   rocking. → Replace "rocking" narrative with steering-dominated finding.

3. **Internal inconsistency: 86.9% escape vs "mean escape displacement 0.93 m".**
   Table tab:results (L1595) lists mean escape displacement 0.93 m, yet 86.9%
   episodes supposedly escape at the 3.0 m line (and Limitation (i) L2282 says
   escaped episodes exit at median 3.02 m). If 86.9% reach 3.0 m, mean displacement
   cannot be 0.93 m. The 0.93 m / milestone figures look like a stale (older,
   shallow-bed) run mixed with a different success number. → Replace headline with
   new controlled eval.

4. **Seed numbering wrong + outdated status.**
   Abstract "single seed; 3 additional seeds in progress"; Limitation (0) L2264
   "single seed (seed_1)… Seeds 3,4,5 currently training". Training seeds are 0–4.
   Final status: **seeds 1 and 3 are the two good seeds** (2&4 collapsed early,
   ~226k/242k; seed 0 trained but weaker). → State 2 validated good seeds (1,3).

5. **Particle count stale.** POMDP L550 "~150k particles per environment". Current
   0.60 m bed = 140×140×24 grid → ~497k particles/env (31.8M / 64 envs). → Update.

6. **Bi & Ding comparison hinges on 86.9% (L2247).** Recompute/soften once headline
   number is the new controlled eval.

## B. Stale results to replace with new in-engine comparison

New controlled eval (Newton, AAU rover, correct pre-reset escape flag + blowup
guard, sinkage 0.20+0.28 m, 16 envs):

| Controller | escape | mechanism |
|---|---|---|
| rocking (scripted ±drive) | **0%** | oscillates in place |
| constant_drive (floor straight) | **35%** | drives out when soil favorable |
| policy seed1 | **100%** | floor + hard steering |
| policy seed3 | **90–95%** | floor + steering + some reverse |

(Scaled sweep at 0.15/0.20/0.25/0.28 × 30 trials running → final CIs + depth curve.)

- **Replace** Escape Performance (tab:results) headline with the 4-controller
  comparison table + escape-vs-sinkage figure + action-behavior figure.
- **Delete** §Cross-Engine Sim-to-Sim Validation (L2105–2222) entirely.
- **Rewrite** Limitation (iii) (L2298) — it leans on the dropped Chrono study;
  reframe around the in-engine baseline comparison + honest "no hardware / no
  cross-engine generalization claim" statement.
- **Why-MPM** (L2227) and Conclusion: keep the MPM argument but drop the "rocking"
  framing; the steering strategy is the emergent behavior.
- Keep dual-brain A→B sim2sim section (separate from Chrono) but verify its
  "preliminary results" are real before relying on them.

## Edits NOT depending on sweep (safe to do now)
- Drop Chrono section + abstract Chrono sentence + keyword "sim-to-sim validation"
  (re cross-engine), Replication block.
- rocking→steering language fixes (items 2).
- seed numbering/status (item 4), particle count (item 5).

## Edits gated on sweep completion
- New results table + figures, headline escape number in abstract/intro/conclusion,
  Bi & Ding number.
