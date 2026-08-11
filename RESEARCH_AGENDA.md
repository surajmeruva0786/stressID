# Research Agenda — From "StressID is leaky" to a Methods Contribution

> Written 2026-08-11. Supersedes the forward-looking parts of
> `RESEARCH_PROGRESS.md` §12/§15.5. Everything already *measured* stays where it
> is; this document is only about what to do next and why.

---

## 0. The problem with the current position

What we have is real and reproducible (`LEAKY_PROTOCOL.md`): random splits on
StressID inflate scores, the inflation attaches specifically to identity-bearing
features, and a signal-free identity oracle (0.628) beats honest models that
process the actual physiology (0.540).

What we have is also, as a paper, **a critique of one dataset**. The standard
reviewer response to that class of submission is:

> "Subject-independent evaluation is standard practice and has been discussed in
> the affective-computing and EEG literature for a decade. What is new here?"

That objection is fair against the current framing and cannot be answered by
adding more StressID runs. Four architecture-level nulls and a confirmed 0.519
plateau (§15.7) mean **there is no more signal to extract from this dataset by
modelling.** The remaining value is entirely in generalising the *measurement*.

**The pivot:** stop treating leakage as a property of StressID, and start
treating it as a **quantity that can be measured a priori, predicted across
datasets, and reported as a mandatory baseline.** A dataset critique gets
rejected; a diagnostic that other people run on their own data gets cited.

### Novelty must come from these four places, not from "we used GroupKFold"

| # | Claim | Novel? |
|---|---|---|
| — | "Subject-independent splits are better" | **No.** Known since ~2015. Never lead with this. |
| L1 | Task/stimulus leakage survives subject-independent splits — a *second* axis nobody removes | **Yes**, for this dataset class |
| L2 | Inflation is *predictable* from two training-free statistics (I, C) | **Yes** — this is the engine |
| L5 | The leak is **label-side (self-report idiosyncrasy)**, not signal-side (biometrics) | **Yes, and quotable** |
| L6 | Leakage is a continuum parameterised by k-shot subject calibration, not a binary | **Yes**, and it makes the paper constructive |

---

## 1. Workstreams, in priority order

Each has: the question, the experiment, the prediction, and **what would kill it.**
A workstream that gets killed is a result and gets written up as one — that
discipline is why the existing negative results are trustworthy (§13).

---

### L0 — Prior-art sweep (do this FIRST, 2–3 days, blocking)

Before any compute. Search for existing work on:

- "data leakage" / "subject-dependent evaluation" in affective computing, HAR,
  EEG emotion recognition, wearable stress detection
- Kapoor & Narayanan, *Leakage and the Reproducibility Crisis in ML-based
  Science* — the template for how this kind of paper is structured and the
  citation that will be demanded
- Any existing re-evaluation of WESAD / DEAP / AMIGOS / StressID under corrected
  protocols
- Subject-identity probing / "identity confound" in physiological ML

**Deliverable:** `research_way/RELATED_WORK.md` with a positioning paragraph
stating exactly which of L1/L2/L5/L6 remain unclaimed.

**Kill condition:** if someone has already published the I/C prediction model
(L2), drop L2 to a replication and promote **L5** to the headline. If someone has
published the label-side result (L5), promote **L1**. Do not proceed to compute
until this is answered — it decides the paper's title.

---

### L1 — The second leak: task/stimulus overlap (1 week, local data only)

**Question.** Our protocol is subject-disjoint but **task-overlapping**. Every
test recording's task (Stroop, Math, Counting1/2, Speaking, …) also appears in
training, and each task carries a fixed stress semantic for essentially everyone.
A model can therefore score by recognising *the stimulus* and recalling that
stimulus's usual label — the same shortcut as the identity oracle, keyed on task
instead of person. The modality-availability confound (§9.1) is a *symptom* of
this, not a separate phenomenon.

**Experiment — the 2×2 protocol grid.** Four cells, same features, same models,
only the split rule changes:

| | task overlap allowed | task-disjoint |
|---|---|---|
| **subject overlap allowed** | leaky-leaky (≈ published protocol) | — |
| **subject-disjoint** | our current protocol | **both-disjoint** |

Implementation: extend `research_way/src/splits.py` with a
`make_splits(..., group_by={"subject","task","both"})` argument; the assertion
`assert_no_leakage` generalises to whichever grouping is active. Reuse
`src/classical.py` unchanged.

Also build the **task oracle** — the exact mirror of E-D: predict a test
recording's label from the majority label of *that task's* other recordings in
train, using no signal at all.

**Prediction.** Monotone decrease left-to-right and top-to-bottom:
`0.735 → 0.519 → ? → ?`, with the both-disjoint cell approaching the majority
floor (0.418). The task oracle should score high — probably higher than the
identity oracle, since task→label is more deterministic than person→label.

**Why it matters.** If the both-disjoint cell collapses, then *even the corrected
literature* — every paper that responsibly used subject-independent splits —
still overstates generalisation, because it never removed the stimulus shortcut.
That is a claim about the corrected state of the art, not about StressID.

**Controls (mandatory).** Task-disjoint folds change class balance drastically
(Relax is ~all-negative, Stroop ~all-positive). Report the per-cell majority
floor alongside every cell, and report *margin over floor*, never raw F1. Without
this the comparison is meaningless and a reviewer will say so.

**What would kill it.** Both-disjoint ≈ subject-disjoint (no additional drop
beyond the class-balance artefact). Then task leakage is negligible here, write
it up as a null and L1 becomes a one-paragraph control in the main paper.

---

### L2 — The engine: predict inflation before training anything (3–4 weeks)

**Question.** Is leakage inflation a *predictable function* of two statistics
that can be computed without training the target model?

- **I — subject identifiability.** Accuracy of a logistic regression predicting
  *which subject* a recording came from, relative to chance. StressID: physio
  0.649 vs 0.016 chance = 41.5×.
- **C — within-subject label consistency.** Fraction of a subject's recordings
  carrying that subject's modal label. StressID: 0.708.

**Hypothesis.** `inflation ≈ g(I, C)`, monotone increasing in both, ≈ 0 when
either is at its floor. Start with the analytical bound: the identity oracle
cannot exceed what C permits, and cannot be reached unless I is high enough for
the model to recover identity from features. Fit the simplest 2-parameter form
that works; do not reach for anything fancier than it earns.

**Experiment.** Get 5–8 (dataset × modality) points. Candidates, easiest first:

| Dataset | Why | Access |
|---|---|---|
| **WESAD** | 15 subjects, wearable, the most-cited stress benchmark; already planned as O5/O7 | public, small |
| **UBFC-Phys** | rPPG + EDA stress, subject-rich | public on request |
| **SWELL-KW** | knowledge-work stress, HRV | public |
| **K-EmoCon** | multimodal, continuous annotations | public |
| **AMIGOS / DEAP** | EEG-adjacent, huge literature, high identifiability expected | public on request |
| **StressID** | our anchor point, already measured | local |

For each: compute I and C (cheap, no target model), then *measure* actual
inflation by running the E-A protocol swap. Plot predicted vs measured.

**Deliverable.** If a 2-parameter model predicts inflation with r > 0.8 across
the points, practitioners get a **10-minute, training-free estimate of how much
their random-split number is inflated.** That is the artifact the paper is built
around and the reason it gets cited rather than filed.

**What would kill it.** No correlation across datasets, or the relationship is
dominated by a third factor (recordings-per-subject, class balance). Fallback:
downgrade L2 from "predictive law" to "descriptive multi-dataset audit" — still
publishable, considerably less exciting. Decide by the 4th data point, not the 8th.

**Risk to manage.** This is the workstream that can eat two months on data
acquisition and preprocessing alone. Time-box each dataset to one week; if the
signals won't load, drop it and move on. Three datasets done properly beats six
half-loaded.

---

### L5 — The mechanism: is the leak in the signal or in the labels? (3–5 DAYS, local data only)

> **Highest novelty per hour of any item in this document. Run it early —
> alongside L0 — because it can change the paper's title.**

**Question.** We have been telling a signal-side story: *physiology is a
biometric, so the model recognises the person.* But the number that actually
makes the identity oracle work is **C = 0.708, within-subject label consistency**
— and that is a property of **self-report**, not of physiology.

People use rating scales idiosyncratically. One participant lives at 7–9, another
never goes above 4. Binarising a 0–10 self-assessment at a **global** threshold
therefore makes each subject's label near-constant, and "which subject is this"
becomes a near-sufficient statistic for the label. On this account the leak is a
**labelling artefact**, and the biometric story is a passenger.

**Experiment.** `StressID Dataset new/self_assessments.csv` holds the continuous
0–10 ratings per subject × task — the raw material the binary label was derived
from.

1. Re-derive the label by **per-subject median split** of the continuous rating
   (removes subject-level scale bias *by construction*; keeps within-subject
   ordering intact).
2. Recompute **C** under the new labelling. Prediction: 0.708 → ≈0.5 by design.
3. Re-run the entire `prove_leakage.py` chain on the re-derived labels: E-A
   protocol swap, E-C identity probe, E-D identity oracle.
4. **I should barely move** (features are untouched — physiology is exactly as
   biometric as it was). **Inflation and the oracle should collapse** if the
   mechanism is label-side.

**Interpretation — this is the point of the experiment.**

| Outcome | What it means | Headline |
|---|---|---|
| Inflation collapses, I unchanged | The leak was **label-side idiosyncrasy** | *"Self-report idiosyncrasy, not physiology, is what random splits exploit"* — new, generalises to every self-reported affect dataset |
| Inflation persists | The leak is genuinely **signal-side biometric** | Current story confirmed and *strengthened* by having ruled out the alternative |
| Partial | Decompose and report the split | Quantified attribution — still a strong result |

**Every outcome is publishable.** There is no failure mode, only three different
papers. That asymmetry is why it runs first.

**Caveat to state honestly in the write-up.** Per-subject median splitting is not
a "more correct" ground truth — it discards genuine between-subject differences
in true stress level, and a person who really was calm throughout gets half their
recordings labelled stressed. It is a **diagnostic manipulation for attributing
the leak**, not a proposed relabelling of the dataset. Say this explicitly or a
reviewer will say it for you.

---

### L6 — The constructive half: leakage as a continuum, not a binary (1 week)

**Question.** The leaky/honest dichotomy is a false one. Real deployments *do*
get some labelled data from the target user. The right axis is **k** = number of
labelled recordings from the test subject available at training time.

**Experiment.** Sweep k ∈ {0, 1, 2, 3, 5, 8, all}, with the k calibration
recordings drawn from that subject's *non-evaluated* tasks (the c2 discipline —
never touch an evaluated recording). Plot macro F1 vs k with CIs.

- **k = 0** is our honest protocol → 0.519
- **k = all** is approximately the leaky protocol → ≈0.735
- The published literature claims it is reporting k = 0 while actually sitting
  somewhere near k = all. **The curve is the argument** — one figure that makes
  the whole critique legible without a single table.

`c2_subject_relative` (0.5371) is already a point on this curve; formalise it.

**Why this makes the paper.** It converts a destructive finding into a
prescription: *you do not have to ban subject overlap — you have to report where
on the calibration curve you sit.* Reviewers reward critiques that ship a
remedy. This is the remedy.

---

### L3 — Field audit: how much of the literature is affected? (2–3 weeks, parallelisable)

**Question.** What fraction of published results on these benchmarks use a leaky
protocol, and what would their corrected numbers be?

**Method.**
1. Pre-register inclusion criteria **before** collecting. Papers 2019–2026 using
   StressID / WESAD / SWELL / DEAP / AMIGOS for stress or affect classification.
   Semantic Scholar API for retrieval.
2. Code each paper's protocol: `subject-independent` / `random-split` /
   **`unreported`**. The `unreported` bucket is itself a finding — expect it to
   be large.
3. Two-rater coding with agreement statistics (Cohen's κ). A second pass by a
   different method or reader is required; single-rater coding will be
   challenged and the challenge would be correct.
4. Apply L2's model to estimate each paper's leakage-free number.

**Deliverable.** "Reported vs. estimated leakage-free" across N≈40–80 papers,
plus the share that never states its protocol. This is what converts a methods
note into a paper the community argues about.

**Do not start this before L2 has a working model** — without L2 the corrected
column cannot be filled and the audit is just a protocol tally.

---

### L4 — Ship the tool: `leakcheck` (1 week, do LAST)

Package the diagnostics so others can run them in one command on their own data:

- subject-identity probe → **I**
- within-subject label consistency → **C**
- **identity oracle** and **metadata/availability oracle** baselines
- majority floor
- predicted inflation from L2
- a one-page reporting checklist for papers

Tools get cited by people who never read the paper. This is the highest-leverage
week in the plan, and it must come last — it packages results, it does not
produce them.

---

## 2. Recommended sequencing

```
Week 1        L0 prior-art sweep        ──┐ both are cheap, local, and
Week 1–2      L5 label-vs-signal        ──┘ together they fix the title
Week 2–3      L1 task-leakage 2×2 grid
Week 3–6      L2 multi-dataset (WESAD first, 1 week each, hard time-box)
Week 5        L6 calibration curve      (runs in parallel — existing code)
Week 6–9      L3 literature audit       (needs L2's model)
Week 9–10     L4 leakcheck package
Week 10–12    Writing; arXiv preprint to stake the claim; then submit
```

**Decision gate at end of Week 3.** With L0, L5 and L1 in hand the title is
determined. If L5 came back label-side, that becomes the headline and L2 becomes
supporting evidence. If both L5 and L1 are nulls, the honest conclusion is that
the existing StressID result is the whole contribution — write the short version,
submit it to a workshop, and stop. **Do not enter L2/L3's two-month commitment
without a positive result from L5 or L1.**

---

## 3. Venues

| Venue | Fit | Note |
|---|---|---|
| **IEEE T-AFFC** | best fit | journal; methodology critiques land well; no deadline pressure |
| **ACII** | strong | the community that must hear it |
| **NeurIPS D&B** | good if L2+L3 both work | it is where StressID itself was published — symmetry helps |
| **IMWUT / UbiComp** | good | wearable-stress audience; values deployment realism (L6) |

Post the preprint to arXiv the moment L5 and L1 are settled. This area is
crowded enough that a claim can be scooped in a quarter.

---

## 4. Standing rules carried forward

Unchanged from §15.6 and the reason the existing negatives are trustworthy:

1. **Pre-register each experiment's hypothesis in this file before running it.**
2. **Log every run, including failures**, in `reports/runs_index.csv`. Never prune.
3. **No SOTA claim without out-of-search confirmation** on partitions fixed in
   advance.
4. **Never quote the campaign maximum** as a clean estimate — the max over many
   runs is biased upward (measured at +0.0135 in §15.7).
5. **A null is a result.** Write it up with the same care as a positive.

---

## 5. What this agenda explicitly does not do

- **No more architecture work on StressID.** Four nulls, a confirmed plateau, and
  a 1.2 M-parameter transformer that loses to an SVC on 32×32 pixel statistics.
  The ceiling is the data, not the model. Any further architecture run needs a
  written justification for why it is not the fifth null.
- **No attempt to beat 0.72.** It is reachable by adopting the leaky protocol —
  our own baselines hit 0.735/0.740 there — and doing so would reproduce the
  exact error this project exists to document.
- **No re-download or data-integrity work.** Settled 2026-08-04: byte-identical,
  and neither leakage nor the availability confound lives in the files.
