# The Leaky Protocol — What It Is, and Proof That It Inflates StressID Scores

> **Status:** all numbers below are regenerated from scratch by
> [`research_way/prove_leakage.py`](research_way/prove_leakage.py) on the full
> 700-recording corpus. Raw outputs: `research_way/reports/leakage_proof/`.
> Last run: 2026-08-06.
>
> **Reproduce:** `cd research_way && python prove_leakage.py` (~4 min, CPU only,
> uses cached features — no GPU and no re-preprocessing needed).

---

## 1. The claim in one paragraph

StressID has 64 subjects who each performed ~11 tasks, giving 700 recordings. How
you divide those into train and test decides what the model is rewarded for
learning. If you shuffle the **recordings** and take 80/20, the same person
appears on both sides — the model sees person `2ea4` doing Math in training and
is then tested on `2ea4` doing Stroop. Physiological signals are close to a
biometric, so the model can succeed by recognising the *person* and recalling
what that person usually reported, without learning anything about stress. We
call that the **leaky protocol**. Splitting by **subject** instead (GroupKFold)
removes the shortcut, and scores drop. The drop is not a regression — it is the
removal of a measurement error.

**Crucially: leakage inflates scores. It does not depress them.** The published
0.72 is high partly because of this. Our ~0.52 is lower because we took it away.

---

## 2. The two protocols, precisely

| | Leaky (random KFold) | Leakage-free (subject GroupKFold) |
|---|---|---|
| Unit that is shuffled | recording | **subject** |
| Can subject X be in train *and* test? | yes | never |
| Measured train/test subject overlap | **58.4 of 64 subjects** | **0.0** |
| Question it answers | "…for people it has already met?" | "…for a person it has never seen?" |
| Matches deployment? | no | yes |

That overlap row is measured, not asserted — `mechanism.json → subject_overlap`.
Under the leaky protocol **91% of subjects (58.4/64) sit on both sides of the
split simultaneously.**

Implementation: `research_way/src/splits.py`. It computes subject-level folds
once, writes them to disk, and every experiment reuses them, so all numbers are
apples-to-apples. It also ships an assertion that fails loudly on overlap:

```python
def assert_no_leakage(folds):
    for f in folds:
        tr, va, te = set(f["train"]), set(f["val"]), set(f["test"])
        assert not (tr & te), f"fold {f['fold']}: train/test subject overlap"
        ...
    all_test = [s for f in folds for s in f["test"]]
    assert len(all_test) == len(set(all_test)), "a subject is tested twice"
```

---

## 3. Proof

Four experiments. Each is designed to kill a specific objection.

### E-A — The symptom: same features, same model, only the split rule changes

Nothing varies except how rows are assigned to folds. Full corpus, 700
recordings, 5 folds, macro F1.

| Features | Model | Leakage-free | **Leaky** | Inflation |
|---|---|---|---|---|
| **physio** | RF | 0.540 | **0.638** | **+0.098** |
| **physio** | logreg | 0.565 | 0.624 | +0.058 |
| video | RF | 0.622 | 0.678 | +0.056 |
| feature_fusion | RF | 0.683 | 0.735 | +0.051 |
| audio | RF | 0.678 | 0.715 | +0.037 |
| audio | logreg | 0.686 | 0.705 | +0.019 |
| *availability_only* | *RF* | *0.696* | *0.693* | ***−0.003*** |
| *availability_only* | *logreg* | *0.697* | *0.698* | ***+0.001*** |

Every real modality gains from leakage. **Physiology gains most (+0.098)** —
exactly what the fingerprint explanation predicts, since it is the most
person-identifying modality. It is also the modality the origin paper reports its
best unimodal number on.

### E-B — Negative control: the row that *must not* move, and doesn't

The obvious objection to E-A: maybe random KFold just produces easier folds for
some unrelated reason — different fold sizes, kinder class balance, luckier
seeds. If so, *every* row would rise, including ones that cannot possibly benefit
from knowing who the subject is.

`availability_only` is that control. It discards all signal content and uses only
three bits: does physio / audio / video exist for this recording. Those bits
carry **zero** subject identity — they are a property of the task, not the
person. Prediction: inflation ≈ 0.

Measured: **−0.003 (RF) and +0.001 (logreg)** — indistinguishable from zero,
while physio moves +0.098 in the same run.

> This is what separates "leakage" from "the split was just easier." The lift in
> E-A is specifically attached to features that identify people.

### E-C — The mechanism: physiology is a fingerprint

E-A shows the symptom. This shows *why*. We discard the stress labels entirely
and ask a different question: **can a classifier tell which of the 64 subjects a
recording came from?**

| Modality | Subject-ID accuracy | Chance | Ratio |
|---|---|---|---|
| **physio** | **0.649** | 0.016 | **41.5× chance** |
| audio | 0.467 | 0.016 | 29.9× |
| video | 0.403 | 0.016 | 25.8× |

From a few seconds of ECG/EDA/respiration statistics, a plain logistic regression
picks the right person out of 64 **two-thirds of the time**. Identity is sitting
in the features in enormous quantity.

Note the ordering: physio > audio > video for identifiability, and physio has the
largest leakage inflation in E-A. The mechanism and the symptom line up.

### E-D — The smoking gun: the identity oracle

The decisive experiment. Build a predictor that uses **no signal whatsoever** —
no ECG, no audio, no video, not one feature. For each test recording it does
exactly one thing:

> "Which subject is this? Look up that subject's *other* recordings that landed
> in the training set, and predict whatever they usually reported."

Under the leaky protocol this is well defined for **100% of test recordings**,
because every test subject also appears in train. Results:

| Predictor | Signal used | macro F1 |
|---|---|---|
| **Identity oracle, leaky split** | **none — subject ID only** | **0.628** |
| Honest physio model, GroupKFold | full ECG + EDA + respiration | 0.540 |
| Honest video model, GroupKFold | full face video | 0.622 |
| Honest fusion model, GroupKFold | all three modalities | 0.683 |

**A lookup table keyed on "who is this person" scores 0.628 — beating the honest
physiological model (0.540) and the honest video model (0.622) that actually
process the signals.**

Under GroupKFold the same oracle is not merely worse, it is **undefined**: the
test subject has no training recordings to look up. That asymmetry *is* the leak.

Why it works: **within-subject label consistency is 0.708.** People are
repetitive — most subjects report the same way across most of their tasks. So
identity is worth ~0.63 macro F1 on its own, and a leaky split hands that to the
model for free.

---

## 4. What this does and does not prove

**Does prove:**
- Random-split evaluation on StressID measurably rewards subject recognition (E-A).
- The lift is specific to identity-bearing features, not a fold artifact (E-B).
- The features contain identity in bulk — 41.5× chance (E-C).
- Identity alone, with no signal, beats real leakage-free models (E-D).

**Does not prove:**
- *Where* the leak lives. The chain E-C → E-D runs through **C = 0.708, the
  within-subject label consistency** — and that is a property of self-report, not
  of physiology. The biometric reading (physiology identifies people) and the
  labelling reading (participants use 0–10 scales idiosyncratically, so a global
  binarisation threshold makes each subject's label near-constant) both predict
  everything measured above. Discriminating them is workstream **L5** in
  [`RESEARCH_AGENDA.md`](RESEARCH_AGENDA.md), and it is the next experiment to run.
- That subject-disjoint splitting removes *all* leakage. These folds are
  subject-disjoint but **task-overlapping**; stimulus leakage is untested here and
  is workstream **L1**.
- That published StressID results are *only* leakage. The origin paper also uses
  far richer handcrafted features (98 HRV/EDA/RRV descriptors, OpenFace AUs) than
  our crude window statistics, plus SMOTE. Leakage is one component of the gap,
  not all of it. A true like-for-like reimplementation of the competitor papers on
  our splits has **not** been run.
- Anything about the *dataset files*. See §6.

---

## 5. Consequences for the headline numbers

| System | Protocol | macro F1 |
|---|---|---|
| Origin paper (SVM + average-rule fusion) | random 80/20 + SMOTE | 0.72 wF1 / 0.65 bal-acc |
| Our classical baselines | **leaky**, to match | 0.735 |
| Our best system, confirmed out-of-search | **leakage-free** | **0.519** |
| Deep MST-temporal (1.2 M params) | leakage-free | 0.485 |
| Majority class | — | 0.418 |

The uncomfortable implication, stated plainly: **we can beat the published number
whenever we choose to.** Switching to random splits puts our own baselines at
0.735, above the paper's 0.72. It would be meaningless, so we do not report it as
a result. Under a protocol where no subject appears on both sides, the ceiling on
this data is ~0.52, and our campaign reached it.

**The gap between those two numbers is the contribution.**

---

## 6. What leakage is *not*

Two distinct issues are easy to conflate, and only one lives in the data:

| | Subject leakage | Modality-availability confound |
|---|---|---|
| Where it lives | evaluation protocol (`src/splits.py`) | the recording design itself |
| Effect | **inflates** scores | inflates scores |
| Fixable by re-downloading? | **no** — no data involved | **no** — present in every copy |
| Fix | split by subject | score only on all-modality recordings |

The second one: audio was only captured for the 7 speech tasks, which are mostly
the stressful ones. `P(stress | audio present) = 0.709` vs
`P(stress | audio absent) = 0.311`. So "does an audio file exist" predicts the
label at 0.697 macro F1 with zero signal content — beating a 1.2 M-parameter
transformer. Re-verified on the freshly re-downloaded corpus 2026-08-04:
unchanged, because it is structural.

**Neither issue is a defect in the downloaded files.** The dataset was
re-downloaded on 2026-08-04 and verified byte-identical to the previous copy
(777 physiological files matching by MD5; Audio/Videos/label CSVs matching by
name, size and hash). Low scores are the honest protocol working correctly.

---

## 7. Correction to earlier documentation

Earlier prose in `RESEARCH_PROGRESS.md` §10.3 and
`research_way/RESULTS_AND_COMPARISON.md` §3.2 reported physio leakage inflation as
**+0.127 weighted F1 (0.670 → 0.543)**.

**That figure does not reproduce and is not supported by any stored artifact.**
Checked against every run on disk:

| Source | physio + RF inflation |
|---|---|
| `results/full/e1_leakage_inflation.csv` | +0.093 |
| `results/full_run1_20260729/e1_leakage_inflation.csv` | +0.093 |
| `prove_leakage.py`, fresh run 2026-08-06 | **+0.098** |
| *prose claim in §10.3* | *+0.127 — unsupported* |

The leakage-free value (0.543) is correct; the leaky value (0.670) is not — no run
produced it. The reproducible figure is **+0.098 macro F1 / +0.098 weighted F1**.

The qualitative conclusion is unchanged — physiology still inflates most, by a
wide margin, and the negative control still sits at zero. But the effect is
**~23% smaller than previously written**, and the corrected number is what should
be quoted. The affected prose has been fixed in place.

---

## 8. Files

| Path | What |
|---|---|
| `research_way/prove_leakage.py` | regenerates everything here |
| `research_way/reports/leakage_proof/protocol_swap.csv` | E-A/E-B raw scores, both protocols |
| `research_way/reports/leakage_proof/inflation.csv` | leaky − leakage-free, per model/feature |
| `research_way/reports/leakage_proof/mechanism.json` | E-C probe, E-D oracle, overlap counts |
| `research_way/src/splits.py` | the leakage-free protocol + its assertions |
| `research_way/src/baselines.py` | classical baselines under both protocols |
