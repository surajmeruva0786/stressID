# R9 - does whole-recording physiology (frequency-domain HRV above all) add
# anything the 10 s window features could not carry?
#
# Identical to R8a in every respect -- same seed, same folds, same views, same
# window and sequence candidates, same pruning, and deliberately the SAME EIGHT
# feature-set names -- so the only difference is that the three physio-bearing
# sets now also contain the physglobal block. The two extra diagnostic sets
# (physglobal alone, phys_window_only) are excluded here precisely because
# adding candidates would confound the comparison with the thing being tested.
#
# repeats=1 gives 5 folds, and because the splitter is seeded identically those
# five folds ARE R8a's first five, so the pairing is exact and cheap. If it
# looks promising it gets extended to 15.
#
# SOTA_JOBS=4: the affect3/regression runs are using the rest of the box.

Set-Location "F:\stressID\research_way"
$env:SOTA_JOBS = "4"

$note = "R9: adds the physglobal block (frequency-domain HRV LF/HF/LF-HF, whole-recording time-domain HRV, EDA tonic-phasic, respiration) to the three physio-bearing feature sets. Everything else identical to R8a including the seed, so folds pair exactly. Duration-dependent counts were removed from this block first -- duration alone scores 0.5923 macro F1 on StressID and is a task label in disguise."

python -u -m src.sota --run-name s9_physglobal --views raw,rel,z --repeats 1 `
    --seed 101 --fast --bags 12 --max-size 25 --cum-keep 0.90 --scopes all700 `
    --feature-sets phys,audio,video,audio+video,phys+video,phys+audio,all,all+avail `
    --windows raw,rel --torch-archs gru,attn --n-par 1 --inner-folds 3 `
    --notes $note 2>&1 | Out-File -Encoding utf8 "F:\stressID\research_way\r9.log"

"r9 complete" | Out-File "F:\stressID\research_way\r9.done"
