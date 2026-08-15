# R9b - settle the physglobal question at 15 folds.
#
# At 5 folds R9 gave +0.0075 with paired p=0.051 and CI [-0.0001, +0.0151]:
# four wins, one exact tie, no losses. That is the same underpowered signature
# R6/R7 showed before R8 resolved it at fifteen folds, so it gets the same
# treatment rather than a verdict either way.
#
# Seeded identically to R8a, so folds 0-4 reproduce R9 exactly and folds 5-14
# are new. Pairs against s8a_final_r3 fold for fold.

Set-Location "F:\stressID\research_way"
# The first attempt at SOTA_JOBS=8 died with MemoryError at fold 3, while the
# regression run still held the rest of the box. That has finished, leaving
# 10 GB free and an idle machine, so this attempt uses parallel candidates:
# 3 worker processes x 2 threads each. Folds 0-2 reproduced exactly before the
# crash, so re-running them costs time but risks nothing.
$env:SOTA_JOBS = "6"

$note = "R9b: physglobal at 3 repeats (15 folds), pairing fold-for-fold against s8a_final_r3. Settles whether frequency-domain HRV helps, after the 5-fold read came in at +0.0075 with p=0.051 and no fold worse."

python -u -m src.sota --run-name s9b_physglobal_r3 --views raw,rel,z --repeats 3 `
    --seed 101 --fast --bags 12 --max-size 25 --cum-keep 0.90 --scopes all700 `
    --feature-sets phys,audio,video,audio+video,phys+video,phys+audio,all,all+avail `
    --windows raw,rel --torch-archs gru,attn --n-par 3 --inner-folds 3 `
    --notes $note 2>&1 | Out-File -Encoding utf8 "F:\stressID\research_way\r9b.log"

"r9b complete" | Out-File "F:\stressID\research_way\r9b.done"
