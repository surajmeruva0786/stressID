# The other two StressID targets, run back to back.
#
# Configuration mirrors what the binary campaign established: raw + rel views,
# window-level candidates, bagged greedy selection on inner OOF, 90% pruning.
# Seed 101 -- the partition family the binary headline was confirmed on, and one
# the binary search never tuned against.
#
# SOTA_JOBS is raised to 10 because the user's other training jobs have
# finished; the 6-core cap existed to leave them half the machine.

Set-Location "F:\stressID\research_way"
$env:SOTA_JOBS = "10"

$noteA = "T1: 3-class affect (negative / neutral / positive-arousal), 253/202/245. Same recipe as the binary campaign - raw+rel views, window-level candidates, bagged greedy on inner OOF, 90% pruning. Greedy maximises macro F1 over the argmax."
$noteB = "T2: stress-score regression, 0-10 continuous. Same recipe; greedy maximises negative RMSE. Reported with RMSE, MAE, Pearson and Spearman correlation, and R2."

python -u -m src.sota_tasks --task affect3 --run-name t1_affect3 `
    --views raw,rel --windows raw,rel --seed 101 --bags 12 --max-size 25 `
    --cum-keep 0.90 --inner-folds 3 --notes $noteA 2>&1 |
    Out-File -Encoding utf8 "F:\stressID\research_way\t1.log"

python -u -m src.sota_tasks --task regression --run-name t2_score `
    --views raw,rel --windows raw,rel --seed 101 --bags 12 --max-size 25 `
    --cum-keep 0.90 --inner-folds 3 --notes $noteB 2>&1 |
    Out-File -Encoding utf8 "F:\stressID\research_way\t2.log"

"tasks complete" | Out-File "F:\stressID\research_way\tasks.done"
