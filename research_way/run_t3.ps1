# T3 - 3-class affect on c364, the scope that matches the origin paper's Table 3.
#
# The paper's multimodal baselines are evaluated on its 370 all-modality tasks,
# not the full corpus. Our c364 is the same subset. Comparing our all700 affect3
# number against the paper's Table 3 would repeat exactly the scope-mismatch
# error this campaign keeps catching, so this run fills the matching cell.

Set-Location "F:\stressID\research_way"
$env:SOTA_JOBS = "8"

$note = "T3: 3-class affect restricted to c364, matching the origin paper's 370-task multimodal evaluation scope."

python -u -m src.sota_tasks --task affect3 --run-name t3_affect3_c364 --scope c364 `
    --views raw,rel --windows raw,rel --seed 101 --bags 12 --max-size 25 `
    --cum-keep 0.90 --inner-folds 3 --notes $note 2>&1 |
    Out-File -Encoding utf8 "F:\stressID\research_way\t3.log"

"t3 complete" | Out-File "F:\stressID\research_way\t3.done"
