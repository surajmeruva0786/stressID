# R11 - the comparison the campaign never made: elaborate pipeline vs a plain
# random forest on concatenated features, on IDENTICAL folds.
#
# This repo already contained "feature_fusion + RF" scoring 0.7366 macro F1
# under random 80/20 x 10. The campaign's final configuration scores 0.7484
# under subject-shared 5-fold x 3. Those are different protocols, so the
# apparent +0.012 is not a comparison at all -- and until it is made properly,
# there is no evidence that eleven rounds of feature engineering, window-level
# modelling, ensembling and threshold tuning bought anything over a baseline
# that took one line.
#
# Restricting to one feature set and one model makes the greedy ensemble
# degenerate to that single model, so this is a genuine simple baseline run
# through the identical harness, seed, folds and scoring.

Set-Location "F:\stressID\research_way"
$env:SOTA_JOBS = "6"

$noteA = "R11a: plain RF on concatenated features (feature set 'all', raw view, single model, no window candidates, no sequence models). The greedy ensemble degenerates to this one candidate. Same seed and folds as s8a_final_r3."
$noteB = "R11b: same as R11a but logreg instead of RF, as a second simple reference."

python -u -m src.sota --run-name s11a_simple_rf --views raw --repeats 3 `
    --seed 101 --fast --bags 4 --max-size 3 --cum-keep 1.0 --scopes all700,c364 `
    --feature-sets all --models rf --n-par 3 --inner-folds 3 `
    --notes $noteA 2>&1 | Out-File -Encoding utf8 "F:\stressID\research_way\r11a.log"

python -u -m src.sota --run-name s11b_simple_logreg --views raw --repeats 3 `
    --seed 101 --fast --bags 4 --max-size 3 --cum-keep 1.0 --scopes all700,c364 `
    --feature-sets all --models logreg --n-par 3 --inner-folds 3 `
    --notes $noteB 2>&1 | Out-File -Encoding utf8 "F:\stressID\research_way\r11b.log"

"r11 complete" | Out-File "F:\stressID\research_way\r11.done"
