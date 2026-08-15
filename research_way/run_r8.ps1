# R8 - properly powered paired comparison of the final configuration against the
# baseline, both at 3 repeats (15 outer folds) on unseen seed-101 partitions.
#
# The 5-fold comparison in R6/R7 gave +0.0195 macro F1 at paired p=0.22: the
# improvement replicated on every metric and both scopes, but 5 folds have
# almost no statistical power and the mean was pulled by a single +0.064 fold.
# 15 folds is the cheapest way to find out whether that gain is real.
#
# The two runs are sequential, not parallel, on purpose: this box also runs the
# user's other training jobs, and two concurrent sweeps push it into swap.

Set-Location "F:\stressID\research_way"

$noteA = "R8a: final configuration at 3 repeats (15 outer folds) on unseen seed-101 partitions, for a powered paired test against the baseline."
$noteB = "R8b: R1 baseline configuration at 3 repeats (15 outer folds), same seed and partitions as R8a."

python -u -m src.sota --run-name s8a_final_r3 --views raw,rel,z --repeats 3 `
    --seed 101 --fast --bags 12 --max-size 25 --cum-keep 0.90 --scopes all700 `
    --windows raw,rel --torch-archs gru,attn --n-par 1 --inner-folds 3 `
    --notes $noteA *> "F:\stressID\research_way\r8a.log"

python -u -m src.sota --run-name s8b_base_r3 --views raw --repeats 3 `
    --seed 101 --fast --bags 12 --max-size 25 --scopes all700 `
    --n-par 1 --inner-folds 3 `
    --notes $noteB *> "F:\stressID\research_way\r8b.log"

"R8 chain complete" | Out-File "F:\stressID\research_way\r8.done"
