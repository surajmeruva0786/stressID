# R9c - the ABLATION ARM for the physglobal question, under byte-identical
# settings to R9b.
#
# Why this run exists. R9 was compared against R8a on the assumption that the
# two differed only in the physglobal block. They did not: R8a ran with
# SOTA_JOBS=10 / n_par=1 (10 threads per model) and R9 with SOTA_JOBS=4 /
# n_par=1 (4 threads). Thread count changes floating-point summation order
# inside XGBoost and LightGBM, which changes the trees, which changes the
# predictions. Proof came from R9b itself: at 2 threads it scored 0.8067 on the
# fold where R9 at 4 threads scored 0.7996, with everything else identical.
#
# So the +0.0075 from R9 was confounded by compute settings. This arm runs the
# same 15 folds, the same seed, the same SOTA_JOBS=6 / n_par=3, and differs from
# R9b in exactly one respect: --exclude-blocks physglobal. Comparing R9c against
# R9b is then a real ablation.

Set-Location "F:\stressID\research_way"
$env:SOTA_JOBS = "6"

$note = "R9c: the physglobal ABLATION arm. Byte-identical to R9b -- same seed, same folds, same feature-set names, same thread budget -- except the physglobal block is dropped from every feature set. Exists because the earlier R9-vs-R8a comparison differed in thread count as well as features, and thread count alone changes booster predictions."

python -u -m src.sota --run-name s9c_ablate_physglobal --views raw,rel,z --repeats 3 `
    --seed 101 --fast --bags 12 --max-size 25 --cum-keep 0.90 --scopes all700 `
    --feature-sets phys,audio,video,audio+video,phys+video,phys+audio,all,all+avail `
    --exclude-blocks physglobal `
    --windows raw,rel --torch-archs gru,attn --n-par 3 --inner-folds 3 `
    --notes $note 2>&1 | Out-File -Encoding utf8 "F:\stressID\research_way\r9c.log"

"r9c complete" | Out-File "F:\stressID\research_way\r9c.done"
