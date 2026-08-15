# R10 - does voice quality (F0, perturbation, HNR, alpha ratio, Hammarberg)
# add anything over the log-mel audio block?
#
# Run on c364, not all700, and that choice is deliberate. c364 has constant
# recording duration and constant modality availability, so the protocol
# confounds carry exactly zero information there -- a task-identity classifier
# scores 0.4176 on it, which IS the majority baseline. It is therefore the scope
# where a feature has to earn its keep on signal alone. It is also the scope
# where every recording actually has audio, so an audio ablation is testing
# something present in every row rather than in 54% of them.
#
# Both arms sequential, same SOTA_JOBS, same n_par, same seed, differing only in
# --exclude-blocks. R9 established that thread count perturbs booster
# predictions, so matched settings are not optional.

Set-Location "F:\stressID\research_way"
$env:SOTA_JOBS = "6"

$noteA = "R10a: voice-quality arm. audioglobal present. c364 scope, 15 folds, seed 101."
$noteB = "R10b: ablation arm, byte-identical to R10a except --exclude-blocks audioglobal."

python -u -m src.sota --run-name s10a_voice --views raw,rel,z --repeats 3 `
    --seed 101 --fast --bags 12 --max-size 25 --cum-keep 0.90 --scopes c364 `
    --feature-sets phys,audio,video,audio+video,phys+video,phys+audio,all,all+avail `
    --windows raw,rel --torch-archs gru,attn --n-par 3 --inner-folds 3 `
    --notes $noteA 2>&1 | Out-File -Encoding utf8 "F:\stressID\research_way\r10a.log"

python -u -m src.sota --run-name s10b_voice_ablate --views raw,rel,z --repeats 3 `
    --seed 101 --fast --bags 12 --max-size 25 --cum-keep 0.90 --scopes c364 `
    --feature-sets phys,audio,video,audio+video,phys+video,phys+audio,all,all+avail `
    --exclude-blocks audioglobal `
    --windows raw,rel --torch-archs gru,attn --n-par 3 --inner-folds 3 `
    --notes $noteB 2>&1 | Out-File -Encoding utf8 "F:\stressID\research_way\r10b.log"

"r10 complete" | Out-File "F:\stressID\research_way\r10.done"
