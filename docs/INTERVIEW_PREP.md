# Interview prep

Five hard questions per phase, with concise answers.

---

## Phase 0: consolidation

**1. Why move the old code to `legacy/` instead of deleting it? When would you delete it?**
The TF/TFLite scripts are the only working int8 TFLite export path and a reference for the Phase 2 export work. Keeping them frozen and lint-checked costs nothing. Delete once the unified PyTorch → ONNX export path reproduces what they do.

**2. Why drop the `mc` console script instead of packaging `legacy`?**
Packaging it would install a top-level module named `legacy` (previously `src`) into site-packages, where it can shadow or collide with other packages. Legacy code is reference material, not a library, so it runs from the repo root.

**3. The framework's quantization only converts `nn.Linear`. What does that mean for the existing ResNet-18 results?**
ResNet-18 has one Linear layer (the classifier). Nearly all compute is in convolutions, which stay FP32, so "quantized ResNet-18" results so far are close to a no-op. Static per-channel conv quantization is a prerequisite for any quantization claim (Phase 3).

**4. What does the CI smoke experiment catch that unit tests don't, and what can't it catch?**
It catches integration breaks: config parsing, data loading, the compression pipeline, evaluation and DB logging all wired together. It can't catch numerical correctness or performance regressions, because it uses random data and tiny iteration counts on a shared runner.

**5. How do you make sure every number in the README traces to a run?**
The experiment DB is the only results output. Every row has an experiment ID, and artifacts (config, metrics, fingerprint, traces) are stored under that ID. Figures are generated from the DB by a script. No hand-typed numbers; missing results are marked `[PENDING: run <config>]`.
