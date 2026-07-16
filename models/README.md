# models/

Trained checkpoints and exported artifacts land here at runtime. Everything in
this directory is git-ignored (see `.gitignore`) except this README, so large
model files are never committed.

Artifacts produced by the scripts:

| File / dir | Produced by | Framework |
|---|---|---|
| `baseline_resnet18.pt` | `python train.py` / `mc train` | PyTorch |
| `compressed.pt` | `python run_experiment.py ...` | PyTorch |
| `baseline_model/` (SavedModel) | `python train_baseline.py` | TensorFlow |
| `quantized_dynamic_range.tflite` | `python quantize_model.py` | TensorFlow |
| `quantized_integer_only.tflite` | `python quantize_model.py` | TensorFlow |
| `student_model/` (SavedModel) | `python distill_model.py` | TensorFlow |

Delete anything here freely; it is all regenerable.
