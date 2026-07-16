# data/

Datasets are downloaded here on first run and are git-ignored (see
`.gitignore`) except this README.

- **CIFAR-10 / CIFAR-100** — downloaded automatically by torchvision
  (PyTorch framework) or `tf.keras.datasets` (TensorFlow scripts) the first
  time you run training or a benchmark.
- **Tiny ImageNet** — not auto-downloaded. Place it here as
  `data/tiny-imagenet-200/{train,val}` in `ImageFolder` layout if you use
  `dataset: tiny_imagenet`.

Nothing in this directory needs to be committed; it is all regenerable.
