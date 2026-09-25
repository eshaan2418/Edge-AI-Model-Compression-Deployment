from __future__ import annotations

import torch

from edge_ai_compression.data.loaders import build_loaders, num_classes_for


def test_fake_loader_shapes_offline():
    # The `fake` dataset must work with no files and no network access.
    train_loader, test_loader = build_loaders(
        "fake", data_dir="unused", batch_size=8, num_workers=0, limit_samples=16
    )

    x, y = next(iter(train_loader))
    assert x.shape == (8, 3, 32, 32)
    assert x.dtype == torch.float32
    assert y.shape == (8,)
    assert int(y.max()) < 10 and int(y.min()) >= 0

    # limit_samples caps each split.
    assert len(train_loader.dataset) == 16
    assert len(test_loader.dataset) == 16
    assert num_classes_for("fake") == 10


def test_fake_loader_aliases():
    for alias in ("synthetic", "debug", "random"):
        train_loader, _ = build_loaders(
            alias, data_dir="unused", batch_size=4, num_workers=0, limit_samples=8
        )
        x, y = next(iter(train_loader))
        assert x.shape == (4, 3, 32, 32)
        assert y.shape == (4,)


def test_imagenet_layout_required_and_loaded(tmp_path):
    import pytest
    from PIL import Image

    from edge_ai_compression.data.loaders import build_loaders

    with pytest.raises(FileNotFoundError, match="ImageFolder layout"):
        build_loaders("imagenet", str(tmp_path))
    for split in ("train", "val"):
        for cls in ("n01", "n02"):
            d = tmp_path / "imagenet" / split / cls
            d.mkdir(parents=True)
            Image.new("RGB", (300, 260), color=(10, 20, 30)).save(d / "a.JPEG")
    train, val = build_loaders("imagenet", str(tmp_path), batch_size=2, num_workers=0)
    x, y = next(iter(val))
    assert x.shape == (2, 3, 224, 224) and set(y.tolist()) <= {0, 1}
    assert len(train.dataset) == 2
