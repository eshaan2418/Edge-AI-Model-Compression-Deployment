from __future__ import annotations

import argparse

import torch

from .distillation import DistillConfig, run_distillation
from .models import create_model
from .pruning import apply_global_unstructured_pruning, remove_pruning_reparametrization
from .quantization import dynamic_quantize_linear_layers
from .train import TrainConfig
from .train import main as train_main


def _cmd_train(args: argparse.Namespace) -> None:
    cfg = TrainConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        max_batches=args.max_batches,
    )
    train_main(cfg)


def _cmd_prune(args: argparse.Namespace) -> None:
    model = create_model()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model = apply_global_unstructured_pruning(model, amount=args.amount)
    model = remove_pruning_reparametrization(model)
    torch.save({"model_state": model.state_dict()}, args.output)
    print(f"Saved pruned model to {args.output}")


def _cmd_quantize(args: argparse.Namespace) -> None:
    model = create_model()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    qmodel = dynamic_quantize_linear_layers(model)
    torch.save({"model_state": qmodel.state_dict()}, args.output)
    print(f"Saved dynamically quantized model to {args.output}")


def _cmd_distill(args: argparse.Namespace) -> None:
    cfg = DistillConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        teacher_ckpt=args.teacher,
    )
    run_distillation(cfg)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mc", description="Model compression CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    pt = sub.add_parser("train", help="Train baseline model")
    pt.add_argument("--data-dir", default="data")
    pt.add_argument("--batch-size", type=int, default=128)
    pt.add_argument("--epochs", type=int, default=1)
    pt.add_argument("--lr", type=float, default=0.1)
    pt.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Cap batches per train/eval loop for a fast smoke test.",
    )
    pt.set_defaults(func=_cmd_train)

    # prune
    pp = sub.add_parser("prune", help="Prune a checkpointed model")
    pp.add_argument("--checkpoint", required=True)
    pp.add_argument("--amount", type=float, default=0.5)
    pp.add_argument("--output", default="models/pruned.pt")
    pp.set_defaults(func=_cmd_prune)

    # quantize
    pq = sub.add_parser("quantize", help="Dynamically quantize a checkpointed model")
    pq.add_argument("--checkpoint", required=True)
    pq.add_argument("--output", default="models/quantized_dynamic.pt")
    pq.set_defaults(func=_cmd_quantize)

    # distill
    pd = sub.add_parser("distill", help="Knowledge distillation from a teacher checkpoint")
    pd.add_argument("--teacher", required=True)
    pd.add_argument("--data-dir", default="data")
    pd.add_argument("--batch-size", type=int, default=128)
    pd.add_argument("--epochs", type=int, default=1)
    pd.add_argument("--lr", type=float, default=0.05)
    pd.set_defaults(func=_cmd_distill)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
