"""CLI: show one compression recipe's full metadata.

python -m edge_ai_compression.recipes.show raspberry_pi_fast
"""

from __future__ import annotations

import argparse

from edge_ai_compression.recipes.catalog import available_recipes, get_recipe


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m edge_ai_compression.recipes.show", description=__doc__
    )
    p.add_argument("recipe", help=f"Recipe name. One of: {', '.join(available_recipes())}.")
    args = p.parse_args(argv)

    try:
        r = get_recipe(args.recipe)
    except KeyError as exc:
        print(str(exc))
        return 1

    print(f"Recipe        : {r.name}")
    print(f"Description   : {r.description}")
    print(f"When to use   : {r.when_to_use}")
    print(f"Pruning       : amount={r.pruning_amount}, mode={r.pruning_mode}")
    print(f"Quantization  : {'on' if r.quantization else 'off'} ({r.quantization_mode})")
    print(f"Distillation  : {'on' if r.distillation else 'off'}")
    print(f"Hardware      : {r.hardware_profile}")
    print(f"Export format : {r.export_format}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
