"""CLI: list available compression recipes.

python -m edge_ai_compression.recipes.list
"""

from __future__ import annotations

import argparse

from edge_ai_compression.recipes.catalog import available_recipes, get_recipe


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="python -m edge_ai_compression.recipes.list", description=__doc__
    )
    p.parse_args(argv)

    names = available_recipes()
    width = max(len(n) for n in names)
    print(f"{len(names)} recipes:\n")
    for name in names:
        r = get_recipe(name)
        print(f"  {name:<{width}}  {r.description}")
    print("\nRun `... recipes.show <name>` for full details.")


if __name__ == "__main__":
    main()
