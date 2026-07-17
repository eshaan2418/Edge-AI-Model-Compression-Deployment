"""Named compression recipes that compose existing pipeline pieces."""

from edge_ai_compression.recipes.catalog import Recipe, available_recipes, get_recipe

__all__ = ["Recipe", "available_recipes", "get_recipe"]
