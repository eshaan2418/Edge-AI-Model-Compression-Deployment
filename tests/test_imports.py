from importlib import import_module


def test_imports():
    import_module("edge_ai_compression.core.pipeline")
    import_module("edge_ai_compression.core.runner")
    import_module("edge_ai_compression.optimization.search.bayesian_search")
    import_module("src.model_compression.datasets")
