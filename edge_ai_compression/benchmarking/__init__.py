"""Latency / memory measurement, statistics, and environment fingerprinting.

Intentionally import-free: the benchmark worker runs as
``python -m edge_ai_compression.benchmarking.worker`` and must not pull in torch
before it starts timing the cold start.
"""
