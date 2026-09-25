# Small-LM track (Phase 8, stretch)

Design: [`DECISIONS.md`](DECISIONS.md) (Phase 8, D8.x). No results are quoted here.

**Question:** does compression degrade post-trained *behavior* before raw *capability*?

| Piece | Module | Notes |
|---|---|---|
| Tokenizer | `lm/tokenizer.py` | byte-level (256 ids + PAD/BOS/EOS/SEP), offline, nothing to version |
| Data | `lm/data.py` | TinyStories, TinyStories-Instruct (`Words:` constraint), synthetic stories for tests, SFT masking, preference pairs (rejected = required words swapped out) |
| Model | `lm/model.py` | pre-norm GPT, explicit qkv/proj/fc1/fc2 (quantizable), tied embeddings, KV cache; `gpt_10m` / `gpt_25m` / `gpt_50m` |
| Training | `lm/train.py` | `pretrain`, `sft` (response-only loss), `distill` (logit KD, the speculative draft), `dpo` (frozen reference); `lm_runs.csv` with parent run ids |
| Metrics | `lm/eval.py` | perplexity (capability), constraint rate (behavior), TTFT / tokens/s / KV-cache bytes, speculative decoding (lossless for greedy, acceptance rate) |
| Ladder | `lm/compress.py` | fp, W8A8, W8, W4-g32, W4A8, W3-g32 on block linears; `lm_evals.csv` |
| Pipeline | `lm/pipeline.py` | one YAML, resumable: pretrain → SFT → DPO → draft → ladder on each stage |
| Analysis | `analysis/lm_degradation.py` | per-rung capability and behavior relative to float, gap with a seed-bootstrap CI; in `reproduce.sh` and the paper |

```bash
python -m edge_ai_compression.lm.pipeline --config edge_ai_compression/configs/lm/smoke.yml      # seconds, CPU
python -m edge_ai_compression.lm.pipeline --config edge_ai_compression/configs/lm/tinystories_gpt_10m.yml --set seed=0
```

Caveats:
- Quantized metrics come from fake-quant simulation, so only the float and speculative rows'
  speed numbers are meaningful.
- Byte-level tokens make sequences about 4× longer than BPE, so rates are per byte-token.
- The behavior metric is narrow: it checks for required words, not story quality.

**Results:** [PENDING: run `configs/lm/tinystories_gpt_*.yml` for 3 seeds (notebooks/tracks.ipynb),
then `./reproduce.sh`]
