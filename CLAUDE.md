# rag-retrieval-gate

Regression gate for RAG retrieval. Scores a baseline and a candidate retrieval config on the same
labelled questions (hit@k, MRR@k, recall@k, nDCG@k, required context, hard negatives, citation checks)
and exits 1 when a metric drops by more than `--max-drop` AND a paired bootstrap 95% CI (10k resamples,
seed 0) excludes zero. Also records build time, p50/p95 latency, peak RSS and model size (not gated).
See README.md for usage.

## Layout
- `src/rag_gate/` — data (fixture + BEIR loaders), retrievers (BM25, dense, hybrid RRF, cross-encoder
  re-rank, chunking), metrics, stats (paired bootstrap), gate (run + cost profiling, compare), cli.
- `configs/` — retrieval configs. `baseline.json` is the reference; CI gates `candidate-bm25-b05.json`
  (must pass) and `candidate-chunk-12.json` (must fail with `--no-bootstrap`, as a self-test; six
  questions give the bootstrap no power, so the fixture is threshold-only).
- `fixtures/policies.json` — committed offline fixture. `data/` and `reports/` are git-ignored.
- `scripts/fetch_beir.py` — SciFact download (UKP zip, or `--source huggingface` mirror needing pyarrow).
- `scripts/results_table.py` — runs each config in its own process and prints the README results table.

## Constraints
- Core path is standard library only; numpy / sentence-transformers are optional extras. Keep it that way.
- Dense models load from local cache unless `--allow-download`. Never download silently.
- Metrics are about document IDs; chunks collapse to their parent at its best rank.
- Every number in README.md must be reproducible by a command shown next to it. Re-run and update
  the tables when retriever or metric code changes.
- No tuning on test queries. RRF k=60, fusion/rerank depth 100 are fixed literature defaults. Any new
  tunable must be chosen a priori or on a non-test split, and the README must say which.
- Models run on CPU by default (`--device cpu`) so latency numbers are comparable.
- Never commit BEIR data. Fetcher checksums are pinned; update them only after verifying the source.

## Commands
- `pip install -e ".[dev]"` then `python -m pytest -q`
- `rag-gate gate --dataset fixtures/policies.json --k 3 --baseline configs/baseline.json --candidate configs/<c>.json`
- `rag-gate gate --dataset data/beir/scifact --k 10 ...` after `python scripts/fetch_beir.py --acknowledge-licence`
