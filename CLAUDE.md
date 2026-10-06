# rag-retrieval-gate

Regression gate for RAG retrieval. Scores a baseline and a candidate retrieval config on the same
labelled questions (hit@k, MRR@k, recall@k, nDCG@k, required context, hard negatives, citation checks)
and exits 1 when any metric is worse than `--max-drop`. See README.md for usage.

## Layout
- `src/rag_gate/` — data (fixture + BEIR loaders), retrievers (BM25, dense, chunking), metrics, gate, cli.
- `configs/` — retrieval configs. `baseline.json` is the reference; CI gates `candidate-bm25-b05.json`
  (must pass) and `candidate-chunk-12.json` (must fail, as a self-test).
- `fixtures/policies.json` — committed offline fixture. `data/` and `reports/` are git-ignored.
- `scripts/fetch_beir.py` — SciFact download (UKP zip, or `--source huggingface` mirror needing pyarrow).

## Constraints
- Core path is standard library only; numpy / sentence-transformers are optional extras. Keep it that way.
- Dense models load from local cache unless `--allow-download`. Never download silently.
- Metrics are about document IDs; chunks collapse to their parent at its best rank.
- Every number in README.md must be reproducible by a command shown next to it. Re-run and update
  the tables when retriever or metric code changes.
- Never commit BEIR data. Fetcher checksums are pinned; update them only after verifying the source.

## Commands
- `pip install -e ".[dev]"` then `python -m pytest -q`
- `rag-gate gate --dataset fixtures/policies.json --k 3 --baseline configs/baseline.json --candidate configs/<c>.json`
- `rag-gate gate --dataset data/beir/scifact --k 10 ...` after `python scripts/fetch_beir.py --acknowledge-licence`
