# rag-foundation

Controlled baseline and retrieval-only RAG experiments.

## Files

- `study.toml` defines the baseline and all experiment variants.
- this README contains the human summary and completed results.

## Wave 1 retrieval-only results

| experiment | Hit@k | Precision@k | Recall@k | MRR@k | NDCG@k | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.8000 | 0.1727 | 0.7771 | 0.6742 | 0.6917 | 922.37 |
| dense-topk3 | 0.7467 | 0.2644 | 0.7224 | 0.6611 | 0.6669 | 911.00 |
| dense-topk10 | 0.8667 | 0.0976 | 0.8570 | 0.6826 | 0.7193 | 833.84 |
| dense-topk20 | 0.9000 | 0.0513 | 0.8977 | 0.6849 | 0.7301 | 919.28 |
| sparse-bm25 | 0.6867 | 0.0742 | 0.6688 | 0.5095 | 0.5438 | 1722.37 |
| hybrid-rrf | 0.8467 | 0.0935 | 0.8321 | 0.6530 | 0.6899 | 1788.07 |
| faiss-flat | 0.8000 | 0.1727 | 0.7771 | 0.6742 | 0.6917 | 961.00 |

## Interpretation

- `dense-topk20` is best for recall.
- `dense-topk10` is the best middle ground.
- `dense-topk3` improves precision but loses recall.
- `sparse-bm25` is worse than dense retrieval on SciFact.
- `hybrid-rrf` improves over baseline but loses to dense top-k10/top-k20.
- `faiss-flat` matches Qdrant quality, so Qdrant HNSW is not a quality confounder here.
