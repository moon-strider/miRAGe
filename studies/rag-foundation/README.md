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

## Wave 2 embedding-model results

All rows use the same full SciFact split, Qdrant HNSW cosine store, token 1024/128 chunks, and `search-dense-topk5-v1` so the isolated variable is the embedding model.

| embedding model | Hit@k | Precision@k | Recall@k | MRR@k | NDCG@k | p50 ms | p95 ms | projected 1m query cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| text-embedding-3-small | 0.8000 | 0.1727 | 0.7771 | 0.6736 | 0.6912 | 846.58 | 1261.62 | 0.23 |
| text-embedding-3-large | 0.8433 | 0.1847 | 0.8292 | 0.7173 | 0.7390 | 938.26 | 1419.79 | 2.58 |
| text-embedding-ada-002 | 0.7733 | 0.1693 | 0.7598 | 0.6657 | 0.6838 | 834.32 | 1225.73 | 2.01 |
| pplx-embed-v1-0.6b | 0.7967 | 0.1767 | 0.7841 | 0.6789 | 0.7010 | 887.52 | 1321.75 | 0.00 |
| pplx-embed-v1-4b | 0.8400 | 0.1865 | 0.8289 | 0.7112 | 0.7351 | 821.12 | 1234.36 | 0.60 |
| gemini-embedding-001 | 0.9567 | 0.2147 | 0.9509 | 0.8688 | 0.8869 | 977.78 | 1181.49 | 3.52 |

## Wave 2 interpretation

- `gemini-embedding-001` is the clear winner among completed embedding experiments.
- Compared with `text-embedding-3-small`, Gemini improves Recall@k from 0.7771 to 0.9509 and NDCG@k from 0.6912 to 0.8869.
- `text-embedding-3-large` and `pplx-embed-v1-4b` are close; `pplx-4b` has slightly lower NDCG but better p50/p95 latency and lower projected query cost.
- `text-embedding-ada-002` is worse than the current baseline and should not be used for the next retrieval sweeps.
- After the embedding sweep, chunking/search sweeps use `gemini-embedding-001` as the fixed embedding baseline.

## Wave 3 LLM-free macro chunking results

All completed rows use full SciFact, Gemini embeddings, Qdrant HNSW cosine, dense top-k5, no reranker, no tool policy, and no LLM generation.

| chunking | Hit@k | Precision@k | Recall@k | MRR@k | NDCG@k | p50 ms | p95 ms | projected 1m query cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| token 1024/128 | 0.9567 | 0.2147 | 0.9509 | 0.8688 | 0.8869 | 977.78 | 1181.49 | 3.52 |
| token 512/64 | 0.9533 | 0.2261 | 0.9476 | 0.8688 | 0.8861 | 1102.24 | 1282.03 | 3.52 |
| token 2048/256 | 0.9567 | 0.2140 | 0.9509 | 0.8688 | 0.8869 | 1037.36 | 1229.87 | 3.52 |
| sentence 1024/128 | 0.9567 | 0.2147 | 0.9509 | 0.8688 | 0.8869 | 1000.22 | 1206.27 | 3.52 |

## Wave 3 interpretation

- Token 1024/128 remains the preferred baseline: it ties the best Recall@k and NDCG@k while keeping the lowest latency among tied variants.
- Token 512/64 slightly improves Precision@k, but loses Recall@k/NDCG@k and is slower, so it is not a better general baseline.
- Token 2048/256 and sentence 1024/128 match quality but add latency, so they do not justify replacing the simpler token 1024/128 baseline.
- The old embedding-threshold splitter is no longer treated as true semantic chunking; true LLM semantic chunking is reported separately below.

## Wave 4 LLM-free macro search results

All rows use full SciFact, Gemini embeddings, token 1024/128 chunks, Qdrant HNSW cosine, no reranker, no tool policy, and no LLM generation.

| search | Hit@k | Precision@k | Recall@k | MRR@k | NDCG@k | p50 ms | p95 ms | projected 1m query cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense top-k5 | 0.9567 | 0.2147 | 0.9509 | 0.8688 | 0.8869 | 977.78 | 1181.49 | 3.52 |
| dense top-k10 | 0.9767 | 0.1108 | 0.9750 | 0.8716 | 0.8956 | 1092.50 | 1359.70 | 3.52 |
| dense MMR top-k10 | 0.9100 | 0.0987 | 0.8935 | 0.8394 | 0.8371 | 4381.35 | 5204.48 | 3.52 |
| hybrid RRF top-k10 | 0.9700 | 0.1094 | 0.9670 | 0.7479 | 0.7975 | 1974.43 | 2180.90 | 3.52 |

## Wave 4 interpretation

- Dense top-k10 is the best completed search variant: it improves Recall@k and NDCG@k over dense top-k5 with moderate latency increase.
- Dense MMR is worse on every quality metric and much slower, so it should not be used in the main path.
- Hybrid RRF improves Recall@k over top-k5 but loses badly on MRR@k/NDCG@k and is slower than dense top-k10, so lexical fusion is not useful for the SciFact main baseline.
- Current retrieval baseline: Gemini embeddings, token 1024/128 chunks, Qdrant HNSW cosine, dense top-k10, no reranker/tool policy/LLM generation.

## Wave 5 LLM semantic chunking results

This row uses full SciFact, Ollama Cloud Nemotron chunk-plan generation, Gemini embeddings, Qdrant HNSW cosine, dense top-k5, no reranker, no tool policy, and no answer generation. The chunk planner produced 5,183 cached plans and 5,510 materialized chunks.

| chunking | planner | Hit@k | Precision@k | Recall@k | MRR@k | NDCG@k | p50 ms | p95 ms | projected 1m query cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LLM semantic 1024 | nemotron-3-nano-30b | 0.9500 | 0.2248 | 0.9459 | 0.8726 | 0.8887 | 1165.05 | 2195.99 | 3.52 |

## Wave 5 interpretation

- LLM semantic chunking slightly improves MRR@k over dense top-k5 token chunking, but does not beat the current dense top-k10 baseline.
- Compared with the current baseline, NDCG@k is lower: 0.8887 vs 0.8956, and Recall@k is lower: 0.9459 vs 0.9750.
- Latency is worse than token top-k5 and top-k10, so this is not the SciFact default.
- SciFact is mostly short documents: 5,173 of 5,183 documents fit within the 1024-token chunk limit, so this dataset is not a strong test for semantic chunking.
- Next semantic chunking check should be Qasper, where longer paper-like documents make semantic boundaries more likely to matter.

## Wave 6 cross-dataset baseline results

All rows use the current SciFact retrieval baseline: Gemini embeddings, token 1024/128 chunks, Qdrant HNSW cosine, dense top-k10, no reranker, no tool policy, and no LLM generation.

| dataset | eval split | examples | Hit@k | Precision@k | Recall@k | MRR@k | NDCG@k | p50 ms | p95 ms | projected 1m query cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qasper | dev | 945 | 0.3556 | 0.0751 | 0.3556 | 0.2662 | 0.2878 | 782.61 | 928.60 | 1.95 |
| FiQA | test | 648 | 0.8441 | 0.1596 | 0.6559 | 0.6381 | 0.5681 | 754.05 | 923.06 | 2.00 |

## Wave 6 interpretation

- The SciFact baseline transfers reasonably to FiQA, but not to Qasper.
- Qasper is the weakest dataset for the current dense baseline and should be the priority for the next chunking and reranking checks.
- FiQA is strong enough to act as the second baseline dataset for reranker validation after Qasper-specific weakness is understood.
