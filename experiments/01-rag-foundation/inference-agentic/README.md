# inference-agentic

This experiment varies tool-use policy or orchestration style while keeping the
rest of the retrieval stack fixed.

## Moving variables

- `tool_policy_id`
- `reranker_id`

## Fixed variables

- preprocessing: `prep-basic-clean-v1`
- chunking: `chunk-token-500-50-v1`
- chunking model: `none`
- store embedding: `emb-bge-small-en-v1.5`
- query embedding: `emb-bge-small-en-v1.5`
- prompt variant: `prompt-grounded-citations-v1`

## Runtime status

- `none` remains the baseline single-pass path
- `tool-context-expansion-v1` runs one bounded follow-up retrieval with larger `top_k`
- `tool-react-v1` runs one bounded follow-up retrieval with a rewritten evidence-seeking query
- rerankers are all ONNX-backed fastembed cross-encoders
- `rerank-minilm-l6-v1`: smallest MiniLM baseline
- `rerank-minilm-l12-v1`: slightly larger MiniLM candidate for higher ranking quality
- `rerank-jina-tiny-v1`: fast long-context English reranker
- `rerank-jina-turbo-v1`: stronger long-context English reranker with slightly higher runtime cost
