# Full Dataset RAG Batch Summary

## Record Counts

- Input records: `219`
- Analyzable paired records: `219`
- Status counts: `{'ok': 219}`

## Primary Paired Results

Mean diff is `vectorless_tree_rag - embedding_rag`. For hallucination, negative favors vectorless.

| Group | N | Metric | Emb mean | Vec mean | Diff | 95% bootstrap CI | sign-test p | Vec better | Ties | Emb better |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|
| overall | 219 | feature_coverage | 0.9772 | 0.9840 | 0.0068 | [-0.0080, 0.0228] | 0.3449 | 17 | 191 | 11 |
| overall | 219 | faithfulness | 0.9726 | 0.9794 | 0.0068 | [-0.0091, 0.0231] | 0.8238 | 11 | 199 | 9 |
| overall | 219 | hallucination_rate | 0.0274 | 0.0206 | -0.0068 | [-0.0231, 0.0091] | 0.8238 | 11 | 199 | 9 |
| overall | 219 | citation_accuracy | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] | 1.0000 | 0 | 219 | 0 |
| overall | 219 | claim_count | 2.6530 | 2.6758 | 0.0228 | [-0.0411, 0.0959] | 0.4101 | 30 | 166 | 23 |
| overall | 219 | quality_score | 2.9224 | 2.9428 | 0.0204 | [-0.0174, 0.0582] | 0.3713 | 26 | 174 | 19 |
| dataset=GAITEX \| model=llama3.1:8b | 73 | feature_coverage | 0.9829 | 0.9760 | -0.0068 | [-0.0377, 0.0240] | 0.7266 | 3 | 65 | 5 |
| dataset=GAITEX \| model=llama3.1:8b | 73 | faithfulness | 0.9178 | 0.9381 | 0.0203 | [-0.0290, 0.0694] | 0.8238 | 11 | 53 | 9 |
| dataset=GAITEX \| model=llama3.1:8b | 73 | hallucination_rate | 0.0822 | 0.0619 | -0.0203 | [-0.0696, 0.0288] | 0.8238 | 11 | 53 | 9 |
| dataset=GAITEX \| model=llama3.1:8b | 73 | citation_accuracy | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] | 1.0000 | 0 | 73 | 0 |
| dataset=GAITEX \| model=llama3.1:8b | 73 | claim_count | 2.7123 | 2.8356 | 0.1233 | [-0.0137, 0.2603] | 0.1078 | 17 | 48 | 8 |
| dataset=GAITEX \| model=llama3.1:8b | 73 | quality_score | 2.8185 | 2.8523 | 0.0338 | [-0.0742, 0.1477] | 1.0000 | 12 | 48 | 13 |
| dataset=GAITEX \| model=mistral-small:24b | 73 | feature_coverage | 0.9726 | 0.9760 | 0.0034 | [-0.0240, 0.0342] | 1.0000 | 7 | 60 | 6 |
| dataset=GAITEX \| model=mistral-small:24b | 73 | faithfulness | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] | 1.0000 | 0 | 73 | 0 |
| dataset=GAITEX \| model=mistral-small:24b | 73 | hallucination_rate | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 1.0000 | 0 | 73 | 0 |
| dataset=GAITEX \| model=mistral-small:24b | 73 | citation_accuracy | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] | 1.0000 | 0 | 73 | 0 |
| dataset=GAITEX \| model=mistral-small:24b | 73 | claim_count | 2.8219 | 2.6986 | -0.1233 | [-0.2603, 0.0000] | 0.1671 | 6 | 54 | 13 |
| dataset=GAITEX \| model=mistral-small:24b | 73 | quality_score | 2.9726 | 2.9760 | 0.0034 | [-0.0240, 0.0342] | 1.0000 | 7 | 60 | 6 |
| dataset=GAITEX \| model=qwen3:14b | 73 | feature_coverage | 0.9760 | 1.0000 | 0.0240 | [0.0103, 0.0411] | 0.0156 | 7 | 66 | 0 |
| dataset=GAITEX \| model=qwen3:14b | 73 | faithfulness | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] | 1.0000 | 0 | 73 | 0 |
| dataset=GAITEX \| model=qwen3:14b | 73 | hallucination_rate | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 1.0000 | 0 | 73 | 0 |
| dataset=GAITEX \| model=qwen3:14b | 73 | citation_accuracy | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] | 1.0000 | 0 | 73 | 0 |
| dataset=GAITEX \| model=qwen3:14b | 73 | claim_count | 2.4247 | 2.4932 | 0.0685 | [-0.0137, 0.1507] | 0.1797 | 7 | 64 | 2 |
| dataset=GAITEX \| model=qwen3:14b | 73 | quality_score | 2.9760 | 3.0000 | 0.0240 | [0.0103, 0.0411] | 0.0156 | 7 | 66 | 0 |

## Retrieval Frequencies

### overall

Top embedding contexts:
- `gaitex_dataset_motion_capture_provenance`: 219
- `gaitex_repeated_task_context`: 219
- `gaitex_speed_protocol_context`: 108

Top vectorless nodes:
- `gaitex_dataset_motion_capture_provenance`: 219
- `gaitex_repeated_task_context`: 219
- `gaitex_speed_protocol_context`: 108

### dataset=GAITEX | model=llama3.1:8b

Top embedding contexts:
- `gaitex_dataset_motion_capture_provenance`: 73
- `gaitex_repeated_task_context`: 73
- `gaitex_speed_protocol_context`: 36

Top vectorless nodes:
- `gaitex_dataset_motion_capture_provenance`: 73
- `gaitex_repeated_task_context`: 73
- `gaitex_speed_protocol_context`: 36

### dataset=GAITEX | model=mistral-small:24b

Top embedding contexts:
- `gaitex_dataset_motion_capture_provenance`: 73
- `gaitex_repeated_task_context`: 73
- `gaitex_speed_protocol_context`: 36

Top vectorless nodes:
- `gaitex_dataset_motion_capture_provenance`: 73
- `gaitex_repeated_task_context`: 73
- `gaitex_speed_protocol_context`: 36

### dataset=GAITEX | model=qwen3:14b

Top embedding contexts:
- `gaitex_dataset_motion_capture_provenance`: 73
- `gaitex_repeated_task_context`: 73
- `gaitex_speed_protocol_context`: 36

Top vectorless nodes:
- `gaitex_dataset_motion_capture_provenance`: 73
- `gaitex_repeated_task_context`: 73
- `gaitex_speed_protocol_context`: 36

## Notes

- All comparisons are paired within the same case and LLM.
- Mean difference is vectorless_tree_rag minus embedding_rag.
- For hallucination_rate, negative mean difference favors vectorless_tree_rag because lower is better.
- Sign-test p-values are exact two-sided tests over non-tied paired differences.
- Bootstrap CIs are percentile intervals for the mean paired difference.
- Records with status=no_features are counted but excluded from inferential paired tests.
