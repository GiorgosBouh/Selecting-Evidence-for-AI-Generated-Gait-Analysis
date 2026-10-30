# Full Dataset RAG Batch Summary

## Record Counts

- Input records: `780`
- Analyzable paired records: `540`
- Status counts: `{'ok': 540, 'no_features': 240}`

## Primary Paired Results

Mean diff is `vectorless_tree_rag - embedding_rag`. For hallucination, negative favors vectorless.

| Group | N | Metric | Emb mean | Vec mean | Diff | 95% bootstrap CI | sign-test p | Vec better | Ties | Emb better |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|
| overall | 540 | feature_coverage | 0.9748 | 0.9608 | -0.0141 | [-0.0275, -0.0023] | 0.0728 | 41 | 439 | 60 |
| overall | 540 | faithfulness | 0.9821 | 0.9727 | -0.0094 | [-0.0226, 0.0025] | 0.7202 | 33 | 470 | 37 |
| overall | 540 | hallucination_rate | 0.0179 | 0.0273 | 0.0094 | [-0.0026, 0.0225] | 0.7202 | 33 | 470 | 37 |
| overall | 540 | citation_accuracy | 1.0000 | 0.9870 | -0.0130 | [-0.0222, -0.0037] | 0.0156 | 0 | 533 | 7 |
| overall | 540 | claim_count | 4.4093 | 4.0056 | -0.4037 | [-0.5333, -0.2778] | 0.0000 | 67 | 300 | 173 |
| overall | 540 | quality_score | 2.9390 | 2.8932 | -0.0459 | [-0.0934, -0.0045] | 0.2898 | 48 | 432 | 60 |
| dataset=Clinical Gait Signals \| model=llama3.1:8b | 180 | feature_coverage | 0.9550 | 0.9349 | -0.0201 | [-0.0380, -0.0023] | 0.0300 | 19 | 125 | 36 |
| dataset=Clinical Gait Signals \| model=llama3.1:8b | 180 | faithfulness | 0.9772 | 0.9646 | -0.0126 | [-0.0296, 0.0053] | 0.0652 | 12 | 144 | 24 |
| dataset=Clinical Gait Signals \| model=llama3.1:8b | 180 | hallucination_rate | 0.0228 | 0.0354 | 0.0126 | [-0.0054, 0.0296] | 0.0652 | 12 | 144 | 24 |
| dataset=Clinical Gait Signals \| model=llama3.1:8b | 180 | citation_accuracy | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] | 1.0000 | 0 | 180 | 0 |
| dataset=Clinical Gait Signals \| model=llama3.1:8b | 180 | claim_count | 4.3111 | 4.0778 | -0.2333 | [-0.4722, 0.0000] | 0.0399 | 33 | 94 | 53 |
| dataset=Clinical Gait Signals \| model=llama3.1:8b | 180 | quality_score | 2.9094 | 2.8641 | -0.0453 | [-0.0928, 0.0016] | 0.3057 | 26 | 119 | 35 |
| dataset=Clinical Gait Signals \| model=mistral-small:24b | 180 | feature_coverage | 0.9853 | 0.9522 | -0.0331 | [-0.0677, -0.0032] | 0.0192 | 7 | 153 | 20 |
| dataset=Clinical Gait Signals \| model=mistral-small:24b | 180 | faithfulness | 0.9860 | 0.9593 | -0.0268 | [-0.0607, 0.0031] | 0.7905 | 6 | 166 | 8 |
| dataset=Clinical Gait Signals \| model=mistral-small:24b | 180 | hallucination_rate | 0.0140 | 0.0407 | 0.0268 | [-0.0032, 0.0604] | 0.7905 | 6 | 166 | 8 |
| dataset=Clinical Gait Signals \| model=mistral-small:24b | 180 | citation_accuracy | 1.0000 | 0.9611 | -0.0389 | [-0.0722, -0.0111] | 0.0156 | 0 | 173 | 7 |
| dataset=Clinical Gait Signals \| model=mistral-small:24b | 180 | claim_count | 4.3278 | 3.6500 | -0.6778 | [-0.9444, -0.4444] | 0.0000 | 18 | 87 | 75 |
| dataset=Clinical Gait Signals \| model=mistral-small:24b | 180 | quality_score | 2.9573 | 2.8318 | -0.1255 | [-0.2560, -0.0130] | 0.0192 | 7 | 153 | 20 |
| dataset=Clinical Gait Signals \| model=qwen3:14b | 180 | feature_coverage | 0.9841 | 0.9952 | 0.0110 | [0.0015, 0.0208] | 0.0192 | 15 | 161 | 4 |
| dataset=Clinical Gait Signals \| model=qwen3:14b | 180 | faithfulness | 0.9831 | 0.9942 | 0.0111 | [0.0011, 0.0209] | 0.0414 | 15 | 160 | 5 |
| dataset=Clinical Gait Signals \| model=qwen3:14b | 180 | hallucination_rate | 0.0169 | 0.0058 | -0.0111 | [-0.0210, -0.0011] | 0.0414 | 15 | 160 | 5 |
| dataset=Clinical Gait Signals \| model=qwen3:14b | 180 | citation_accuracy | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] | 1.0000 | 0 | 180 | 0 |
| dataset=Clinical Gait Signals \| model=qwen3:14b | 180 | claim_count | 4.5889 | 4.2889 | -0.3000 | [-0.4667, -0.1500] | 0.0003 | 16 | 119 | 45 |
| dataset=Clinical Gait Signals \| model=qwen3:14b | 180 | quality_score | 2.9504 | 2.9836 | 0.0332 | [0.0041, 0.0628] | 0.0414 | 15 | 160 | 5 |

## Retrieval Frequencies

### overall

Top embedding contexts:
- `clinical_dataset_biomechanical_feature_provenance`: 540
- `pathology_context_neurological_orthopedic_gait`: 507
- `perry_data2_phase_support_asymmetry`: 435
- `perry_data2_progression_speed_length`: 402
- `turning_performance_data2_context`: 78

Top vectorless nodes:
- `perry_data2_phase_support_asymmetry`: 470
- `perry_data2_progression_speed_length`: 381
- `pathology_context_neurological_orthopedic_gait`: 272
- `clinical_dataset_biomechanical_feature_provenance`: 258
- `turning_performance_data2_context`: 206
- `perry_velocity_cadence_timing`: 15
- `neurodegenerative_pd_gait_context`: 13
- `clinical_gait_analysis_general_context`: 10

### dataset=Clinical Gait Signals | model=llama3.1:8b

Top embedding contexts:
- `clinical_dataset_biomechanical_feature_provenance`: 180
- `pathology_context_neurological_orthopedic_gait`: 169
- `perry_data2_phase_support_asymmetry`: 145
- `perry_data2_progression_speed_length`: 134
- `turning_performance_data2_context`: 26

Top vectorless nodes:
- `perry_data2_phase_support_asymmetry`: 180
- `clinical_dataset_biomechanical_feature_provenance`: 148
- `perry_data2_progression_speed_length`: 131
- `pathology_context_neurological_orthopedic_gait`: 12
- `dataset_papers_root`: 1

### dataset=Clinical Gait Signals | model=mistral-small:24b

Top embedding contexts:
- `clinical_dataset_biomechanical_feature_provenance`: 180
- `pathology_context_neurological_orthopedic_gait`: 169
- `perry_data2_phase_support_asymmetry`: 145
- `perry_data2_progression_speed_length`: 134
- `turning_performance_data2_context`: 26

Top vectorless nodes:
- `perry_data2_phase_support_asymmetry`: 142
- `perry_data2_progression_speed_length`: 126
- `pathology_context_neurological_orthopedic_gait`: 117
- `turning_performance_data2_context`: 103
- `perry_velocity_cadence_timing`: 8
- `clinical_gait_analysis_general_context`: 8
- `neurodegenerative_pd_gait_context`: 8
- `clinical_dataset_biomechanical_feature_provenance`: 4

### dataset=Clinical Gait Signals | model=qwen3:14b

Top embedding contexts:
- `clinical_dataset_biomechanical_feature_provenance`: 180
- `pathology_context_neurological_orthopedic_gait`: 169
- `perry_data2_phase_support_asymmetry`: 145
- `perry_data2_progression_speed_length`: 134
- `turning_performance_data2_context`: 26

Top vectorless nodes:
- `perry_data2_phase_support_asymmetry`: 148
- `pathology_context_neurological_orthopedic_gait`: 143
- `perry_data2_progression_speed_length`: 124
- `clinical_dataset_biomechanical_feature_provenance`: 106
- `turning_performance_data2_context`: 103
- `perry_velocity_cadence_timing`: 7
- `neurodegenerative_pd_gait_context`: 5
- `clinical_gait_analysis_general_context`: 2

## Notes

- All comparisons are paired within the same case and LLM.
- Mean difference is vectorless_tree_rag minus embedding_rag.
- For hallucination_rate, negative mean difference favors vectorless_tree_rag because lower is better.
- Sign-test p-values are exact two-sided tests over non-tied paired differences.
- Bootstrap CIs are percentile intervals for the mean paired difference.
- Records with status=no_features are counted but excluded from inferential paired tests.
