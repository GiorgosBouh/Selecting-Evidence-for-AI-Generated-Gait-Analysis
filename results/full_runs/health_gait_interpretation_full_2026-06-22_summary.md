# Full Dataset RAG Batch Summary

## Record Counts

- Input records: `1194`
- Analyzable paired records: `726`
- Status counts: `{'ok': 726, 'no_features': 468}`

## Primary Paired Results

Mean diff is `vectorless_tree_rag - embedding_rag`. For hallucination, negative favors vectorless.

| Group | N | Metric | Emb mean | Vec mean | Diff | 95% bootstrap CI | sign-test p | Vec better | Ties | Emb better |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|
| overall | 726 | feature_coverage | 0.9972 | 0.8601 | -0.1371 | [-0.1612, -0.1147] | 0.0000 | 6 | 608 | 112 |
| overall | 726 | faithfulness | 0.9964 | 0.8613 | -0.1351 | [-0.1596, -0.1125] | 0.0000 | 9 | 607 | 110 |
| overall | 726 | hallucination_rate | 0.0036 | 0.1387 | 0.1351 | [0.1125, 0.1593] | 0.0000 | 9 | 607 | 110 |
| overall | 726 | citation_accuracy | 1.0000 | 0.8747 | -0.1253 | [-0.1501, -0.1033] | 0.0000 | 0 | 635 | 91 |
| overall | 726 | claim_count | 2.1956 | 1.6956 | -0.5000 | [-0.5716, -0.4298] | 0.0000 | 26 | 472 | 228 |
| overall | 726 | quality_score | 2.9900 | 2.4575 | -0.5326 | [-0.6299, -0.4431] | 0.0000 | 9 | 605 | 112 |
| dataset=Health&Gait \| model=llama3.1:8b | 242 | feature_coverage | 0.9983 | 0.6258 | -0.3725 | [-0.4318, -0.3143] | 0.0000 | 0 | 142 | 100 |
| dataset=Health&Gait \| model=llama3.1:8b | 242 | faithfulness | 0.9954 | 0.6295 | -0.3659 | [-0.4265, -0.3081] | 0.0000 | 3 | 141 | 98 |
| dataset=Health&Gait \| model=llama3.1:8b | 242 | hallucination_rate | 0.0046 | 0.3705 | 0.3659 | [0.3079, 0.4264] | 0.0000 | 3 | 141 | 98 |
| dataset=Health&Gait \| model=llama3.1:8b | 242 | citation_accuracy | 1.0000 | 0.6653 | -0.3347 | [-0.3926, -0.2769] | 0.0000 | 0 | 161 | 81 |
| dataset=Health&Gait \| model=llama3.1:8b | 242 | claim_count | 2.3058 | 1.3058 | -1.0000 | [-1.1570, -0.8347] | 0.0000 | 15 | 103 | 124 |
| dataset=Health&Gait \| model=llama3.1:8b | 242 | quality_score | 2.9891 | 1.5501 | -1.4391 | [-1.6759, -1.2098] | 0.0000 | 3 | 139 | 100 |
| dataset=Health&Gait \| model=mistral-small:24b | 242 | feature_coverage | 0.9986 | 0.9545 | -0.0441 | [-0.0716, -0.0220] | 0.0005 | 0 | 230 | 12 |
| dataset=Health&Gait \| model=mistral-small:24b | 242 | faithfulness | 1.0000 | 0.9545 | -0.0455 | [-0.0723, -0.0227] | 0.0005 | 0 | 230 | 12 |
| dataset=Health&Gait \| model=mistral-small:24b | 242 | hallucination_rate | 0.0000 | 0.0455 | 0.0455 | [0.0227, 0.0723] | 0.0005 | 0 | 230 | 12 |
| dataset=Health&Gait \| model=mistral-small:24b | 242 | citation_accuracy | 1.0000 | 0.9587 | -0.0413 | [-0.0661, -0.0207] | 0.0020 | 0 | 232 | 10 |
| dataset=Health&Gait \| model=mistral-small:24b | 242 | claim_count | 2.1198 | 1.7934 | -0.3264 | [-0.4050, -0.2438] | 0.0000 | 6 | 167 | 69 |
| dataset=Health&Gait \| model=mistral-small:24b | 242 | quality_score | 2.9986 | 2.8223 | -0.1763 | [-0.2865, -0.0882] | 0.0005 | 0 | 230 | 12 |
| dataset=Health&Gait \| model=qwen3:14b | 242 | feature_coverage | 0.9948 | 1.0000 | 0.0052 | [0.0014, 0.0098] | 0.0312 | 6 | 236 | 0 |
| dataset=Health&Gait \| model=qwen3:14b | 242 | faithfulness | 0.9938 | 1.0000 | 0.0062 | [0.0017, 0.0118] | 0.0312 | 6 | 236 | 0 |
| dataset=Health&Gait \| model=qwen3:14b | 242 | hallucination_rate | 0.0062 | 0.0000 | -0.0062 | [-0.0118, -0.0017] | 0.0312 | 6 | 236 | 0 |
| dataset=Health&Gait \| model=qwen3:14b | 242 | citation_accuracy | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] | 1.0000 | 0 | 242 | 0 |
| dataset=Health&Gait \| model=qwen3:14b | 242 | claim_count | 2.1612 | 1.9876 | -0.1736 | [-0.2397, -0.1074] | 0.0000 | 5 | 202 | 35 |
| dataset=Health&Gait \| model=qwen3:14b | 242 | quality_score | 2.9824 | 3.0000 | 0.0176 | [0.0050, 0.0334] | 0.0312 | 6 | 236 | 0 |

## Retrieval Frequencies

### overall

Top embedding contexts:
- `health_gait_dataset_schema`: 726
- `fukuchi_speed_effects`: 573
- `perry_velocity_cadence_timing`: 471
- `perry_support_phase_stability`: 369
- `perry_step_stride_progression`: 252

Top vectorless nodes:
- `perry_support_phase_stability`: 383
- `perry_velocity_cadence_timing`: 331
- `perry_data2_progression_speed_length`: 328
- `pathology_context_neurological_orthopedic_gait`: 204
- `clinical_dataset_biomechanical_feature_provenance`: 201
- `perry_step_stride_progression`: 178
- `health_gait_csv_parameter_mapping`: 136
- `perry_data2_phase_support_asymmetry`: 121

### dataset=Health&Gait | model=llama3.1:8b

Top embedding contexts:
- `health_gait_dataset_schema`: 242
- `fukuchi_speed_effects`: 191
- `perry_velocity_cadence_timing`: 157
- `perry_support_phase_stability`: 123
- `perry_step_stride_progression`: 84

Top vectorless nodes:
- `perry_data2_progression_speed_length`: 236
- `clinical_dataset_biomechanical_feature_provenance`: 126
- `perry_data2_phase_support_asymmetry`: 118
- `health_gait_csv_parameter_mapping`: 110
- `perry_support_phase_stability`: 87
- `perry_velocity_cadence_timing`: 20
- `fukuchi_speed_effects`: 18
- `perry_step_stride_progression`: 6

### dataset=Health&Gait | model=mistral-small:24b

Top embedding contexts:
- `health_gait_dataset_schema`: 242
- `fukuchi_speed_effects`: 191
- `perry_velocity_cadence_timing`: 157
- `perry_support_phase_stability`: 123
- `perry_step_stride_progression`: 84

Top vectorless nodes:
- `perry_velocity_cadence_timing`: 150
- `perry_support_phase_stability`: 148
- `pathology_context_neurological_orthopedic_gait`: 93
- `perry_step_stride_progression`: 86
- `perry_data2_progression_speed_length`: 52
- `fukuchi_speed_effects`: 44
- `health_gait_csv_parameter_mapping`: 15

### dataset=Health&Gait | model=qwen3:14b

Top embedding contexts:
- `health_gait_dataset_schema`: 242
- `fukuchi_speed_effects`: 191
- `perry_velocity_cadence_timing`: 157
- `perry_support_phase_stability`: 123
- `perry_step_stride_progression`: 84

Top vectorless nodes:
- `perry_velocity_cadence_timing`: 161
- `perry_support_phase_stability`: 148
- `pathology_context_neurological_orthopedic_gait`: 111
- `perry_step_stride_progression`: 86
- `clinical_dataset_biomechanical_feature_provenance`: 75
- `perry_data2_progression_speed_length`: 40
- `fukuchi_speed_effects`: 29
- `health_gait_csv_parameter_mapping`: 11

## Notes

- All comparisons are paired within the same case and LLM.
- Mean difference is vectorless_tree_rag minus embedding_rag.
- For hallucination_rate, negative mean difference favors vectorless_tree_rag because lower is better.
- Sign-test p-values are exact two-sided tests over non-tied paired differences.
- Bootstrap CIs are percentile intervals for the mean paired difference.
- Records with status=no_features are counted but excluded from inferential paired tests.
