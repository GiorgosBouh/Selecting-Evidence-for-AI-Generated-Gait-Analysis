# Full Dataset RAG Batch Summary

## Record Counts

- Input records: `267`
- Analyzable paired records: `264`
- Status counts: `{'ok': 264, 'no_features': 3}`

## Primary Paired Results

Mean diff is `vectorless_tree_rag - embedding_rag`. For hallucination, negative favors vectorless.

| Group | N | Metric | Emb mean | Vec mean | Diff | 95% bootstrap CI | sign-test p | Vec better | Ties | Emb better |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|
| overall | 264 | feature_coverage | 0.9749 | 1.0000 | 0.0251 | [0.0119, 0.0412] | 0.0000 | 18 | 246 | 0 |
| overall | 264 | faithfulness | 0.9788 | 1.0000 | 0.0212 | [0.0078, 0.0380] | 0.0010 | 11 | 253 | 0 |
| overall | 264 | hallucination_rate | 0.0212 | 0.0000 | -0.0212 | [-0.0380, -0.0078] | 0.0010 | 11 | 253 | 0 |
| overall | 264 | citation_accuracy | 0.9886 | 1.0000 | 0.0114 | [0.0000, 0.0265] | 0.2500 | 3 | 261 | 0 |
| overall | 264 | claim_count | 2.8409 | 2.9091 | 0.0682 | [0.0227, 0.1174] | 0.0106 | 18 | 241 | 5 |
| overall | 264 | quality_score | 2.9211 | 3.0000 | 0.0789 | [0.0293, 0.1413] | 0.0000 | 18 | 246 | 0 |
| dataset=Above-Knee Amputee Gait \| model=llama3.1:8b | 88 | feature_coverage | 0.9420 | 1.0000 | 0.0580 | [0.0212, 0.1034] | 0.0020 | 10 | 78 | 0 |
| dataset=Above-Knee Amputee Gait \| model=llama3.1:8b | 88 | faithfulness | 0.9451 | 1.0000 | 0.0549 | [0.0170, 0.1042] | 0.0156 | 7 | 81 | 0 |
| dataset=Above-Knee Amputee Gait \| model=llama3.1:8b | 88 | hallucination_rate | 0.0549 | 0.0000 | -0.0549 | [-0.1042, -0.0170] | 0.0156 | 7 | 81 | 0 |
| dataset=Above-Knee Amputee Gait \| model=llama3.1:8b | 88 | citation_accuracy | 0.9659 | 1.0000 | 0.0341 | [0.0000, 0.0795] | 0.2500 | 3 | 85 | 0 |
| dataset=Above-Knee Amputee Gait \| model=llama3.1:8b | 88 | claim_count | 2.7614 | 2.9091 | 0.1477 | [0.0455, 0.2727] | 0.0386 | 10 | 76 | 2 |
| dataset=Above-Knee Amputee Gait \| model=llama3.1:8b | 88 | quality_score | 2.7981 | 3.0000 | 0.2019 | [0.0629, 0.3837] | 0.0020 | 10 | 78 | 0 |
| dataset=Above-Knee Amputee Gait \| model=mistral-small:24b | 88 | feature_coverage | 0.9913 | 1.0000 | 0.0087 | [0.0019, 0.0178] | 0.1250 | 4 | 84 | 0 |
| dataset=Above-Knee Amputee Gait \| model=mistral-small:24b | 88 | faithfulness | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] | 1.0000 | 0 | 88 | 0 |
| dataset=Above-Knee Amputee Gait \| model=mistral-small:24b | 88 | hallucination_rate | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 1.0000 | 0 | 88 | 0 |
| dataset=Above-Knee Amputee Gait \| model=mistral-small:24b | 88 | citation_accuracy | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] | 1.0000 | 0 | 88 | 0 |
| dataset=Above-Knee Amputee Gait \| model=mistral-small:24b | 88 | claim_count | 2.8750 | 2.8864 | 0.0114 | [-0.0455, 0.0682] | 1.0000 | 4 | 81 | 3 |
| dataset=Above-Knee Amputee Gait \| model=mistral-small:24b | 88 | quality_score | 2.9913 | 3.0000 | 0.0087 | [0.0019, 0.0178] | 0.1250 | 4 | 84 | 0 |
| dataset=Above-Knee Amputee Gait \| model=qwen3:14b | 88 | feature_coverage | 0.9913 | 1.0000 | 0.0087 | [0.0019, 0.0178] | 0.1250 | 4 | 84 | 0 |
| dataset=Above-Knee Amputee Gait \| model=qwen3:14b | 88 | faithfulness | 0.9913 | 1.0000 | 0.0087 | [0.0019, 0.0178] | 0.1250 | 4 | 84 | 0 |
| dataset=Above-Knee Amputee Gait \| model=qwen3:14b | 88 | hallucination_rate | 0.0087 | 0.0000 | -0.0087 | [-0.0178, -0.0019] | 0.1250 | 4 | 84 | 0 |
| dataset=Above-Knee Amputee Gait \| model=qwen3:14b | 88 | citation_accuracy | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] | 1.0000 | 0 | 88 | 0 |
| dataset=Above-Knee Amputee Gait \| model=qwen3:14b | 88 | claim_count | 2.8864 | 2.9318 | 0.0455 | [0.0114, 0.0909] | 0.1250 | 4 | 84 | 0 |
| dataset=Above-Knee Amputee Gait \| model=qwen3:14b | 88 | quality_score | 2.9739 | 3.0000 | 0.0261 | [0.0057, 0.0534] | 0.1250 | 4 | 84 | 0 |

## Retrieval Frequencies

### overall

Top embedding contexts:
- `amputee_ankle_sagittal_rom_asymmetry`: 225
- `amputee_joint_moment_asymmetry`: 219
- `amputee_hip_compensation_pattern`: 99
- `amputee_knee_sagittal_pattern`: 78
- `amputee_ankle_propulsion_push_off`: 54

Top vectorless nodes:
- `amputee_joint_moment_asymmetry`: 228
- `amputee_ankle_sagittal_rom_asymmetry`: 225
- `amputee_hip_compensation_pattern`: 102
- `amputee_knee_sagittal_pattern`: 78
- `amputee_ankle_propulsion_push_off`: 54

### dataset=Above-Knee Amputee Gait | model=llama3.1:8b

Top embedding contexts:
- `amputee_ankle_sagittal_rom_asymmetry`: 75
- `amputee_joint_moment_asymmetry`: 73
- `amputee_hip_compensation_pattern`: 33
- `amputee_knee_sagittal_pattern`: 26
- `amputee_ankle_propulsion_push_off`: 18

Top vectorless nodes:
- `amputee_joint_moment_asymmetry`: 76
- `amputee_ankle_sagittal_rom_asymmetry`: 75
- `amputee_hip_compensation_pattern`: 34
- `amputee_knee_sagittal_pattern`: 26
- `amputee_ankle_propulsion_push_off`: 18

### dataset=Above-Knee Amputee Gait | model=mistral-small:24b

Top embedding contexts:
- `amputee_ankle_sagittal_rom_asymmetry`: 75
- `amputee_joint_moment_asymmetry`: 73
- `amputee_hip_compensation_pattern`: 33
- `amputee_knee_sagittal_pattern`: 26
- `amputee_ankle_propulsion_push_off`: 18

Top vectorless nodes:
- `amputee_joint_moment_asymmetry`: 76
- `amputee_ankle_sagittal_rom_asymmetry`: 75
- `amputee_hip_compensation_pattern`: 34
- `amputee_knee_sagittal_pattern`: 26
- `amputee_ankle_propulsion_push_off`: 18

### dataset=Above-Knee Amputee Gait | model=qwen3:14b

Top embedding contexts:
- `amputee_ankle_sagittal_rom_asymmetry`: 75
- `amputee_joint_moment_asymmetry`: 73
- `amputee_hip_compensation_pattern`: 33
- `amputee_knee_sagittal_pattern`: 26
- `amputee_ankle_propulsion_push_off`: 18

Top vectorless nodes:
- `amputee_joint_moment_asymmetry`: 76
- `amputee_ankle_sagittal_rom_asymmetry`: 75
- `amputee_hip_compensation_pattern`: 34
- `amputee_knee_sagittal_pattern`: 26
- `amputee_ankle_propulsion_push_off`: 18

## Notes

- All comparisons are paired within the same case and LLM.
- Mean difference is vectorless_tree_rag minus embedding_rag.
- For hallucination_rate, negative mean difference favors vectorless_tree_rag because lower is better.
- Sign-test p-values are exact two-sided tests over non-tied paired differences.
- Bootstrap CIs are percentile intervals for the mean paired difference.
- Records with status=no_features are counted but excluded from inferential paired tests.
