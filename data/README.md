# Dataset access and local inputs

The raw data are deliberately not stored in this repository. Download them from the original sources, observe each dataset's access and reuse terms, and place the required files at the paths below. Paths are relative to the repository root because the loaders use relative paths.

## 1. Clinical Gait Signals

**What was used:** participant-level gait features aggregated from wearable-sensor trial files. The experiment used the healthy-control group as a reference and applied the feature rules documented in the manuscript.

- Dataset paper: [Scientific Data, DOI 10.1038/s41597-025-05959-w](https://doi.org/10.1038/s41597-025-05959-w)
- Dataset record: [Figshare, DOI 10.6084/m9.figshare.28806086](https://doi.org/10.6084/m9.figshare.28806086)
- Expected archive: `Data_2_Clinical_Gait_Signals_with_Wearable_Sensors_from_Healthy_Neurological/dataset.zip`
- Case-builder script: `scripts/build_data2_biomechanical_cases.py`
- Runner input after case construction: `outputs/data2_biomechanical_cases_2026-06-15.jsonl`

The archive should retain the dataset's internal paths, including metadata JSON files and processed trial signals. The exact name of the downloaded archive may differ; rename or copy it to the expected path without modifying its contents.

## 2. GAITEX

**What was used:** timestamp CSV files and associated metadata to form participant-condition cases. The full raw motion arrays were not analysed by this study's loader.

- Dataset paper/preprint: [arXiv:2507.21069](https://arxiv.org/abs/2507.21069)
- Dataset record listed by the dataset maintainers: [Zenodo record 15729055](https://zenodo.org/records/15729055)
- Expected archive: `Data_3_GAITEX/data.zip`

**Identifier to verify:** an earlier manuscript draft and experimental notes cite Zenodo record `15729056`, while the dataset maintainers' GitHub page currently links to record `15729055`. Confirm which record/version was actually downloaded for the paper before final public release; do not substitute a version without checking its contents.

## 3. Health&Gait

**What was used:** measured gait parameters, participant measures, and estimated gait parameters for traceability. The model input uses the measured parameters and participant metadata; estimated values are retained for provenance.

- Dataset paper: [Scientific Data, DOI 10.1038/s41597-024-04327-4](https://doi.org/10.1038/s41597-024-04327-4)
- Dataset record: [Zenodo record 14039922](https://zenodo.org/records/14039922)
- Expected files:
  - `prepared_data/health_gait/gait_parameters.csv`
  - `prepared_data/health_gait/patients_measures.csv`
  - `prepared_data/health_gait/gait_parameters_estimation.csv`

The loader requires these three CSV tables. It does not require the videos, pose files, optical-flow images, or other large modalities for this experiment. Obtain the CSVs from the original dataset release and retain the original participant IDs and column names.

## 4. Above-Knee Amputee Gait

**What was used:** processed bilateral kinematic and kinetic MATLAB data, grouped by subject and speed condition, plus the subject-information workbook.

- Dataset record: [Figshare, DOI 10.6084/m9.figshare.10308443](https://doi.org/10.6084/m9.figshare.10308443)
- The Figshare record currently reports a total download size of about 12 GB; downloading only the relevant files may be possible through the Figshare interface/API.
- Expected files:
  - `Data_4_Above_Knee_Amputee_Gait/subject_zips/*.zip`
  - `Data_4_Above_Knee_Amputee_Gait/docs/Subject Information.xlsx`

Keep the `TF##/Matlab Workspace/TF##_DATA.mat` structure inside each subject archive. The loader reads these workspaces directly.

## Expected local layout

```text
repository-root/
├── Data_2_Clinical_Gait_Signals_with_Wearable_Sensors_from_Healthy_Neurological/
│   └── dataset.zip
├── Data_3_GAITEX/
│   └── data.zip
├── Data_4_Above_Knee_Amputee_Gait/
│   ├── docs/Subject Information.xlsx
│   └── subject_zips/*.zip
└── prepared_data/health_gait/
    ├── gait_parameters.csv
    ├── patients_measures.csv
    └── gait_parameters_estimation.csv
```

Do not commit the downloaded datasets. The repository `.gitignore` excludes these local input folders. Dataset downloads may be subject to terms separate from this code release.
