# Generate Raw Prompt Text

Use this note only as an operational companion to `docs/project/dataset_contract.md`.

Goal:
- write one prompt text file for each paired `(input, target)` sample in `data/{data_root}`
- save the results under:
  - `data/{data_root}/train/raw/prompt/*.txt`
  - `data/{data_root}/val/raw/prompt/*.txt`

Prompt guidance:
- describe the intended edit visible between the input and target pair
- include the required task keywords: `{keywords}`
- keep prompts task-aligned rather than stylistically random
- light variation across samples is acceptable when it preserves the same primitive-task meaning

Rules:
- keep filenames aligned with the paired sample ids already present under `raw/images/`
- keep this file subordinate to `docs/project/dataset_contract.md`
- do not generate any latent caches or other derived artifacts in this step
