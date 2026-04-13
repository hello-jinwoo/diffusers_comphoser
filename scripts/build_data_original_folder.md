# Build `original/` Split Folders

Use this note only as an operator checklist for step 1 of the canonical contract in `docs/project/dataset_contract.md`.

Goal:
- create `data/{dataname}/train/original/` and `data/{dataname}/val/original/`
- preserve the contract-aligned substructure that will later feed the raw-image builder

Expected output structure:

```text
data/{dataname}/
  train/original/
    images/input/
    images/target/
    prompt/
  val/original/
    images/input/
    images/target/
    prompt/
```

Operator inputs to record before copying files:
- source root: `{src_root}`
- input source description: `{input_description}`
- target source description: `{target_description}`
- notes: `{note1}`, `{note2}`, ...

Rules:
- keep this file subordinate to `docs/project/dataset_contract.md`
- relocate or copy the original assets only; do not resize, rename, or preprocess in this step
- keep train/val split decisions explicit and stable
- do not create extra artifacts outside the contract folders
