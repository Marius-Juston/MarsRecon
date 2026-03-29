# MarsCLIP Execution Roadmap

This document turns the embedding workflow into a finite execution plan. The workflow spec in `reports/MarsCLIP_Embedding_Workflow.md` remains the target architecture; this roadmap tracks what is done, what is next, how each stage is validated, and what artifacts should exist for the final report.

## Status Snapshot

Current position: Phase 0 is complete, and we have a tested observation-level MarsCLIP baseline scaffold plus the first patch-level Stage A bridge. We are not yet at the workflow-accurate Stage A MAE / Stage B training pipeline.

Overall workflow progress: `[###-------] ~30%`

This percentage is intentionally conservative. It reflects alignment to the full workflow, not just the amount of code already written.

## Stage Tracker

### 1. Phase 0: Data Infrastructure And Baseline Scaffold

Status: `complete`

Progress: `[##########] 100%`

What exists:
- HiRISE ingestion, calibration, COG support, spatial indexing, and strip-aware sampling from the original repo
- Observation manifest builder
- Observation-level MarsCLIP dataset
- Offline rationale expansion cache
- Text collator
- First tri-modal baseline model
- Visualization utility
- Tests for each new MarsCLIP block

Validation:
- Focused unit and integration tests pass for manifest, dataset, rationale cache, text batching, model, and preview utility
- Real-sample preview generation works on local data

Final report artifacts:
- Dataset overview table
- Modality availability summary
- Sample preview figure

Exit criteria:
- Stable manifest and dataset contracts
- Reproducible tests for every MarsCLIP v1 data block

### 2. Stage A1: Patch-Level Dataset And Sampler Bridge

Status: `in_progress`

Progress: `[#####-----] 50%`

Why it is next:
- This is the biggest remaining gap between the current code and the workflow
- The workflow is patch-level and scale-aware; our current MarsCLIP baseline is observation-level

Deliverables:
- Patch-level dataset for `0.005 deg` local crops
- Patch metadata with centroid, bounds, scale, valid-channel flags, nodata mask, and source observation metadata
- Tightened nodata handling using a workflow-aligned valid-patch rule
- Preview utility for sampled patches
- Patch statistics summary script or report output

What exists now:
- `MarsCLIPPatchDataset` on top of `MarsHiRISE`
- patch-record builder from strip-aware sampler centres
- dominant-observation and overlap-fraction metadata
- patch validity accounting via nodata / valid-pixel fraction
- patch-compatible preview path
- focused Stage A1 unit tests

Validation:
- Unit tests for patch extraction
- Unit tests for nodata thresholding
- Unit tests for valid-channel / missing-band handling
- Deterministic smoke test on a small local subset
- Saved patch preview grid from real data

Final report artifacts:
- Patch sampling diagram
- Valid vs dropped patch statistics
- Patch gallery figure

Exit criteria:
- We can sample valid training patches reproducibly from local HiRISE COLOR data
- Each patch carries the metadata required by Stage A

### 3. Stage A2: MAE Vision Encoder

Status: `pending`

Progress: `[----------] 0%`

Deliverables:
- ViT-based masked autoencoder
- Nodata-aware patch dropping before random masking
- Random masking over valid patches only
- Decoder and reconstruction loss
- Scale-aware positional encoding
- If feasible in v1, independent spectral band tokenization with missing-band handling

Validation:
- Unit tests for patch-mask construction
- Unit tests for masking ratios and dropped-token logic
- Shape tests for encoder / decoder
- One-batch forward/backward smoke test
- Reconstruction visualization on held-out patches

Final report artifacts:
- Stage A architecture figure
- Masked input / reconstructed output figure
- Stage A loss curve

Exit criteria:
- Stage A model trains on real patch batches without crashing
- Reconstruction loss decreases in a smoke run

### 4. Stage A3: Stage A Training, Checkpointing, And Embedding Sanity

Status: `pending`

Progress: `[----------] 0%`

Deliverables:
- Training script or CLI
- Checkpoint save / load support
- Configurable hyperparameters
- Logging of reconstruction metrics
- Embedding export for sanity checks

Validation:
- Resume-from-checkpoint works
- Short training run completes
- Throughput and memory are measured
- Nearest-neighbor retrieval among image embeddings is qualitatively sensible

Final report artifacts:
- Training configuration table
- Checkpoint selection note
- Embedding visualization or nearest-neighbor gallery

Exit criteria:
- We have a reusable Stage A pretrained vision encoder

### 5. Stage B1: Paired Multi-Scale Crops And Workflow-Aligned Encoders

Status: `pending`

Progress: `[#---------] 10%`

Why this is not zero:
- We already have a baseline text path, geo-context path, and contrastive model scaffold
- But it is still much simpler than the workflow

Deliverables:
- Paired local/global crop sampler
- Stage A encoder reused as the visual backbone
- Geometry encoder aligned more closely with the workflow
- Location encoder aligned more closely with the workflow
- Frozen text embedding path over raw and expanded rationale text
- Optional FiLM conditioning or another explicit geometry-conditioning mechanism
- Context fusion target for text and location

Validation:
- Unit tests for paired-crop validity
- Tests for crop overlap / validity intersection rules
- Tests for encoder output shapes and normalization
- Smoke test for one Stage B batch

Final report artifacts:
- Paired local/global crop figure
- Stage B encoder diagram
- Context fusion diagram

Exit criteria:
- We can produce aligned local/global image embeddings and context targets for one batch

### 6. Stage B2: Contrastive Training Aligned To The Workflow

Status: `pending`

Progress: `[----------] 0%`

Deliverables:
- Training loop for Stage B
- Soft-target or workflow-appropriate contrastive loss
- Cross-scale consistency loss
- Logging of image-text, image-context, and scale-consistency metrics

Validation:
- One-epoch smoke run
- Loss terms are finite and decrease in a short run
- Checkpoint load / resume works

Final report artifacts:
- Stage B loss curves
- Training setup table
- Qualitative retrieval examples during training

Exit criteria:
- We have a trained multimodal embedding model, not just model code

### 7. Evaluation, Ablations, And Final Report Artifacts

Status: `pending`

Progress: `[----------] 0%`

Deliverables:
- Retrieval tasks:
- `text -> image`
- `image -> text`
- `location -> image`
- `local -> global`
- Ablations:
- no expanded rationale
- no viewing parameters
- no location
- no scale encoding
- no Stage A pretraining
- Reproducible evaluation script
- Figure generation pipeline

Validation:
- Metrics are reproducible from saved checkpoints
- Qualitative galleries are saved deterministically
- Ablation table can be regenerated from code

Final report artifacts:
- Main results table
- Ablation table
- Retrieval galleries
- Embedding-space visualization
- Limitations / failure cases section

Exit criteria:
- We can defend the embedding quality with both quantitative and qualitative evidence

### 8. Reproducibility And Submission Packaging

Status: `pending`

Progress: `[----------] 0%`

Deliverables:
- Final experiment configs
- Exact train / eval commands
- Environment notes
- Clean report references to generated figures and tables
- Final test pass on the implemented code paths

Validation:
- End-to-end rerun instructions are complete
- Focused tests still pass
- Final figures regenerate from saved outputs or scripts

Final report artifacts:
- Reproducibility appendix
- Command appendix
- Repo structure summary

Exit criteria:
- Someone else can follow the repo and reproduce the main pipeline decisions

## Immediate Next Step

The next best step is:

`Implement Stage A1: the patch-level dataset and sampler bridge.`

That means:
- move from observation thumbnails to workflow-aligned `0.005 deg` patches
- carry through nodata-aware validity accounting
- preserve scale, location, viewing, and channel-availability metadata per patch
- add tests and real-data patch previews

This is the highest-leverage next move because it closes the largest structural gap between the current baseline and the workflow.

## Done Criteria For The Whole Project

The project is complete when all of the following are true:

- Stage A patch dataset exists and is validated
- Stage A MAE pretraining runs and produces a reusable checkpoint
- Stage B paired multi-scale alignment runs on top of the Stage A encoder
- Retrieval and ablation evaluation scripts produce stable results
- Final report figures and tables are generated from the codebase
- Focused tests pass for all implemented MarsCLIP blocks

At that point, the project is done. There are extensions we could add later, but they are not required for a complete MarsCLIP report.
