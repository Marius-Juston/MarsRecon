# MarsCLIP Execution Roadmap

This document turns the embedding workflow into a finite execution plan. The workflow spec in `reports/MarsCLIP_Embedding_Workflow.md` remains the target architecture; this roadmap tracks what is done, what is next, how each stage is validated, and what artifacts should exist for the final report.

## Status Snapshot

Current position: Phase 0 and Stage A are complete. Stage B has advanced beyond a text-only baseline: **B1a-geo** (image + text + geo-context contrastive training) and **B1a-geo+** (positional location encoders, separate geo temperature, geo warmup) have been run and analyzed on Olympus splits. **B1a-pairs**—deterministic local/global views, cached dual SatMAE features, CACo-style local↔global InfoNCE, and local↔global retrieval in train/eval—is **implemented in code** (`stage_b.align_marsclip`, `evaluate_marsclip`) and documented in `CODEX.md`, but the **first full training run sized like production** is still the main open loop. Optional follow-ups: softer geo metrics (geocell hit-rate, distance correlation), and **offline** patch text augmentation (human- or Cursor-assisted JSON, not runtime LLM).

Overall workflow progress: `[########--] ~75%`

This percentage is intentionally conservative. It reflects alignment to the full workflow, not just the amount of code already written.

## Comparison References

- `SatMAE`
  Use as inspiration for Stage A multi-spectral masked autoencoding, band handling, and later Stage B transfer comparisons.
  Source: https://sustainlab-group.github.io/SatMAE/

- `Scale-MAE`
  Use as inspiration for scale/GSD-aware Stage A ablations, especially scale encoding and future low/high-frequency reconstruction losses.
  Source: https://arxiv.org/pdf/2212.14532

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

Status: `complete`

Progress: `[##########] 100%`

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
- explicit Stage A1 default `min_valid_fraction=0.5`, matching the workflow's ">50% nodata" rule
- patch-compatible preview path
- patch summary / report utility
- focused Stage A1 unit tests
- lightweight real-data smoke test on Olympus Mons
- saved real-data patch preview and patch summary artifacts

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

Status: `complete`

Progress: `[##########] 100%`

Deliverables:
- ViT-based masked autoencoder
- Nodata-aware patch dropping before random masking
- Random masking over valid patches only
- Decoder and reconstruction loss
- Scale-aware positional encoding
- If feasible in v1, independent spectral band tokenization with missing-band handling

What exists now:
- Stage A MAE module wired to the Stage A1 patch contract
- valid-patch filtering with the shared `min_valid_fraction=0.5` threshold
- random masking restricted to valid patches
- scale-aware positional encoding from patch scale features
- reconstruction loss computed only on masked valid pixels
- direct batch collation from Stage A1 patch samples
- focused unit tests and a lightweight real-data forward/backward smoke test
- masked/reconstructed preview utility with real Olympus Mons artifact generation
- verified short training smoke where reconstruction loss decreases on real Olympus Mons patches

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

Status: `complete`

Progress: `[##########] 100%`

Deliverables:
- Training script or CLI
- Checkpoint save / load support
- Configurable hyperparameters
- Logging of reconstruction metrics
- Embedding export for sanity checks

What exists now:
- minimal `train_marsclip_mae.py` CLI
- MAE dataloader / batch collation for Stage A1 patch samples
- checkpoint save / load helpers
- JSON loss history and loss-curve artifact generation
- before/after reconstruction previews from the training script
- lightweight real-data training smoke on Olympus Mons
- periodic checkpoint saving
- periodic reconstruction preview saving
- best-checkpoint tracking
- summary JSON artifact for each run
- verified resume-from-checkpoint path on real data
- embedding export utility on top of saved MAE checkpoints
- nearest-neighbor gallery generation
- PCA-based embedding scatter generation
- real Olympus Mons embedding sanity artifacts from a saved checkpoint
- longer 40-step Stage A run with periodic checkpoints on Olympus Mons
- longer 80-step CUDA Stage A run with GPU-backed checkpoint selection at batch size `8`

Validation:
- Resume-from-checkpoint works
- Short training run completes
- Throughput and memory are measured
- Nearest-neighbor retrieval among image embeddings is qualitatively sensible

Final report artifacts:
- Training configuration table
- Checkpoint selection rationale
- Embedding visualization or nearest-neighbor gallery

Exit criteria:
- We have a reusable Stage A pretrained vision encoder

### 5. Stage B1: Paired Multi-Scale Crops And Workflow-Aligned Encoders

Status: `in_progress`

Progress: `[######----] 65%`

Why this is not complete:
- Trainer implements **global patch** + **deterministic centered local crop** (resize to same resolution), both through the **frozen Stage A** encoder, shared image projector, and local↔global contrastive term; geo + text paths match the B1a-geo roadmap.
- Still missing vs a fancier workflow doc: stochastic multi-scale sampling, FiLM conditioning, and optional separate “expanded rationale” ingestion beyond the Olympus short strings.

Deliverables:
- Paired local/global crop sampler ~~(minimal: centered crop fraction)~~ **done in trainer cache path**
- Stage A encoder reused as the visual backbone **done**
- Geometry / location encoding **partially aligned** (coords + optional RFF/SIREN/SH stacks in `geo_encoders.py`; eval shows 1↔1 geo retrieval is a weak metric on this split—see CODEX).
- Frozen text embedding path over raw rationales **done** (T5 + cache).
- ~~Optional FiLM~~ still open.
- Context fusion beyond concat+projector **open**.

Validation:
- Unit tests for **`make_local_view`** and crop bounds (**`tests/test_align_marsclip_helpers.py`**)
- Tests for paired-crop validity / overlap intersection rules **still open** if we add stochastic crops
- End-to-end **GPU smoke** (small `--max-patches`, `--epochs 2`) and one **production-scale B1a-pairs run** **open**

Final report artifacts:
- Paired local/global crop figure
- Stage B encoder diagram
- Context fusion diagram

Exit criteria:
- We can produce aligned local/global image embeddings **and validate local↔global retrieval improves or stabilizes vs geo-only** on held-out Olympus splits (run + eval archived next to checkpoint).

### 6. Stage B2: Contrastive Training Aligned To The Workflow

Status: `in_progress`

Progress: `[####------] 40%`

Note: Core contrastive training for B1a (image–text–geo–local/global) exists; this stage is “complete” only when workflows call for additional loss variants and full ablations.

Deliverables:
- Training loop for Stage B **(B1a trainer exists; extend with ablations / soft targets as needed)**
- Soft-target or workflow-appropriate contrastive loss **open**
- Cross-scale consistency loss **partial (local↔global InfoNCE in B1a-pairs)**
- Logging of image-text, image-context, and scale-consistency metrics **partial (W&B + val retrieval)**

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

**Primary (blocking “Stage B1 done” feeling):**

1. **Run B1a-pairs training** using the canonical flags in `CODEX.md`: `--paired-views local_global --local-crop-fraction 0.5 --lg-loss-weight 0.5`, same Olympus assets and Stage A checkpoint family as prior B1a runs. Log `train/lg_*`, `val/local_to_global_*`, and compare against the best **B1a-geo+** checkpoint on image↔text.
2. **Evaluate** `best_checkpoint.pt` with `evaluate_marsclip.py`, full val (`--max-patches` large enough), optionally `--paired-views-override local_global` for older ckpts.

**Secondary (recommended soon):**

3. ~~Implement **coarse geo metrics** (geocell hit-rate…)~~ **Done in code**: training and `evaluate_marsclip` report ``image_to_geo_geocell_*deg_topk_any_neighbor`` when batch metadata carries ``centroid_lat/lon`` (patch dataset now emits them). Re-cache or use a fresh cache if an older warmup JSON lacks centroids.

4. **Offline augmentation**: **Partially done**: JSONL loader + optional ``--patch-text-augment-jsonl`` on ``align_marsclip`` concatenate per-patch ``augment`` to ``rationale_raw``. A checked-in 384-patch train sample lives under ``assets/offline_augment/``. **Smoke OK (2026-04-27)** — B1a-pairs + augment on 320 train / 128 val patches, 2 epochs, run dir ``…/marsclip_align/20260427/20260427_211717_olympus-b1a-pairs-augment-smoke-v1/``; scale to full Olympus + longer schedule for production.

**Tertiary / polish:**

5. Extend unit tests if we add non-deterministic crops or richer batch contracts.

This ordering keeps momentum on the **largest unexplored empirical gap** (first real B1a-pairs run) while deferring nicer metrics and text enrichment until pairs are measured.

## Done Criteria For The Whole Project

The project is complete when all of the following are true:

- Stage A patch dataset exists and is validated
- Stage A MAE pretraining runs and produces a reusable checkpoint
- Stage B paired multi-scale alignment runs on top of the Stage A encoder
- Retrieval and ablation evaluation scripts produce stable results
- Final report figures and tables are generated from the codebase
- Focused tests pass for all implemented MarsCLIP blocks

At that point, the project is done. There are extensions we could add later, but they are not required for a complete MarsCLIP report.
