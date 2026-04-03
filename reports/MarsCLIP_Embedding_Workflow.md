# MarsCLIP: embedding generation workflow

A complete technical specification for generating multi-modal Mars surface embeddings from HiRISE imagery. This document describes the full pipeline from raw PDS data through two-stage self-supervised pretraining, producing a unified embedding space that encodes surface geology, planetary location, orbital illumination, and scientific context.

Execution status, bounded stages, and progress tracking live in `reports/MarsCLIP_Execution_Roadmap.md`.

---

## Phase 0: data infrastructure (implemented)

The data pipeline converts NASA PDS HiRISE Reduced Data Records into a TorchGeo-compatible dataset with strip-polygon-aware sampling. The key components, already built and validated, are summarised here for completeness.

**Coordinate reference system.** All coordinates use the Mars IAU 2000 geographic CRS (`+proj=longlat +a=3396190 +b=3376200 +no_defs`). Each HiRISE JP2 carries its own per-observation Equirectangular projection; rasterio reprojects into the common geographic CRS at load time. The geographic CRS is the only valid common hub because each observation uses a different `CENTER_LATITUDE` for its Equirectangular projection (DSMAP.CAT).

**COG conversion.** JPEG2000 files are converted to Cloud-Optimised GeoTIFFs with 512-pixel internal tiling and overview levels [2, 4, 8, 16]. This enables O(patch_area) random-access reads during training, compared to O(image_area) for JP2 codeblock decompression. BigTIFF (`IF_SAFER`) handles COLOR products that exceed the 4 GiB Classic TIFF limit. Worker processes are recycled after each conversion (`max_tasks_per_child=1`) to prevent glibc heap retention from exhausting RAM.

**Strip footprint extraction.** The CORNER1–4 columns in the PDS cumulative index (RDRCUMINDEX) are corners of the projected image rectangle, not the strip footprint — they are identical to the `MINIMUM/MAXIMUM_LATITUDE/LONGITUDE` bounding box. True footprints are extracted by reading band 1 of each COG at the coarsest overview level, computing the convex hull of non-zero pixels, and reprojecting the hull vertices to geographic coordinates. Results are cached in a versioned GeoPackage (`spatial_cache_*_v3.gpkg`).

**Strip-aware sampling.** `HiRISEGeoSampler` pre-computes a grid of patch centres within each strip polygon and retains only those whose patch rectangle intersects the polygon. This prevents 20–60% of wasted samples that a bounding-box sampler would produce. The sampler yields `(x_slice, y_slice, t_slice)` tuples for each patch.

**Radiometric calibration.** Raw DN values are converted to I/F reflectance via `I/F = DN × SCALING_FACTOR + OFFSET`, with per-product constants from the detached LBL file. A nodata mask is captured before calibration and restored after, preventing the additive offset from contaminating zero-valued fill pixels.

---

## Stage A: masked autoencoder pretraining

### A.1 Objective

The first pretraining stage learns visual representations of the Martian surface without any labels. HiRISE has approximately 200,000 products spanning the full range of Mars geology — volcanic plains, polar ice, aeolian dunes, impact craters, layered sediments — but zero pixel-level annotations at scale. Self-supervised learning via masked image modelling is the established solution to this data regime.

The masked autoencoder (MAE) objective works by masking a large fraction of the input image and training the model to reconstruct the missing patches from the visible context. This forces the encoder to learn spatial structure, texture statistics, and spectral correlations that are useful for any subsequent vision task, without requiring curated labels. The MAE paradigm was introduced by He et al. (2022, "Masked Autoencoders Are Scalable Vision Learners", CVPR) and subsequently adapted for multi-spectral satellite imagery by Cong et al. (2022, "SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery", NeurIPS). Stage A follows the SatMAE architecture with three Mars-specific adaptations described below.

### A.2 Input representation

A training sample is a single HiRISE patch of size 0.005° × 0.005° (approximately 593 × 593 pixels at native resolution), loaded from the COG sidecar via windowed read and reprojected to the geographic CRS. The patch has up to 3 spectral channels: near-infrared (NIR, 900 nm), red (RED, 700 nm), and blue-green (BG, 500 nm), sourced from the `_COLOR.JP2` product. When only the `_RED.JP2` is available, the RED channel is loaded from that file and NIR/BG are marked as missing.

The patch is accompanied by metadata extracted from the per-product LBL file: `MAP_SCALE` (ground sample distance in metres/pixel), `SCALING_FACTOR`, `OFFSET`, and the CCD configuration (`MRO:BINNING`, `MRO:CCD_FLAG`). The calibrated I/F values (clipped to [0, 1], nodata restored to 0.0) are the model's input — not raw DNs and not display-stretched values. The model therefore learns to represent physical surface reflectance.

### A.3 Independent spectral band tokenization

Standard Vision Transformers tokenize an image by applying a single linear projection across all channels simultaneously: a patch of size P × P × C is flattened into a vector of length P²C and projected to dimension d by a matrix W ∈ ℝ^(P²C × d). This couples the spectral channels at the embedding layer, meaning that if any channel is zero (missing), the entire projection is biased.

SatMAE (Cong et al., 2022) demonstrated that decoupling the spectral channels — projecting each band independently and summing — produces stronger representations for multi-spectral satellite data and naturally handles heterogeneous band availability. MarsCLIP adopts this approach.

For each spectral band c ∈ {NIR, RED, BG}, a separate projection matrix W_c ∈ ℝ^(P² × d) maps the P × P patch from that band into d dimensions. A learnable "missing band token" e_{m,c} ∈ ℝ^d is defined for each band. The patch embedding for the i-th spatial position is:

```
v_i = P_pos,i + S(ρ) + Σ_c [ 𝟙(c valid) · W_c · x_{c,i} + 𝟙(c missing) · e_{m,c} ]
```

where P_pos,i is the 2D sinusoidal positional embedding for position i, S(ρ) is the scale embedding (§A.5), and 𝟙 is the indicator function.

**Why this matters for HiRISE.** The three HiRISE spectral channels come from physically different CCDs with different binning modes. The COLOR product mosaics 6 CCD pairs (IR10/11, RED4/5, BG12/13) that are staggered on the focal plane. Some observations lack the COLOR product entirely — only the RED.JP2 exists. Independent band tokenization means the model can train on RED-only observations without corrupting the embedding with zero-valued IR/BG channels, and the learnable missing tokens e_{m,c} converge during training to represent the marginal expectation of each band given the rest of the dataset. This is strictly superior to zero-filling, which Tran et al. (2017, "Missing Modalities Imputation via Cascaded Residual Autoencoder", CVPR) proved corrupts the empirical risk minimisation objective.

### A.4 Nodata-aware patch masking

The MAE protocol masks a random subset of patches (typically 75%) and trains the decoder to reconstruct them. MarsCLIP adds a preliminary step: patches where more than 50% of pixels are `CORE_NULL = 0` (nodata) are removed from the input sequence entirely, before the random mask is applied.

This is motivated by two considerations. First, self-attention computes softmax(QK^T / √d) over all tokens in the sequence. If a zero-vector token is present, it absorbs attention weight from every query but contributes no information — diluting the attention paid to valid geology. Removing it tightens the attention distribution over real surface features. Second, the sequence length drops from N to N_valid (where N_valid ≤ N), reducing the quadratic memory cost of self-attention from O(N²) to O(N_valid²). For patches near CCD stagger boundaries where 30–60% of tokens may be nodata, this is a significant saving.

The positional embeddings P_pos,i are absolute (tied to spatial position within the patch), so removing tokens does not disrupt the model's understanding of where the remaining valid tokens sit in the image. This is the same mechanism that makes MAE work with 75% masking — the encoder processes only visible tokens, and position is encoded explicitly.

The reconstruction loss (mean squared error) is computed only over masked patches that were valid (nodata fraction < 50%). Patches that were dropped as nodata do not contribute to the loss, preventing the model from learning to "reconstruct" zero-fill regions.

### A.5 Scale-aware positional encoding

Each HiRISE observation has a physical ground sample distance (GSD) recorded in the `MAP_SCALE` field of its LBL file. Within HiRISE, this takes two values: 0.25 m/pixel for RED products at full resolution, and 0.50 m/pixel for COLOR products (which use 2× CCD binning on the IR and BG channels). The GSD tells the model the physical size of each pixel — the same geological feature appears at different pixel scales depending on the product type.

ScaleMAE (Reed et al., 2023, "Scale-MAE: A Scale-Aware Masked Autoencoder for Multiscale Geospatial Representation Learning", ICCV) introduced a continuous sinusoidal encoding of GSD that is additively injected into the patch embeddings. The encoding uses the same functional form as the positional encoding in the original Transformer (Vaswani et al., 2017), but parameterised by the physical GSD rather than a discrete position index:

```
S(ρ) = [sin(ρ · f_1), cos(ρ · f_1), sin(ρ · f_2), cos(ρ · f_2), ..., sin(ρ · f_{d/2}), cos(ρ · f_{d/2})]
```

where f_k = exp(−k · ln(10000) / d) are log-spaced frequencies and ρ is the GSD in metres.

**Why include this when HiRISE has only two GSD values.** Within HiRISE alone, the scale embedding is effectively a learned binary flag. The justification is forward-looking: a Mars foundation model that will incorporate CTX (6 m/pixel), THEMIS (18 m/pixel), MOC (1.5 m/pixel), or CaSSIS (4.5 m/pixel) data needs a continuous scale encoding from the start. Retrofitting a scale dimension into an encoder that was trained without one requires full retraining. Including it now costs nothing (the sinusoidal encoding is parameter-free) and makes multi-mission extension a fine-tuning step rather than an architectural change.

### A.6 Encoder architecture

The encoder is a standard Vision Transformer (ViT-Base or ViT-Large) as described by Dosovitskiy et al. (2021, "An Image is Worth 16x16 Words", ICLR). Patch size P = 16 pixels. For a 593 × 593 input, this produces approximately 37 × 37 = 1369 spatial tokens before nodata dropping and masking.

The encoder processes only the unmasked, valid tokens — the remaining 25% of the original sequence after 75% MAE masking is applied on top of nodata dropping. This is the efficiency advantage of MAE: the encoder never sees the full sequence, so training speed and memory are proportional to the small visible subset.

The output of the encoder is a set of latent vectors h_i ∈ ℝ^d, one per visible token, plus a prepended CLS token h_cls that aggregates global information via self-attention. Stage A's CLS token is not used for reconstruction (the decoder receives all token positions); it becomes the primary output for the multimodal alignment block.

### A.7 Decoder and reconstruction loss

The decoder is a lightweight transformer (4 blocks, d/4 dimension), asymmetric to the encoder. This asymmetry is a key MAE design choice: the encoder does the heavy representational work on the visible subset, while the decoder's only job is to map from latent space back to pixels. This means the encoder learns features that are useful for downstream tasks, not features optimised for pixel prediction.

At the decoder input, learnable mask tokens are inserted at the positions of the masked patches, and positional embeddings are re-added so the decoder knows which spatial location each token (visible or masked) occupies. The decoder outputs a predicted pixel patch for each masked position.

The loss is the mean squared error between predicted and true pixel values, computed only over masked positions:

```
L_MAE = (1 / |M|) Σ_{i ∈ M} ‖ x̂_i − x_i ‖²₂
```

where M is the set of masked patch indices and x_i are the calibrated I/F pixel values. Nodata pixels within each patch are excluded from the MSE computation by masking out the zero-valued pixels in x_i.

### A.8 What Stage A produces

After pretraining on the full HiRISE dataset (no labels, no metadata beyond band availability and GSD), the ViT encoder has learned to predict masked surface patches from visible context. This requires learning: spatial autocorrelation in Martian terrain (crater rims predict crater floors), spectral relationships (NIR reflectance correlates with iron oxide mineralogy), and texture statistics (aeolian ripples have characteristic spatial frequencies). The encoder weights are saved and used to initialise the vision encoder in the multimodal alignment block.

---

## Multimodal Alignment: contrastive multi-modal alignment

### B.1 Objective

The multimodal alignment block takes the visual encoder from Stage A and aligns its output with three auxiliary modalities — planetary location, orbital geometry, and geological text — in a shared d-dimensional embedding space. The alignment is accomplished through contrastive learning: matched pairs (an image patch and its corresponding location/text) are pulled together in embedding space, while unmatched pairs are pushed apart.

This stage follows the SatCLIP paradigm (Klemmer et al., 2025, "SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery", AAAI), which demonstrated that contrastive pretraining between satellite imagery and geographic coordinates produces general-purpose location embeddings that outperform task-specific models on diverse downstream tasks. MarsCLIP extends SatCLIP in three ways: it conditions on orbital viewing geometry via FiLM (to decouple illumination from geology), it aligns with geological text (to enable zero-shot retrieval), and it enforces cross-scale consistency via the CACo loss (to prevent scale collapse in the visual manifold).

### B.2 Sampler extension: paired multi-scale crops

The multimodal alignment block requires each training sample to contain two image crops at different scales, centred on the same geographic point. The local crop I_L is a 0.005° × 0.005° window (approximately 593 × 593 pixels) — the same size used in Stage A. The global crop I_G is a 0.015° × 0.015° window (approximately 1779 × 1779 pixels) providing 3× the spatial context, downsampled to the same pixel dimensions as I_L so both pass through the same ViT.

The sampler validates that both I_L and I_G intersect the strip polygon. Since I_G is larger, some centres that are valid for I_L will produce an I_G that extends beyond the strip boundary. These centres are excluded from the paired-crop set — the sampler maintains two validity masks and takes their intersection.

**Why paired crops.** The cross-area contrastive objective (CACo, Ayush et al., 2021, "Geography-Aware Self-Supervised Learning", ICCV) forces the encoder to produce similar embeddings for local and global views of the same location. Without this constraint, the encoder could learn scale-dependent features — a crater rim at 0.005° might embed far from the same crater at 0.015° — which would fragment the embedding space and make it useless for tasks that operate at varying scales. CACo acts as a topological regulariser that keeps the visual manifold continuous across resolutions.

### B.3 Encoder 1: vision (E_V)

The vision encoder is the ViT from Stage A, with two modifications.

First, the decoder is discarded — only the encoder is retained. The CLS token output h_cls is used as the global image representation, projected through a linear layer to the shared d-dimensional space.

Second, a Feature-wise Linear Modulation (FiLM) layer (Perez et al., 2018, "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI) is inserted after the final transformer block. This layer receives the geometry embedding g_i from E_G and uses it to generate per-feature affine parameters that modulate the visual features:

```
FiLM(h | g_i) = (1 + Δγ) ⊙ h + Δβ
```

where Δγ = W_γ · g_i and Δβ = W_β · g_i are learned linear projections from the geometry space to the feature space.

**Why FiLM and not concatenation.** Orbital geometry (incidence angle, solar longitude) physically scales observed pixel intensities — it is a multiplicative effect on reflectance, not an additive semantic category. A shadow on a crater wall is not a different geological surface; it is the same surface under different illumination. FiLM captures this multiplicative relationship structurally: the γ parameter scales features up or down depending on illumination, and the β parameter shifts the baseline. Concatenating geometry as an additional token would treat it as an independent semantic signal on equal footing with visual tokens, which misrepresents the physics. FiLM produces an output v_i that is an illumination-invariant representation of the surface — the same crater wall embeds identically whether observed at incidence angle 30° or 70°.

The vision encoder produces two outputs per sample: v_L = E_V(I_L, g_i) for the local crop and v_G = E_V(I_G, g_i) for the global crop, both in ℝ^d and L2-normalised to the unit hypersphere.

### B.4 Encoder 2: orbital geometry (E_G)

The geometry encoder maps the observation's viewing parameters to a d-dimensional vector g_i that drives the FiLM conditioning. The inputs are six scalar values extracted from the `VIEWING_PARAMETERS` group in the per-product LBL file:

| Input | LBL field | Encoding | Justification |
|-------|-----------|----------|---------------|
| Incidence angle θ_i | `INCIDENCE_ANGLE` | RFF | High-frequency illumination boundaries |
| Emission angle θ_e | `EMISSION_ANGLE` | RFF | Viewing geometry (near-nadir for HiRISE) |
| Phase angle θ_p | `PHASE_ANGLE` | RFF | Opposition/forward scatter effects |
| Local time t | `LOCAL_TIME` | Periodic | Continuous cyclic variable (0–24 h) |
| Solar longitude L_s | `SOLAR_LONGITUDE` | Periodic | Mars seasonal cycle (0–360°) |
| Sub-solar azimuth φ_ss | `SUB_SOLAR_AZIMUTH` | RFF | Shadow direction within the image |

`NORTH_AZIMUTH` is excluded because it is constant (270.0°) for all Equirectangular products — it encodes the projection convention, not observation geometry.

**Random Fourier Features (RFF).** The three angular parameters (θ_i, θ_e, θ_p) and the sub-solar azimuth are mapped through a fixed Gaussian random matrix B ∈ ℝ^(n_angles × d/2) following Tancik et al. (2020, "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains", NeurIPS). The RFF encoding is:

```
RFF(u) = [cos(u · B), sin(u · B)]
```

This enables the network to resolve sharp illumination boundaries (e.g., the transition between a sunlit crater wall and its shadow) that a raw scalar input cannot represent. Tancik et al. proved that without this projection, neural networks are spectrally biased toward low-frequency functions and fail to capture the high-frequency structure of illumination effects.

**Periodic encoding.** Local time and solar longitude are continuous cyclic variables — 23.9 hours is close to 0.1 hours, and L_s = 359° is close to L_s = 1°. They are mapped to the unit circle:

```
time_emb = [sin(2π · t / 24), cos(2π · t / 24)]
Ls_emb   = [sin(2π · L_s / 360), cos(2π · L_s / 360)]
```

The concatenation of RFF outputs, periodic embeddings, and sub-solar azimuth RFF is passed through a two-layer MLP with GELU activation to produce g_i ∈ ℝ^d.

**Granularity limitation.** All viewing parameters in the LBL are centre-of-observation scalars. A single strip can be 100+ km long, so incidence angle varies by 1–3° from top to bottom. The geometry encoder treats the entire strip as uniformly illuminated. This is acceptable because the dominant illumination effect (overall brightness, shadow direction) is observation-level, and sub-strip gradients are second-order. Per-pixel solar geometry would require SPICE kernel reconstruction, which is outside the RDR data products.

### B.5 Encoder 3: spherical location (E_L)

The location encoder maps planetocentric latitude and longitude to a d-dimensional vector l_i that captures the geographic context of a patch. The input coordinates are the patch centroid (cx, cy) from the sampler, in the Mars geographic CRS (degrees, normalised to [−180, 180] longitude).

The encoding uses spherical harmonics Y_l^m(φ, λ) up to degree L, following Rußwurm et al. (2024, "Geographic Location Encoding with Spherical Harmonics and Sinusoidal Representation Networks", ICLR). This approach has three advantages over raw latitude/longitude or Cartesian (x, y, z) encodings:

1. **Respects spherical topology.** Latitude/longitude coordinates have a coordinate singularity at the poles and a discontinuity at the antimeridian. Spherical harmonics are smooth, continuous functions on the 2-sphere S² with no singularities. On Mars this matters less than on Earth (no data at the poles uses Equirectangular — polar observations use Polar Stereographic), but it ensures the encoding degrades gracefully at high latitudes.

2. **Multi-resolution spatial structure.** Low-degree harmonics (l = 0, 1, 2) capture hemisphere-scale variation (northern lowlands vs. southern highlands). High-degree harmonics (l = 20–40) capture regional patterns (Tharsis bulge, Hellas basin). The model can attend to whatever spatial scale is relevant for the current task.

3. **Orthogonal basis.** The Y_l^m form a complete orthonormal basis on the sphere, meaning they represent arbitrary spatial functions without redundancy. This is strictly better than the Fourier features used by CSP (Mai et al., 2023), which are defined on a flat domain and distort near the poles.

The spherical harmonic features (a vector of length (L+1)²) are passed through a SirenNet — an MLP with periodic sine activations (Sitzmann et al., 2020, "Implicit Neural Representations with Periodic Activation Functions", NeurIPS). SirenNet's periodic activations naturally represent the kind of smoothly varying spatial fields (topography, albedo, thermal inertia) that determine what geology exists at a given location.

The output l_i ∈ ℝ^d is L2-normalised.

### B.6 Encoder 4: geological text (E_T)

The text encoder maps a geological description of the observation's scientific intent to a d-dimensional vector t_i. The raw input is the `RATIONALE_DESC` field from the PDS cumulative index — a 75-character free-text string such as "Syrtis Major pyroxene stratigraphy" or "Seasonal monitoring of polar dunes."

**The data sparsity problem.** Across approximately 200,000 products, there are roughly 500 unique rationale strings. Many are near-duplicates or purely operational ("Calibration target", "Ride-along with CRISM"). A 768-dimensional SciBERT model trained on 500 unique inputs will overfit catastrophically — the text embedding would memorise observation IDs rather than learning geological semantics.

**Solution: LLM-expanded rationales.** Each unique `RATIONALE_DESC` is expanded offline into a geological paragraph using a large language model, with a prompt template:

```
Expand this HiRISE observation rationale into a geological description
(2-3 sentences). Describe the expected surface morphology, mineralogy,
and scientific significance.

Rationale: "{RATIONALE_DESC}"
```

For example, "Syrtis Major pyroxene stratigraphy" becomes a paragraph describing layered pyroxene-bearing units, expected spectral signatures, stratigraphic relationships, and the scientific context of the Syrtis Major volcanic province. This expansion is performed once, cached as a JSON lookup keyed by observation ID, and reused for all subsequent training.

The expanded paragraphs are then encoded by a frozen sentence transformer (such as sentence-T5-base, 768-d, or all-MiniLM-L6-v2, 384-d). The sentence encoder is frozen — it adds zero trainable parameters to MarsCLIP. A single learnable linear projection W_T ∈ ℝ^(d_text × d) maps the sentence embedding to the shared d-dimensional space.

**Why frozen.** The expanded rationales provide a fixed target that anchors the visual embedding to geological semantics. Fine-tuning the text encoder would allow it to co-adapt with the vision encoder, potentially collapsing to a trivial solution where both encoders learn to predict observation ID rather than geology. Freezing the text encoder ensures the visual encoder must learn features that genuinely correspond to geological meaning.

The output t_i ∈ ℝ^d is L2-normalised.

### B.7 Context target: location-text fusion

The contrastive alignment target is not the text embedding alone or the location embedding alone — it is their fusion. Geology is deeply tied to planetary location: polar ice exists at high latitudes, volcanic flows concentrate on the Tharsis bulge, aeolian dunes cluster in low-pressure basins. A pure text target would ignore this spatial prior; a pure location target would ignore scientific intent.

The fused context vector z_i is computed by cross-attention between the text and location embeddings, followed by a gated residual connection:

```
α = softmax(Q_t · K_l^T / √d)    (text queries, location keys)
z_cross = α · V_l                  (location values weighted by text attention)
z_i = LayerNorm(t_i + gate · z_cross)
```

where gate is a learnable scalar initialised to 0 (so the model starts with a pure text target and gradually learns to incorporate location). This allows the model to learn that "pyroxene stratigraphy" has different visual signatures at Syrtis Major (low-albedo volcanic province) versus Nili Fossae (high-albedo phyllosilicate-bearing terrain), even though the text rationale might be similar.

The output z_i ∈ ℝ^d is L2-normalised. This is the "ground truth" that the vision encoder must match: a representation of what surface geology should look like at this location, given the scientific context, independent of illumination or viewing angle.

### B.8 Loss function 1: soft-target InfoNCE

Standard CLIP uses a one-hot cross-entropy loss: image i should match text i and no other. This fails when a patch physically straddles two HiRISE strips with different rationales, or when multiple observations cover the same terrain with different descriptions. The target distribution is not one-hot — it is a soft distribution weighted by physical area fractions.

Let P(i, j) be the ground-truth probability that patch i should match context j. For most patches, P is one-hot (the patch comes from a single observation). For boundary patches where pixel area fractions w_1 and w_2 from two overlapping strips contribute, P distributes mass accordingly. Let Q(i, j) = softmax(v_i · z_j / τ) be the model's predicted distribution.

The loss is the KL divergence from P to Q:

```
L_soft = −(1/N) Σ_i Σ_j P(i,j) · log Q(i,j)
```

This reduces to standard InfoNCE when P is one-hot, so there is no cost to using the soft formulation universally. The temperature τ is a learnable parameter initialised to 0.07, following CLIP (Radford et al., 2021).

This loss is computed twice per batch: once for local crops (L_soft(v_L, z)) and once for global crops (L_soft(v_G, z)), and the two terms are summed.

### B.9 Loss function 2: cross-area contrastive (CACo) with IoVA gating

The CACo loss (Ayush et al., 2021) enforces that local and global crops of the same location embed in the same neighbourhood:

```
L_CACo = −(1/ΣΓ_i) Σ_i Γ_i · log [ exp(v_{L,i} · v_{G,i} / τ_s) / Σ_j exp(v_{L,i} · v_{G,j} / τ_s) ]
```

The IoVA (Intersection over Valid Area) gating function Γ_i prevents the loss from computing gradients on pairs where the local crop falls in a nodata region of the global crop (or vice versa). This happens when CCD stagger creates irregular data boundaries within a single observation.

```
IoVA = Area(M_{L,valid} ∩ M_{G,valid}) / Area(I_L)
Γ_i  = 1 if IoVA > τ_overlap (default 0.3), else 0
```

Without IoVA gating, the model would learn to align valid-data embeddings with nodata-region embeddings, collapsing a portion of the manifold to a meaningless attractor.

### B.10 Total loss

The complete multimodal alignment objective combines the soft-target alignment loss (applied at both scales) and the CACo cross-scale consistency loss:

```
L_total = L_soft(v_L, z) + L_soft(v_G, z) + λ · L_CACo(v_L, v_G)
```

where λ is a weighting hyperparameter (default 0.5) that balances semantic alignment against scale consistency. During the first few epochs, λ can be annealed from 0 to its target value, allowing the vision encoder to first learn to match text/location semantics before the cross-scale constraint is imposed.

### B.11 Training configuration

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Batch size | 512 | Contrastive learning requires large batches for sufficient negatives (Radford et al., 2021) |
| Embedding dim d | 512 | Matches SatCLIP; sufficient for ~500 geological categories |
| Temperature τ | 0.07 (learnable) | CLIP default; allows the model to sharpen or soften the distribution |
| Scale temperature τ_s | 0.10 (fixed) | Slightly warmer — cross-scale alignment is inherently noisier |
| Optimizer | AdamW, lr=1e-4, weight decay=0.05 | Standard for ViT fine-tuning |
| Schedule | Cosine decay with 5-epoch warmup | Prevents early instability from the contrastive loss |
| ViT learning rate | 1e-5 (10× smaller than new heads) | Encoder is pretrained from Stage A; lower rate prevents catastrophic forgetting |
| SH degree L | 40 | Matches SatCLIP-ViT16-L40 configuration |
| RFF matrix σ | 10.0 | Controls the frequency bandwidth of the geometry encoding |

### B.12 What the multimodal alignment block produces

After multimodal alignment, each encoder can be used independently:

**Vision encoder E_V** maps any HiRISE patch (with its viewing geometry) to an illumination-invariant d-dimensional vector on the unit hypersphere. Patches showing similar geology embed nearby, regardless of solar angle, season, or CCD binning.

**Location encoder E_L** maps any (lat, lon) coordinate on Mars to a d-dimensional vector that summarises the visual and geological characteristics of that location, as learned from the entire HiRISE dataset. This is a general-purpose Mars location embedding — the Martian equivalent of what SatCLIP provides for Earth.

**Text encoder E_T** (frozen, plus projection head) maps any geological description to a d-dimensional vector in the same space as the vision and location embeddings. This enables zero-shot retrieval: embed a text query like "layered deposits in canyon walls" and find the nearest image embeddings without any task-specific training.

**Context encoder** (E_L ⊕ E_T fusion) produces the joint location-text representation z_i. This is the "what should be here" vector that can serve as a pseudo-label for unsupervised geological mapping.

---

## Data pipeline extensions required

The existing MarsHiRISE dataset and HiRISEGeoSampler require the following extensions to support the two-stage pretraining:

**1. Extended metadata parsing.** `_ProductMeta.from_lbl` must parse the `VIEWING_PARAMETERS` group in addition to the existing calibration constants. Six new regex patterns extract `INCIDENCE_ANGLE`, `EMISSION_ANGLE`, `PHASE_ANGLE`, `LOCAL_TIME`, `SOLAR_LONGITUDE`, and `SUB_SOLAR_AZIMUTH` as floats.

**2. Enriched sample dict.** `__getitem__` must return additional fields beyond `{"image", "bounds", "crs"}`:

| Field | Type | Source |
|-------|------|--------|
| `geometry_vector` | float32[6] | LBL viewing parameters |
| `rationale` | str | RATIONALE_DESC from spatial index |
| `location` | float32[2] | Patch centroid (cx, cy) from sampler |
| `valid_channels` | bool[3] | Which spectral bands have data |
| `nodata_mask` | bool[H, W] | Per-pixel validity (from calibration fix) |
| `map_scale` | float32 | MAP_SCALE from LBL (GSD in m/px) |

**3. Paired-crop sampler.** `HiRISEGeoSampler` must yield `(I_L, I_G)` pairs for multimodal alignment. For each valid centre, the sampler checks that the 3× global crop also intersects the strip polygon. Centres where the global crop extends beyond the strip are excluded from the paired-crop set. Stage A uses the existing single-crop sampler.

**4. Rationale expansion cache.** A JSON file mapping each unique `RATIONALE_DESC` to its LLM-expanded paragraph. Generated once offline. The spatial index GeoDataFrame carries a `rationale` column (added during `_build_spatial_index`) so the mapping from patch → observation → rationale is O(1) at load time.

**5. Observation-to-rationale column.** The spatial index GeoDataFrame gains a `rationale` column populated from the cumulative index during `_build_spatial_index`. This avoids re-parsing the LBL at every `__getitem__` call.

---

## References

The architecture integrates techniques from the following peer-reviewed sources, listed in the order they appear in the pipeline:

1. He, K. et al. (2022). Masked Autoencoders Are Scalable Vision Learners. CVPR. — MAE pretraining paradigm, 75% masking ratio, asymmetric decoder (Stage A core).
2. Cong, Y. et al. (2022). SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery. NeurIPS. — Independent spectral band tokenization, multi-spectral masked autoencoder for EO (§A.3).
3. Reed, C. et al. (2023). Scale-MAE: A Scale-Aware Masked Autoencoder for Multiscale Geospatial Representation Learning. ICCV. — Continuous GSD sinusoidal encoding S(ρ) (§A.5).
4. Dosovitskiy, A. et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR. — ViT encoder architecture (§A.6).
5. Klemmer, K. et al. (2025). SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery. AAAI. — Contrastive location-image pretraining framework (multimodal alignment core).
6. Rußwurm, M. et al. (2024). Geographic Location Encoding with Spherical Harmonics and Sinusoidal Representation Networks. ICLR. — Spherical harmonic location encoder + SirenNet (§B.5).
7. Perez, E. et al. (2018). FiLM: Visual Reasoning with a General Conditioning Layer. AAAI. — Feature-wise Linear Modulation for conditional invariance (§B.3).
8. Tancik, M. et al. (2020). Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains. NeurIPS. — Random Fourier Features for angular encoding (§B.4).
9. Sitzmann, V. et al. (2020). Implicit Neural Representations with Periodic Activation Functions. NeurIPS. — SirenNet periodic activations for spatial fields (§B.5).
10. Ayush, K. et al. (2021). Geography-Aware Self-Supervised Learning. ICCV. — Cross-area contrastive loss CACo (§B.9).
11. Radford, A. et al. (2021). Learning Transferable Visual Models from Natural Language Supervision. ICML. — CLIP contrastive framework, temperature parameter (§B.8).
12. Gao, Y. et al. (2023). SoftCLIP: Softer Cross-modal Alignment Makes CLIP Stronger. arXiv. — Soft-target KL divergence loss (§B.8).
13. Tran, L. et al. (2017). Missing Modalities Imputation via Cascaded Residual Autoencoder. CVPR. — Proof that zero-filling corrupts ERM; motivation for learnable missing tokens (§A.3).
14. Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS. — Transformer architecture, sinusoidal positional encoding (§A.5, §A.6).
15. Jakubik, J. et al. (2023). Foundation Models for Generalist Geospatial Artificial Intelligence. arXiv. — Prithvi geospatial foundation model, pretraining-finetuning paradigm.
16. Mai, G. et al. (2023). CSP: Self-Supervised Contrastive Spatial Pre-Training for Geospatial-Visual Representations. ICML. — Contrastive spatial pretraining, comparison baseline for location encoders.
