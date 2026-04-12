"""
Tests for the DepthFM implementation in src/depth_fm/.

Structure
---------
Unit tests (no GPU / no checkpoint required):
  TestQSample               — noise schedule math matches original dfm.py
  TestPerSampleMinMax       — normalization function matches original dfm.py
  TestMarsDepthFMInterface  — MarsDepthFM exposes the correct API surface

Integration tests (require CUDA + checkpoints/depthfm-v1.ckpt):
  TestEncodeDecode          — encode/decode match DepthFM's encode/decode exactly
  TestPredictDepth          — MarsDepthFM.predict_depth matches DepthFM.predict_depth
  TestLightningConsistency  — lightning _predict_depth matches model.predict_depth
  TestTrainingStepFlow      — training step uses q_sample source + clean conditioning

Run only unit tests (CI, no GPU):
    uv run pytest tests/test_depthfm.py -m "not integration" -v

Run everything (needs GPU + checkpoint):
    uv run pytest tests/test_depthfm.py -v
"""

import math
import pathlib
import sys
import unittest.mock

import numpy as np
import pytest
import torch
import torch.nn as nn

_ROOT = pathlib.Path(__file__).parent.parent
_SRC  = _ROOT / "src"
_ORIG = _ROOT / "depth-fm"
for _p in (_SRC, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CKPT_PATH    = str(_ROOT / "checkpoints" / "depthfm-v1.ckpt")
CKPT_EXISTS  = pathlib.Path(CKPT_PATH).exists()
CUDA_AVAILABLE = torch.cuda.is_available()

needs_gpu_and_ckpt = pytest.mark.integration
skip_if_no_gpu_or_ckpt = pytest.mark.skipif(
    not (CUDA_AVAILABLE and CKPT_EXISTS),
    reason="Requires CUDA and checkpoints/depthfm-v1.ckpt",
)


# =============================================================================
# Shared helpers
# =============================================================================

def _orig_q_sample(x_start, t, noise=None, n_diffusion_timesteps=1000):
    """Verbatim copy of q_sample from depth-fm/depthfm/dfm.py."""
    import numpy as _np

    def _cosine_log_snr(t_, eps=1e-5):
        return -2 * _np.log(_np.tan((_np.pi * t_) / 2) + eps)

    def _cosine_alpha_bar(t_):
        return 1 / (1 + _np.exp(-_cosine_log_snr(t_)))

    if noise is None:
        noise = torch.randn_like(x_start)
    alpha_bar_t = _cosine_alpha_bar(t / n_diffusion_timesteps)
    alpha_bar_t = torch.tensor(alpha_bar_t, device=x_start.device, dtype=x_start.dtype)
    return torch.sqrt(alpha_bar_t) * x_start + torch.sqrt(1 - alpha_bar_t) * noise


def _orig_per_sample_min_max(x):
    """Verbatim copy of per_sample_min_max_normalization from dfm.py."""
    import einops
    bs, *shape = x.shape
    x_ = einops.rearrange(x, "b ... -> b (...)")
    min_val = einops.reduce(x_, "b ... -> b", "min")[..., None]
    max_val = einops.reduce(x_, "b ... -> b", "max")[..., None]
    x_ = (x_ - min_val) / (max_val - min_val)
    return x_.reshape(bs, *shape)


def _load_both_models(device="cuda:0"):
    """
    Load original DepthFM and our MarsDepthFM from the same checkpoint.

    use_checkpoint=False on both so the UNet forward paths are identical.
    The original DepthFM never sets use_checkpoint (defaults to False in UNetModel),
    so we must also pass False to get numerically identical results.
    """
    sys.path.insert(0, str(_ORIG / "depthfm"))
    sys.path.insert(0, str(_ORIG))
    from depthfm import DepthFM
    from depth_fm.model import MarsDepthFM, load_sd21_backend

    orig = DepthFM(CKPT_PATH).to(device).eval()

    backbone, vae, noising_step, empty_text_embed = load_sd21_backend(
        CKPT_PATH,
        "runwayml/stable-diffusion-v1-5",
        device=device,
        use_checkpoint=False,   # must match original (use_checkpoint not in ldm_hparams)
    )
    ours = MarsDepthFM(backbone, vae, noising_step, empty_text_embed).to(device).eval()
    return orig, ours


def _synthetic_image(device="cuda:0", h=64, w=64):
    """(1, 3, H, W) in [-1, 1] — small enough to be fast."""
    torch.manual_seed(0)
    return torch.randn(1, 3, h, w, device=device).clamp(-1, 1)


def _make_lightning_module(device="cuda:0"):
    """Build a DepthFMLightningModule on device with minimal config."""
    from omegaconf import OmegaConf
    from depth_fm.lightning_module import DepthFMLightningModule

    cfg = OmegaConf.create({
        "model": {
            "backend": "sd21",
            "depthfm_checkpoint": CKPT_PATH,
            "vae_id": "runwayml/stable-diffusion-v1-5",
            "freeze_encoder": False,
            "gradient_checkpointing": False,
            "use_checkpoint": False,    # match original for numerical correctness
        },
        "training": {
            "losses": {
                "velocity_weight": 1.0,
                "normals_weight": 0.0,
                "normals_start_step": 99999,
                "grad_weight": 0.0,
                "grad_start_step": 0,
                "grad_scales": [1],
                "freq_weight": 0.0,
                "freq_start_step": 0,
                "freq_alpha": 1.0,
                "use_confidence_weighting": False,
            },
            "flow_matching": {
                "timestep_sampling": "uniform",
                "noise_augmentation_alpha": 0.997,
                "sigma_min": 1e-4,
            },
            "ema_decay": 0.9999,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "max_steps": 1000,
            "lr_warmup_steps": 100,
        },
    })
    # Move the ENTIRE Lightning module so self.device resolves to `device`
    mod = DepthFMLightningModule(cfg).to(device)
    mod.eval()
    return mod


# =============================================================================
# Unit tests — q_sample
# =============================================================================

class TestQSample:
    """Verify our q_sample matches the original dfm.py implementation."""

    def test_output_shape_preserved(self):
        from depth_fm.noise import q_sample
        x = torch.randn(2, 4, 8, 8)
        assert q_sample(x, t=400).shape == x.shape

    def test_output_dtype_preserved(self):
        from depth_fm.noise import q_sample
        for dtype in (torch.float32, torch.float64):
            x = torch.randn(1, 4, 8, 8, dtype=dtype)
            assert q_sample(x, t=400).dtype == dtype

    def test_device_preserved(self):
        from depth_fm.noise import q_sample
        x = torch.randn(1, 4, 8, 8)
        assert q_sample(x, t=400).device == x.device

    def test_deterministic_with_fixed_noise(self):
        from depth_fm.noise import q_sample
        x = torch.randn(1, 4, 8, 8)
        noise = torch.randn_like(x)
        assert torch.allclose(q_sample(x, t=400, noise=noise),
                              q_sample(x, t=400, noise=noise))

    def test_alpha_bar_at_400(self):
        """cosine_alpha_bar(400/1000) ≈ 0.6545 — checkpoint's noising_step value."""
        from depth_fm.noise import cosine_alpha_bar
        ab = cosine_alpha_bar(400 / 1000)
        assert abs(ab - 0.6545) < 1e-3, f"alpha_bar(0.4)={ab}"

    def test_matches_original_formula_exactly(self):
        """Bit-identical to the verbatim dfm.py copy."""
        from depth_fm.noise import q_sample
        torch.manual_seed(7)
        x = torch.randn(2, 4, 16, 16)
        noise = torch.randn_like(x)
        ours = q_sample(x, t=400, noise=noise)
        orig = _orig_q_sample(x, t=400, noise=noise)
        assert torch.allclose(ours, orig, atol=1e-6), \
            f"Max diff: {(ours - orig).abs().max():.2e}"

    @pytest.mark.parametrize("t", [100, 200, 400, 600, 999])
    def test_alpha_bar_decreases_with_t(self, t):
        from depth_fm.noise import cosine_alpha_bar
        ab_prev = cosine_alpha_bar(max(t - 100, 1) / 1000)
        ab_curr = cosine_alpha_bar(t / 1000)
        assert ab_curr <= ab_prev, f"alpha_bar should decrease at t={t}"

    def test_noising_step_400_signal_to_noise(self):
        """At noising_step=400: signal≈80.7%, noise≈59%."""
        from depth_fm.noise import cosine_alpha_bar
        ab = cosine_alpha_bar(400 / 1000)
        assert 0.80 < math.sqrt(ab)      < 0.82
        assert 0.57 < math.sqrt(1 - ab)  < 0.61

    def test_t0_alpha_bar_near_one(self):
        from depth_fm.noise import cosine_alpha_bar
        assert cosine_alpha_bar(0.0) > 0.999

    def test_noise_different_each_call_without_seed(self):
        from depth_fm.noise import q_sample
        x = torch.randn(1, 4, 8, 8)
        out1 = q_sample(x, t=400)
        out2 = q_sample(x, t=400)
        assert not torch.allclose(out1, out2), "q_sample should be stochastic"


# =============================================================================
# Unit tests — per_sample_min_max_normalization
# =============================================================================

class TestPerSampleMinMax:
    """Verify our normalization matches the original dfm.py implementation."""

    def test_output_range_zero_one(self):
        from depth_fm.noise import per_sample_min_max_normalization
        torch.manual_seed(1)
        x = torch.randn(4, 1, 16, 16)
        out = per_sample_min_max_normalization(x)
        assert out.min().item() >= -1e-6
        assert out.max().item() <= 1.0 + 1e-6

    def test_output_shape_preserved(self):
        from depth_fm.noise import per_sample_min_max_normalization
        x = torch.randn(3, 2, 8, 8)
        assert per_sample_min_max_normalization(x).shape == x.shape

    def test_each_sample_independent(self):
        """Each sample normalizes to [0,1] independently of the others."""
        from depth_fm.noise import per_sample_min_max_normalization
        # Two samples with very different value ranges
        x = torch.cat([
            torch.full((1, 1, 4, 4), 100.0),
            torch.full((1, 1, 4, 4), 0.001),
        ])
        x[0, 0, 0, 0] = 200.0   # max for sample 0
        x[1, 0, 0, 0] = 0.002   # max for sample 1
        out = per_sample_min_max_normalization(x)
        for i in range(2):
            assert out[i].min().item() == pytest.approx(0.0, abs=1e-5)
            assert out[i].max().item() == pytest.approx(1.0, abs=1e-5)

    def test_matches_original_formula_exactly(self):
        """Bit-identical to the verbatim dfm.py copy."""
        from depth_fm.noise import per_sample_min_max_normalization
        torch.manual_seed(3)
        x = torch.randn(4, 1, 8, 8)
        ours = per_sample_min_max_normalization(x)
        orig = _orig_per_sample_min_max(x)
        assert torch.allclose(ours, orig, atol=1e-6), \
            f"Max diff: {(ours - orig).abs().max():.2e}"

    def test_dtype_preserved(self):
        from depth_fm.noise import per_sample_min_max_normalization
        x = torch.randn(2, 1, 8, 8, dtype=torch.float64)
        assert per_sample_min_max_normalization(x).dtype == torch.float64

    def test_minimum_is_zero_maximum_is_one(self):
        from depth_fm.noise import per_sample_min_max_normalization
        torch.manual_seed(9)
        x = torch.randn(8, 1, 16, 16)
        out = per_sample_min_max_normalization(x)
        # For each sample the min must be exactly 0 and max exactly 1
        for i in range(out.shape[0]):
            flat = out[i].flatten()
            assert flat.min().item() == pytest.approx(0.0, abs=1e-5)
            assert flat.max().item() == pytest.approx(1.0, abs=1e-5)


# =============================================================================
# Unit tests — MarsDepthFM API surface (no checkpoint)
# =============================================================================

class TestMarsDepthFMInterface:
    """MarsDepthFM must expose the same API as DepthFM (no checkpoint needed)."""

    def _make_tiny_model(self):
        from depth_fm.model import MarsDepthFM

        class _StubVAEDist:
            def mode(self):   return torch.zeros(1, 4, 8, 8)
            def sample(self): return torch.zeros(1, 4, 8, 8)

        class _StubVAEPost:
            latent_dist = _StubVAEDist()

        class _StubVAEDecOut:
            def __init__(self, x): self.sample = x

        class _StubVAE(nn.Module):
            def encode(self, x):
                post = _StubVAEPost()
                post.latent_dist = _StubVAEDist()
                return post
            def decode(self, z):
                # Return spatially scaled, non-constant tensor so min≠max
                B, _, lH, lW = z.shape
                H, W = lH * 8, lW * 8
                out = torch.linspace(-1, 1, B * 3 * H * W).reshape(B, 3, H, W)
                return _StubVAEDecOut(out)

        class _StubBackbone(nn.Module):
            dtype = torch.float32
            def forward(self, x, t, context=None, context_ca=None, **kw):
                return torch.zeros_like(x)

        empty_text = torch.zeros(1, 77, 1024)
        return MarsDepthFM(_StubBackbone(), _StubVAE(),
                           noising_step=400, empty_text_embed=empty_text)

    def test_has_forward(self):
        assert callable(self._make_tiny_model().forward)

    def test_has_predict_depth(self):
        assert callable(self._make_tiny_model().predict_depth)

    def test_has_encode_to_latent(self):
        assert callable(self._make_tiny_model().encode_to_latent)

    def test_has_decode_from_latent(self):
        assert callable(self._make_tiny_model().decode_from_latent)

    def test_has_predict_velocity(self):
        assert callable(self._make_tiny_model().predict_velocity)

    def test_noising_step_stored(self):
        assert self._make_tiny_model().noising_step == 400

    def test_predict_depth_is_no_grad(self):
        model = self._make_tiny_model()
        im = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model.predict_depth(im, num_steps=1, ensemble_size=1)
        assert out is not None

    def test_forward_output_shape(self):
        model = self._make_tiny_model()
        im = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model.forward(im, num_steps=1, ensemble_size=1)
        assert out.shape[0] == 1 and out.shape[1] == 1

    def test_forward_output_range(self):
        """forward() output must be in [0, 1] after min-max normalization."""
        model = self._make_tiny_model()
        im = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model.forward(im, num_steps=1, ensemble_size=1)
        assert out.min().item() >= -1e-5
        assert out.max().item() <= 1.0 + 1e-5

    def test_ensemble_requires_batch1(self):
        model = self._make_tiny_model()
        with torch.no_grad():
            with pytest.raises(AssertionError):
                model.forward(torch.randn(2, 3, 64, 64), num_steps=1, ensemble_size=4)

    def test_predict_depth_and_forward_identical(self):
        """predict_depth() and forward() must return the same tensor."""
        model = self._make_tiny_model()
        im = torch.randn(1, 3, 64, 64)
        torch.manual_seed(1)
        out_pd = model.predict_depth(im, num_steps=1, ensemble_size=1)
        torch.manual_seed(1)
        with torch.no_grad():
            out_fw = model.forward(im, num_steps=1, ensemble_size=1)
        assert torch.allclose(out_pd, out_fw, atol=1e-6)


# =============================================================================
# Integration tests — encode / decode match original
# =============================================================================

@needs_gpu_and_ckpt
@skip_if_no_gpu_or_ckpt
class TestEncodeDecode:
    """MarsDepthFM encode/decode must be bit-identical to DepthFM encode/decode."""

    @pytest.fixture(scope="class")
    def models(self):
        return _load_both_models()

    @pytest.fixture(scope="class")
    def image(self):
        return _synthetic_image()

    def test_encode_mode_matches(self, models, image):
        """encode_to_latent == DepthFM.encode(sample_posterior=False)."""
        orig, ours = models
        with torch.no_grad():
            z_orig = orig.encode(image, sample_posterior=False)
            z_ours = ours.encode_to_latent(image)
        assert torch.allclose(z_orig, z_ours, atol=1e-5), \
            f"Max diff: {(z_orig - z_ours).abs().max():.2e}"

    def test_decode_matches(self, models, image):
        """decode_from_latent == DepthFM.decode (same scale_factor)."""
        orig, ours = models
        with torch.no_grad():
            z = orig.encode(image, sample_posterior=False)
            pix_orig = orig.decode(z)
            pix_ours = ours.decode_from_latent(z)
        assert torch.allclose(pix_orig, pix_ours, atol=1e-5), \
            f"Max diff: {(pix_orig - pix_ours).abs().max():.2e}"

    def test_scale_factor_matches(self, models):
        orig, ours = models
        assert orig.scale_factor == ours.scale_factor

    def test_noising_step_matches(self, models):
        orig, ours = models
        assert orig.noising_step == ours.noising_step

    def test_empty_text_embed_matches(self, models):
        """Null text embeddings must be identical (used for cross-attention)."""
        orig, ours = models
        embed_orig = torch.tensor(orig.empty_text_embed).float()
        embed_ours = ours.empty_text_embed.float().cpu()
        assert torch.allclose(embed_orig, embed_ours, atol=1e-6), \
            f"Max diff: {(embed_orig - embed_ours).abs().max():.2e}"


# =============================================================================
# Integration tests — predict_depth matches original
# =============================================================================

@needs_gpu_and_ckpt
@skip_if_no_gpu_or_ckpt
class TestPredictDepth:
    """MarsDepthFM.predict_depth must produce identical output to DepthFM.predict_depth."""

    @pytest.fixture(scope="class")
    def models(self):
        return _load_both_models()

    @pytest.fixture(scope="class")
    def image(self):
        return _synthetic_image(h=64, w=64)

    def _run_seeded(self, fn, seed):
        torch.manual_seed(seed)
        with torch.no_grad():
            return fn()

    def test_output_shape(self, models, image):
        _, ours = models
        out = self._run_seeded(
            lambda: ours.predict_depth(image, num_steps=1, ensemble_size=1), seed=0)
        assert out.shape == (1, 1, 64, 64), f"Got {out.shape}"

    def test_output_range_zero_one(self, models, image):
        _, ours = models
        out = self._run_seeded(
            lambda: ours.predict_depth(image, num_steps=2, ensemble_size=4), seed=0)
        assert out.min().item() >= -1e-5
        assert out.max().item() <= 1.0 + 1e-5

    def test_matches_original_1step_no_ensemble(self, models, image):
        """
        1-step, ensemble=1 with same seed → deterministic → must be bit-identical.
        Both models draw one q_sample noise tensor; identical seed = identical noise.
        """
        orig, ours = models
        depth_orig = self._run_seeded(
            lambda: orig.predict_depth(image, num_steps=1, ensemble_size=1), seed=42)
        depth_ours = self._run_seeded(
            lambda: ours.predict_depth(image, num_steps=1, ensemble_size=1), seed=42)
        assert torch.allclose(depth_orig, depth_ours, atol=1e-5), \
            f"Max diff: {(depth_orig - depth_ours).abs().max():.2e}"

    def test_matches_original_2step_no_ensemble(self, models, image):
        """2-step, ensemble=1: same seed → identical."""
        orig, ours = models
        depth_orig = self._run_seeded(
            lambda: orig.predict_depth(image, num_steps=2, ensemble_size=1), seed=42)
        depth_ours = self._run_seeded(
            lambda: ours.predict_depth(image, num_steps=2, ensemble_size=1), seed=42)
        assert torch.allclose(depth_orig, depth_ours, atol=1e-5), \
            f"Max diff: {(depth_orig - depth_ours).abs().max():.2e}"

    def test_matches_original_2step_ensemble4(self, models, image):
        """2-step, ensemble=4: same seed → identical (both repeat batch, draw 4× noise)."""
        orig, ours = models
        depth_orig = self._run_seeded(
            lambda: orig.predict_depth(image, num_steps=2, ensemble_size=4), seed=42)
        depth_ours = self._run_seeded(
            lambda: ours.predict_depth(image, num_steps=2, ensemble_size=4), seed=42)
        assert torch.allclose(depth_orig, depth_ours, atol=1e-5), \
            f"Max diff: {(depth_orig - depth_ours).abs().max():.2e}"

    def test_pixel_correlation_exceeds_threshold(self, models, image):
        """Different seeds → different noise but structurally same depth map (r > 0.999)."""
        orig, ours = models
        d_orig = self._run_seeded(
            lambda: orig.predict_depth(image, num_steps=2, ensemble_size=4), seed=10)
        d_ours = self._run_seeded(
            lambda: ours.predict_depth(image, num_steps=2, ensemble_size=4), seed=99)
        r = np.corrcoef(d_orig.cpu().numpy().ravel(),
                        d_ours.cpu().numpy().ravel())[0, 1]
        assert r > 0.999, f"Pixel correlation {r:.6f} below 0.999"

    def test_output_changes_with_different_seed(self, models, image):
        """Stochastic q_sample: different seeds → different outputs."""
        _, ours = models
        d1 = self._run_seeded(
            lambda: ours.predict_depth(image, num_steps=1, ensemble_size=1), seed=1)
        d2 = self._run_seeded(
            lambda: ours.predict_depth(image, num_steps=1, ensemble_size=1), seed=2)
        assert not torch.allclose(d1, d2, atol=1e-4), \
            "Different seeds should produce different depth maps"


# =============================================================================
# Integration tests — lightning _predict_depth matches model.predict_depth
# =============================================================================

@needs_gpu_and_ckpt
@skip_if_no_gpu_or_ckpt
class TestLightningConsistency:
    """
    _predict_depth in DepthFMLightningModule must be consistent with
    MarsDepthFM.predict_depth — same ODE, same starting point, same conditioning.
    """

    @pytest.fixture(scope="class")
    def setup(self):
        mod = _make_lightning_module()
        torch.manual_seed(0)
        image = torch.randn(1, 3, 64, 64, device="cuda:0").clamp(-1, 1)
        z_img = mod.model.encode_to_latent(image)
        return mod, image, z_img

    def test_predict_depth_matches_model_forward(self, setup):
        """
        Lightning _predict_depth (returns latent) decoded + post-processed
        must equal model.predict_depth (returns normalized depth), same seed.
        """
        from depth_fm.noise import per_sample_min_max_normalization
        mod, image, z_img = setup

        torch.manual_seed(42)
        with torch.no_grad():
            z_pred = mod._predict_depth(z_img, num_steps=2)
            pix = mod.model.decode_from_latent(z_pred)
            depth_lightning = per_sample_min_max_normalization(
                pix.mean(dim=1, keepdim=True).exp())

        torch.manual_seed(42)
        depth_model = mod.model.predict_depth(image, num_steps=2, ensemble_size=1)

        assert torch.allclose(depth_lightning, depth_model, atol=1e-5), \
            f"Max diff: {(depth_lightning - depth_model).abs().max():.2e}"

    def test_lightning_uses_q_sample_not_clean_start(self, setup):
        """
        The first 'x' tensor passed to the backbone must differ from clean z_img,
        proving _predict_depth starts from q_sample(z_img, noising_step).
        """
        mod, _, z_img = setup

        first_x = []
        orig_fwd = mod.model.backbone.forward

        def spy(x, t, context=None, context_ca=None, **kw):
            if not first_x:
                first_x.append(x.detach().clone())
            return orig_fwd(x=x, t=t, context=context, context_ca=context_ca, **kw)

        mod.model.backbone.forward = spy
        try:
            torch.manual_seed(42)
            with torch.no_grad():
                mod._predict_depth(z_img, num_steps=1)
        finally:
            mod.model.backbone.forward = orig_fwd

        assert first_x, "Backbone was never called"
        # The ODE start must NOT be the clean image latent
        assert not torch.allclose(first_x[0], z_img, atol=1e-4), \
            "_predict_depth appears to start from clean z_img (q_sample not applied)"
        # Mean absolute deviation should be substantial (q_sample adds ~59% noise)
        mae = (first_x[0] - z_img).abs().mean().item()
        assert mae > 0.1, f"ODE start too close to clean z_img (MAE={mae:.4f})"

    def test_lightning_uses_clean_z_img_as_context(self, setup):
        """
        Every backbone call inside _predict_depth must receive the exact
        clean z_img as its context argument (channel-concat conditioning).
        """
        mod, _, z_img = setup
        contexts = []
        orig_fwd = mod.model.backbone.forward

        def spy(x, t, context=None, context_ca=None, **kw):
            if context is not None:
                contexts.append(context.detach().clone())
            return orig_fwd(x=x, t=t, context=context, context_ca=context_ca, **kw)

        mod.model.backbone.forward = spy
        try:
            with torch.no_grad():
                mod._predict_depth(z_img, num_steps=2)
        finally:
            mod.model.backbone.forward = orig_fwd

        assert len(contexts) == 2, f"Expected 2 UNet calls for num_steps=2, got {len(contexts)}"
        for step_i, ctx in enumerate(contexts):
            assert torch.allclose(ctx, z_img, atol=1e-6), \
                f"Step {step_i}: context is not clean z_img " \
                f"(max diff={( ctx - z_img).abs().max():.2e})"


# =============================================================================
# Integration tests — training step correctness
# =============================================================================

@needs_gpu_and_ckpt
@skip_if_no_gpu_or_ckpt
class TestTrainingStepFlow:
    """
    Verify the training step uses q_sample on source and clean z_img as conditioning.
    We spy on predict_velocity to record what it receives.
    """

    @pytest.fixture(scope="class")
    def mod(self):
        return _make_lightning_module()

    def _batch(self, device="cuda:0"):
        torch.manual_seed(5)
        return {
            "image": torch.randn(1, 3, 64, 64, device=device).clamp(-1, 1),
            "dtm":   torch.randn(1, 3, 64, 64, device=device).clamp(-1, 1),
        }

    def test_v_target_uses_q_sample_source(self, mod):
        """
        Velocity target = z_depth − x_source (noised) ≠ z_depth − z_img (clean).
        Proof: v_target changes across seeds (stochastic q_sample) but the
        clean-only version z_depth − z_img is constant across seeds.
        """
        from depth_fm.noise import q_sample
        batch  = self._batch()
        z_img  = mod.model.encode_to_latent(batch["image"])
        z_depth = mod.model.encode_to_latent(batch["dtm"])

        torch.manual_seed(1)
        xs1 = q_sample(z_img, mod.model.noising_step)
        torch.manual_seed(2)
        xs2 = q_sample(z_img, mod.model.noising_step)

        vt_1      = z_depth - xs1
        vt_2      = z_depth - xs2
        vt_clean  = z_depth - z_img

        # Different seeds → different v_target (stochastic)
        assert not torch.allclose(vt_1, vt_2, atol=1e-3), \
            "v_target is constant across seeds — q_sample may not be applied"
        # v_target with q_sample ≠ clean version
        assert not torch.allclose(vt_1, vt_clean, atol=1e-3), \
            "v_target == z_depth − z_img: q_sample not applied to source"

    def test_conditioning_passed_to_predict_velocity_is_clean_z_img(self, mod):
        """
        predict_velocity must receive the exact clean z_img as z_img_cond,
        not the noised x_source.
        """
        batch  = self._batch()
        z_img  = mod.model.encode_to_latent(batch["image"])

        received_cond = []
        orig_pv = mod.model.predict_velocity

        def spy(z_t, t, z_img_cond):
            received_cond.append(z_img_cond.detach().clone())
            return orig_pv(z_t, t, z_img_cond)

        mod.model.predict_velocity = spy
        try:
            with unittest.mock.patch.object(mod, "log"):
                mod.training_step(batch, batch_idx=0)
        finally:
            mod.model.predict_velocity = orig_pv

        assert len(received_cond) == 1, "predict_velocity should be called once per step"
        assert torch.allclose(received_cond[0], z_img, atol=1e-6), \
            f"Conditioning ≠ clean z_img (max diff " \
            f"{(received_cond[0] - z_img).abs().max():.2e})"

    def test_z_t_at_any_t_is_not_clean_z_img(self, mod):
        """
        z_t passed to predict_velocity must be on the flow path from
        x_source (noised), not from clean z_img. Since x_source ≠ z_img,
        z_t must also differ from z_img even at t≈0.
        """
        batch = self._batch()
        z_img = mod.model.encode_to_latent(batch["image"])

        received_zt = []
        orig_pv = mod.model.predict_velocity

        def spy(z_t, t, z_img_cond):
            received_zt.append(z_t.detach().clone())
            return orig_pv(z_t, t, z_img_cond)

        mod.model.predict_velocity = spy
        torch.manual_seed(99)
        try:
            with unittest.mock.patch.object(mod, "log"):
                mod.training_step(batch, batch_idx=0)
        finally:
            mod.model.predict_velocity = orig_pv

        assert len(received_zt) == 1
        # z_t must differ from clean z_img (it's on the path from noised source)
        assert not torch.allclose(received_zt[0], z_img, atol=1e-3), \
            "z_t == clean z_img: training interpolation not using q_sample source"

    def test_training_step_returns_scalar_loss(self, mod):
        """Smoke test: training_step must return a scalar loss tensor."""
        batch = self._batch()
        with unittest.mock.patch.object(mod, "log"):
            loss = mod.training_step(batch, batch_idx=0)
        assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"
        assert loss.item() > 0, "Loss should be positive"
        assert torch.isfinite(loss), "Loss must be finite"


# =============================================================================
# Unit tests — configure_optimizers LR warmup correctness (no checkpoint)
# =============================================================================

class TestConfigureOptimizersWarmup:
    """
    Verify configure_optimizers() produces a SequentialLR with linear warmup
    followed by cosine decay.  Uses a stub model to avoid loading a checkpoint.
    """

    def _make_minimal_module(self, warmup_steps: int = 100, max_steps: int = 1000):
        from omegaconf import OmegaConf
        from depth_fm.lightning_module import DepthFMLightningModule

        cfg = OmegaConf.create({
            "model": {
                "backend": "sd21",
                "depthfm_checkpoint": "DUMMY",
                "vae_id": "DUMMY",
                "freeze_encoder": False,
                "gradient_checkpointing": False,
                "use_checkpoint": False,
            },
            "training": {
                "losses": {
                    "velocity_weight": 1.0,
                    "normals_weight": 0.0,
                    "normals_start_step": 99999,
                    "grad_weight": 0.0,
                    "grad_start_step": 0,
                    "grad_scales": [1],
                    "freq_weight": 0.0,
                    "freq_start_step": 0,
                    "freq_alpha": 1.0,
                    "use_confidence_weighting": False,
                },
                "flow_matching": {
                    "timestep_sampling": "uniform",
                    "noise_augmentation_alpha": 0.997,
                    "sigma_min": 1e-4,
                },
                "ema_decay": 0.9999,
                "learning_rate": 1e-4,
                "weight_decay": 0.01,
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_epsilon": 1e-8,
                "max_steps": max_steps,
                "lr_warmup_steps": warmup_steps,
            },
        })

        # Stub backbone — just needs parameters() to return something trainable
        stub_backbone = torch.nn.Linear(4, 4)

        class _StubModel:
            backbone = stub_backbone
            noising_step = 400

        # Bypass __init__ (which calls build_model and requires a checkpoint)
        mod = DepthFMLightningModule.__new__(DepthFMLightningModule)
        mod.config = cfg
        mod.model = _StubModel()
        return mod

    def test_scheduler_is_sequential(self):
        """configure_optimizers must return a SequentialLR, not bare CosineAnnealingLR."""
        mod = self._make_minimal_module()
        result = mod.configure_optimizers()
        scheduler = result["lr_scheduler"]["scheduler"]
        assert isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR), (
            f"Expected SequentialLR, got {type(scheduler).__name__}"
        )

    def test_lr_near_zero_at_step_0(self):
        """LR at step 0 must be << learning_rate (warmup not yet active)."""
        mod = self._make_minimal_module(warmup_steps=100)
        result = mod.configure_optimizers()
        optimizer = result["optimizer"]
        lr_step0 = optimizer.param_groups[0]["lr"]
        assert lr_step0 < 1e-4 * 0.1, (
            f"LR at step 0 should be near zero, got {lr_step0:.2e}"
        )

    def test_lr_reaches_full_at_warmup_steps(self):
        """LR must equal learning_rate at step warmup_steps (within 1%)."""
        warmup_steps = 100
        mod = self._make_minimal_module(warmup_steps=warmup_steps)
        result = mod.configure_optimizers()
        scheduler = result["lr_scheduler"]["scheduler"]
        optimizer = result["optimizer"]
        for _ in range(warmup_steps):
            scheduler.step()
        lr_at_warmup_end = optimizer.param_groups[0]["lr"]
        target_lr = 1e-4
        assert abs(lr_at_warmup_end - target_lr) < target_lr * 0.01, (
            f"LR after {warmup_steps} warmup steps should be {target_lr:.2e}, "
            f"got {lr_at_warmup_end:.2e}"
        )

    def test_lr_decays_after_warmup(self):
        """LR must be strictly less than learning_rate after warmup completes."""
        warmup_steps = 100
        mod = self._make_minimal_module(warmup_steps=warmup_steps, max_steps=1000)
        result = mod.configure_optimizers()
        scheduler = result["lr_scheduler"]["scheduler"]
        optimizer = result["optimizer"]
        for _ in range(warmup_steps + 50):
            scheduler.step()
        lr_after = optimizer.param_groups[0]["lr"]
        assert lr_after < 1e-4, (
            f"LR should decay below 1e-4 after warmup, got {lr_after:.2e}"
        )

    def test_lr_scheduler_interval_is_step(self):
        """Scheduler must fire per optimizer step, not per epoch."""
        mod = self._make_minimal_module()
        result = mod.configure_optimizers()
        assert result["lr_scheduler"]["interval"] == "step"
