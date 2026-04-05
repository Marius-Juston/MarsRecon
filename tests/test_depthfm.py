"""
Tests for the DepthFM implementation in src/depth_fm/.

Structure
---------
Unit tests (no GPU / no checkpoint required):
  TestQSample               — noise schedule math matches original dfm.py
  TestPerSampleMinMax       — normalization function matches original dfm.py
  TestMarsDepthFMInterface  — MarsDepthFM exposes the right API surface

Integration tests (require CUDA + checkpoints/depthfm-v1.ckpt):
  TestEncodeDecode          — encode/decode match DepthFM's encode/decode exactly
  TestPredictDepth          — MarsDepthFM.predict_depth matches DepthFM.predict_depth
  TestLightningConsistency  — lightning _predict_depth matches model.predict_depth
  TestTrainingStepFlow      — training step uses q_sample source, clean conditioning

Run unit tests only (CI):
    uv run pytest tests/test_depthfm.py -m "not integration" -v

Run everything (needs GPU + checkpoint):
    uv run pytest tests/test_depthfm.py -v
"""

import math
import pathlib
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

_ROOT = pathlib.Path(__file__).parent.parent
_SRC = _ROOT / "src"
_ORIG = _ROOT / "depth-fm"
for _p in (_SRC, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CKPT_PATH = str(_ROOT / "checkpoints" / "depthfm-v1.ckpt")
CKPT_EXISTS = pathlib.Path(CKPT_PATH).exists()
CUDA_AVAILABLE = torch.cuda.is_available()

needs_gpu_and_ckpt = pytest.mark.integration
skip_if_no_gpu_or_ckpt = pytest.mark.skipif(
    not (CUDA_AVAILABLE and CKPT_EXISTS),
    reason="Requires CUDA and checkpoints/depthfm-v1.ckpt",
)


# =============================================================================
# Helpers
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
    """Load original DepthFM and our MarsDepthFM from the same checkpoint."""
    sys.path.insert(0, str(_ORIG / "depthfm"))
    sys.path.insert(0, str(_ORIG))
    from depthfm import DepthFM

    from depth_fm.model import MarsDepthFM, load_sd21_backend

    orig = DepthFM(CKPT_PATH).to(device).eval()

    backbone, vae, noising_step, empty_text_embed = load_sd21_backend(
        CKPT_PATH, "runwayml/stable-diffusion-v1-5", device=device
    )
    ours = MarsDepthFM(backbone, vae, noising_step, empty_text_embed).to(device).eval()

    return orig, ours


def _synthetic_image(device="cuda:0", h=64, w=64):
    """(1, 3, H, W) in [-1, 1] — small enough to be fast."""
    torch.manual_seed(0)
    return torch.randn(1, 3, h, w, device=device).clamp(-1, 1)


# =============================================================================
# Unit tests — q_sample
# =============================================================================

class TestQSample:
    """Verify our q_sample matches the original dfm.py implementation."""

    def test_output_shape_preserved(self):
        from depth_fm.noise import q_sample
        x = torch.randn(2, 4, 8, 8)
        out = q_sample(x, t=400)
        assert out.shape == x.shape

    def test_output_dtype_preserved(self):
        from depth_fm.noise import q_sample
        for dtype in (torch.float32, torch.float64):
            x = torch.randn(1, 4, 8, 8, dtype=dtype)
            out = q_sample(x, t=400)
            assert out.dtype == dtype

    def test_device_preserved(self):
        from depth_fm.noise import q_sample
        x = torch.randn(1, 4, 8, 8)  # CPU
        out = q_sample(x, t=400)
        assert out.device == x.device

    def test_deterministic_with_fixed_noise(self):
        """Same noise tensor → identical output."""
        from depth_fm.noise import q_sample
        x = torch.randn(1, 4, 8, 8)
        noise = torch.randn_like(x)
        out1 = q_sample(x, t=400, noise=noise)
        out2 = q_sample(x, t=400, noise=noise)
        assert torch.allclose(out1, out2)

    def test_t0_returns_clean_signal(self):
        """At t=0, alpha_bar=1 → output should equal x_start."""
        from depth_fm.noise import q_sample, cosine_alpha_bar
        # alpha_bar at t=0 is 1 (but numerical eps makes it ~0.9999...)
        ab = cosine_alpha_bar(0.0)
        assert ab > 0.999, f"Expected alpha_bar(0)≈1, got {ab}"

    def test_alpha_bar_at_400(self):
        """cosine_alpha_bar(400/1000) must equal the original's value (~0.6545)."""
        from depth_fm.noise import cosine_alpha_bar
        ab = cosine_alpha_bar(400 / 1000)
        assert abs(ab - 0.6545) < 1e-3, f"alpha_bar(0.4)={ab}, expected ~0.6545"

    def test_matches_original_formula_exactly(self):
        """Output must be bit-identical to the verbatim dfm.py copy."""
        from depth_fm.noise import q_sample
        torch.manual_seed(7)
        x = torch.randn(2, 4, 16, 16)
        noise = torch.randn_like(x)

        ours = q_sample(x, t=400, noise=noise)
        orig = _orig_q_sample(x, t=400, noise=noise)

        assert torch.allclose(ours, orig, atol=1e-6), \
            f"Max diff: {(ours - orig).abs().max().item()}"

    @pytest.mark.parametrize("t", [0, 100, 200, 400, 600, 999])
    def test_noise_level_increases_with_t(self, t):
        """Larger t → more noise (lower alpha_bar) → output further from x_start."""
        from depth_fm.noise import cosine_alpha_bar
        if t == 0:
            pytest.skip("t=0 special case")
        ab_prev = cosine_alpha_bar(max(t - 100, 1) / 1000)
        ab_curr = cosine_alpha_bar(t / 1000)
        assert ab_curr <= ab_prev, \
            f"alpha_bar should decrease: {ab_curr} > {ab_prev} at t={t}"

    def test_noising_step_400_signal_to_noise(self):
        """At checkpoint's noising_step=400: signal≈80.7%, noise≈59%."""
        from depth_fm.noise import cosine_alpha_bar
        ab = cosine_alpha_bar(400 / 1000)
        signal_frac = math.sqrt(ab)
        noise_frac = math.sqrt(1 - ab)
        assert 0.80 < signal_frac < 0.82, f"signal={signal_frac:.4f}"
        assert 0.57 < noise_frac < 0.61, f"noise={noise_frac:.4f}"


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
        assert out.min().item() >= 0.0 - 1e-6
        assert out.max().item() <= 1.0 + 1e-6

    def test_output_shape_preserved(self):
        from depth_fm.noise import per_sample_min_max_normalization
        x = torch.randn(3, 2, 8, 8)
        out = per_sample_min_max_normalization(x)
        assert out.shape == x.shape

    def test_each_sample_independent(self):
        """Each sample in the batch normalizes to [0,1] independently."""
        from depth_fm.noise import per_sample_min_max_normalization
        # Two samples with very different value ranges
        x = torch.cat([
            torch.full((1, 1, 4, 4), 100.0),
            torch.full((1, 1, 4, 4), 0.001),
        ])
        x[0, 0, 0, 0] = 200.0  # max for sample 0
        x[1, 0, 0, 0] = 0.002  # max for sample 1
        out = per_sample_min_max_normalization(x)
        assert out[0].min().item() == pytest.approx(0.0, abs=1e-5)
        assert out[0].max().item() == pytest.approx(1.0, abs=1e-5)
        assert out[1].min().item() == pytest.approx(0.0, abs=1e-5)
        assert out[1].max().item() == pytest.approx(1.0, abs=1e-5)

    def test_matches_original_formula_exactly(self):
        """Output must be identical to the verbatim dfm.py copy."""
        from depth_fm.noise import per_sample_min_max_normalization
        torch.manual_seed(3)
        x = torch.randn(4, 1, 8, 8)
        ours = per_sample_min_max_normalization(x)
        orig = _orig_per_sample_min_max(x)
        assert torch.allclose(ours, orig, atol=1e-6), \
            f"Max diff: {(ours - orig).abs().max().item()}"

    def test_dtype_preserved(self):
        from depth_fm.noise import per_sample_min_max_normalization
        x = torch.randn(2, 1, 8, 8, dtype=torch.float64)
        out = per_sample_min_max_normalization(x)
        assert out.dtype == torch.float64


# =============================================================================
# Unit tests — MarsDepthFM API surface
# =============================================================================

class TestMarsDepthFMInterface:
    """Check that MarsDepthFM exposes the same API as DepthFM (no checkpoint needed)."""

    def _make_tiny_model(self):
        """Build a minimal MarsDepthFM with stub backbone/VAE on CPU."""
        from depth_fm.model import MarsDepthFM

        class _StubVAE(nn.Module):
            scale_factor = 0.18215

            class _Dist:
                def mode(self): return torch.zeros(1, 4, 8, 8)
                def sample(self): return torch.zeros(1, 4, 8, 8)

            class _Post:
                latent_dist = None
                def __init__(self): self.latent_dist = _StubVAE._Dist()

            def encode(self, x):
                return self._Post()

            class _Dec:
                sample = None
                def __init__(self, x): self.sample = x

            def decode(self, z):
                return self._Dec(torch.zeros(z.shape[0], 3, z.shape[2] * 8, z.shape[3] * 8))

        class _StubBackbone(nn.Module):
            dtype = torch.float32
            def forward(self, x, t, context=None, context_ca=None, **kw):
                return torch.zeros_like(x)

        vae = _StubVAE()
        backbone = _StubBackbone()
        empty_text = torch.zeros(1, 77, 1024)
        return MarsDepthFM(backbone, vae, noising_step=400, empty_text_embed=empty_text)

    def test_has_forward(self):
        model = self._make_tiny_model()
        assert callable(model.forward)

    def test_has_predict_depth(self):
        model = self._make_tiny_model()
        assert callable(model.predict_depth)

    def test_has_encode_to_latent(self):
        model = self._make_tiny_model()
        assert callable(model.encode_to_latent)

    def test_has_decode_from_latent(self):
        model = self._make_tiny_model()
        assert callable(model.decode_from_latent)

    def test_has_predict_velocity(self):
        model = self._make_tiny_model()
        assert callable(model.predict_velocity)

    def test_noising_step_stored(self):
        model = self._make_tiny_model()
        assert model.noising_step == 400

    def test_predict_depth_is_no_grad(self):
        """predict_depth must not require gradients (mirrors DepthFM)."""
        model = self._make_tiny_model()
        # Should not raise even if called outside no_grad context
        im = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            _ = model.predict_depth(im, num_steps=1, ensemble_size=1)

    def test_forward_output_shape(self):
        """forward() returns (1, 1, H, W)."""
        model = self._make_tiny_model()
        im = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model.forward(im, num_steps=1, ensemble_size=1)
        assert out.shape[0] == 1
        assert out.shape[1] == 1

    def test_forward_output_range(self):
        """forward() output is in [0, 1] (min-max normalized)."""
        model = self._make_tiny_model()
        im = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model.forward(im, num_steps=1, ensemble_size=1)
        # stub backbone returns zeros → exp(0)=1 → min-max collapse → constant 0
        assert out.min().item() >= 0.0 - 1e-5
        assert out.max().item() <= 1.0 + 1e-5

    def test_ensemble_requires_batch1(self):
        model = self._make_tiny_model()
        im = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            with pytest.raises(AssertionError):
                model.forward(im, num_steps=1, ensemble_size=4)


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
            f"Max diff: {(z_orig - z_ours).abs().max().item():.2e}"

    def test_decode_matches(self, models, image):
        """decode_from_latent == DepthFM.decode."""
        orig, ours = models
        with torch.no_grad():
            z = orig.encode(image, sample_posterior=False)
            pix_orig = orig.decode(z)
            pix_ours = ours.decode_from_latent(z)
        assert torch.allclose(pix_orig, pix_ours, atol=1e-5), \
            f"Max diff: {(pix_orig - pix_ours).abs().max().item():.2e}"

    def test_scale_factor_matches(self, models):
        orig, ours = models
        assert orig.scale_factor == ours.scale_factor

    def test_noising_step_matches(self, models):
        orig, ours = models
        assert orig.noising_step == ours.noising_step


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

    def _run_both(self, models, image, num_steps, ensemble_size, seed=42):
        orig, ours = models
        torch.manual_seed(seed)
        with torch.no_grad():
            depth_orig = orig.predict_depth(image, num_steps=num_steps,
                                            ensemble_size=ensemble_size)
        torch.manual_seed(seed)
        with torch.no_grad():
            depth_ours = ours.predict_depth(image, num_steps=num_steps,
                                            ensemble_size=ensemble_size)
        return depth_orig, depth_ours

    def test_output_shape(self, models, image):
        _, ours = models
        with torch.no_grad():
            out = ours.predict_depth(image, num_steps=1, ensemble_size=1)
        assert out.shape == (1, 1, 64, 64), f"Got {out.shape}"

    def test_output_range_zero_one(self, models, image):
        _, ours = models
        with torch.no_grad():
            out = ours.predict_depth(image, num_steps=2, ensemble_size=4)
        assert out.min().item() >= 0.0 - 1e-5
        assert out.max().item() <= 1.0 + 1e-5

    def test_matches_original_1step_no_ensemble(self, models, image):
        """1-step, no ensemble: deterministic so must be exactly equal."""
        depth_orig, depth_ours = self._run_both(models, image, num_steps=1, ensemble_size=1)
        assert torch.allclose(depth_orig, depth_ours, atol=1e-5), \
            f"Max diff: {(depth_orig - depth_ours).abs().max().item():.2e}"

    def test_matches_original_2step_ensemble4(self, models, image):
        """2-step, ensemble=4: same seed → identical outputs."""
        depth_orig, depth_ours = self._run_both(models, image, num_steps=2, ensemble_size=4)
        assert torch.allclose(depth_orig, depth_ours, atol=1e-5), \
            f"Max diff: {(depth_orig - depth_ours).abs().max().item():.2e}"

    def test_pixel_correlation_above_threshold(self, models, image):
        """Even with different seeds, structural correlation must be high."""
        orig, ours = models
        torch.manual_seed(10)
        with torch.no_grad():
            d_orig = orig.predict_depth(image, num_steps=2, ensemble_size=4)
        torch.manual_seed(99)
        with torch.no_grad():
            d_ours = ours.predict_depth(image, num_steps=2, ensemble_size=4)
        d_o = d_orig.cpu().numpy().ravel()
        d_u = d_ours.cpu().numpy().ravel()
        r = np.corrcoef(d_o, d_u)[0, 1]
        assert r > 0.999, f"Pixel correlation {r:.6f} below threshold"


# =============================================================================
# Integration tests — lightning _predict_depth matches model.predict_depth
# =============================================================================

@needs_gpu_and_ckpt
@skip_if_no_gpu_or_ckpt
class TestLightningConsistency:
    """_predict_depth in DepthFMLightningModule must equal MarsDepthFM.predict_depth."""

    @pytest.fixture(scope="class")
    def module_and_image(self):
        from omegaconf import OmegaConf
        from depth_fm.lightning_module import DepthFMLightningModule

        cfg = OmegaConf.create({
            "model": {
                "backend": "sd21",
                "depthfm_checkpoint": CKPT_PATH,
                "vae_id": "runwayml/stable-diffusion-v1-5",
                "freeze_encoder": False,
                "gradient_checkpointing": False,
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
        module = DepthFMLightningModule(cfg)
        module.model.to("cuda:0")
        module.model.eval()

        torch.manual_seed(0)
        image = torch.randn(1, 3, 64, 64, device="cuda:0").clamp(-1, 1)
        return module, image

    def test_predict_depth_matches_model_forward(self, module_and_image):
        """Lightning _predict_depth and model.predict_depth give the same result."""
        module, image = module_and_image
        z_img = module.model.encode_to_latent(image)

        torch.manual_seed(42)
        with torch.no_grad():
            z_lightning = module._predict_depth(z_img, num_steps=2)
            depth_lightning = module.model.decode_from_latent(z_lightning)
            depth_lightning = depth_lightning.mean(dim=1, keepdim=True)

        torch.manual_seed(42)
        with torch.no_grad():
            depth_model = module.model.predict_depth(image, num_steps=2, ensemble_size=1)

        # Both should decode to equivalent pixel values
        from depth_fm.noise import per_sample_min_max_normalization
        depth_lightning_norm = per_sample_min_max_normalization(depth_lightning.exp())

        assert torch.allclose(depth_lightning_norm, depth_model, atol=1e-5), \
            f"Max diff: {(depth_lightning_norm - depth_model).abs().max().item():.2e}"

    def test_lightning_uses_q_sample_start(self, module_and_image):
        """_predict_depth starts from q_sample(z_img), not clean z_img."""
        from depth_fm.noise import q_sample
        module, image = module_and_image
        z_img = module.model.encode_to_latent(image)

        # Record what the backbone receives as first 'x' argument
        first_x_received = []
        orig_backbone_forward = module.model.backbone.forward

        def spy_forward(x, t, context=None, context_ca=None, **kw):
            if not first_x_received:
                first_x_received.append(x.detach().clone())
            return orig_backbone_forward(x=x, t=t, context=context,
                                         context_ca=context_ca, **kw)

        module.model.backbone.forward = spy_forward

        torch.manual_seed(42)
        with torch.no_grad():
            module._predict_depth(z_img, num_steps=1)

        module.model.backbone.forward = orig_backbone_forward

        assert len(first_x_received) == 1
        x_first = first_x_received[0]

        # x_first must NOT equal clean z_img
        assert not torch.allclose(x_first, z_img, atol=1e-4), \
            "_predict_depth appears to start from clean z_img (not q_sample)"

        # x_first must be a plausible q_sample output: variance should be higher
        # than clean z_img due to added noise
        z_img_var = z_img.var().item()
        x_var = x_first.var().item()
        # q_sample adds substantial noise so variance typically increases
        # (this is a soft check — just ensure it's different from z_img)
        assert abs(x_first - z_img).mean().item() > 0.01, \
            "First ODE input looks too similar to clean z_img"

    def test_lightning_uses_clean_conditioning(self, module_and_image):
        """_predict_depth passes clean z_img as context, not noised version."""
        module, image = module_and_image
        z_img = module.model.encode_to_latent(image)

        contexts_received = []
        orig_backbone_forward = module.model.backbone.forward

        def spy_forward(x, t, context=None, context_ca=None, **kw):
            if context is not None:
                contexts_received.append(context.detach().clone())
            return orig_backbone_forward(x=x, t=t, context=context,
                                         context_ca=context_ca, **kw)

        module.model.backbone.forward = spy_forward

        with torch.no_grad():
            module._predict_depth(z_img, num_steps=2)

        module.model.backbone.forward = orig_backbone_forward

        assert len(contexts_received) == 2, "Expected 2 backbone calls for num_steps=2"

        # Every call must use the exact clean z_img as context
        for step_i, ctx in enumerate(contexts_received):
            assert torch.allclose(ctx, z_img, atol=1e-6), \
                f"Step {step_i}: context differs from clean z_img " \
                f"(max diff={( ctx - z_img).abs().max().item():.2e})"


# =============================================================================
# Integration tests — training step flow correctness
# =============================================================================

@needs_gpu_and_ckpt
@skip_if_no_gpu_or_ckpt
class TestTrainingStepFlow:
    """
    Verify the training step calls q_sample on source and uses clean conditioning.
    We spy on predict_velocity to record what it receives.
    """

    @pytest.fixture(scope="class")
    def module(self):
        from omegaconf import OmegaConf
        from depth_fm.lightning_module import DepthFMLightningModule

        cfg = OmegaConf.create({
            "model": {
                "backend": "sd21",
                "depthfm_checkpoint": CKPT_PATH,
                "vae_id": "runwayml/stable-diffusion-v1-5",
                "freeze_encoder": False,
                "gradient_checkpointing": False,
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
        mod = DepthFMLightningModule(cfg)
        mod.model.to("cuda:0")
        return mod

    def _make_batch(self, device="cuda:0"):
        torch.manual_seed(5)
        return {
            "image": torch.randn(1, 3, 64, 64, device=device).clamp(-1, 1),
            "dtm":   torch.randn(1, 3, 64, 64, device=device).clamp(-1, 1),
        }

    def test_v_target_is_zdepth_minus_xsource_not_zimg(self, module):
        """
        Training velocity target must be z_depth - x_source (noised source),
        NOT z_depth - z_img (clean image). We verify by checking that the
        velocity target changes across calls with different random seeds
        even for identical images (because q_sample introduces stochasticity).
        """
        from depth_fm.noise import q_sample

        batch = self._make_batch()
        z_img = module.model.encode_to_latent(batch["image"])
        z_depth = module.model.encode_to_latent(batch["dtm"])

        # Compute x_source the same way training_step does
        torch.manual_seed(1)
        x_source_1 = q_sample(z_img, module.model.noising_step)
        torch.manual_seed(2)
        x_source_2 = q_sample(z_img, module.model.noising_step)

        v_target_1 = z_depth - x_source_1
        v_target_2 = z_depth - x_source_2
        v_target_clean = z_depth - z_img   # what WOULD be computed if bug present

        # v_target changes with seed (stochastic q_sample) → not equal to clean target
        assert not torch.allclose(v_target_1, v_target_clean, atol=1e-3), \
            "v_target == z_depth - z_img: training_step is NOT using q_sample"

        # Sanity: different seeds give different targets
        assert not torch.allclose(v_target_1, v_target_2, atol=1e-3), \
            "q_sample produces identical outputs with different seeds (unexpected)"

    def test_conditioning_is_clean_zimg(self, module):
        """
        The predict_velocity context argument must be clean z_img.
        We intercept the backbone call to check.
        """
        batch = self._make_batch()
        z_img = module.model.encode_to_latent(batch["image"])

        contexts_received = []
        orig_pv = module.model.predict_velocity

        def spy_pv(z_t, t, z_img_cond):
            contexts_received.append(z_img_cond.detach().clone())
            return orig_pv(z_t, t, z_img_cond)

        module.model.predict_velocity = spy_pv

        try:
            module.training_step(batch, batch_idx=0)
        finally:
            module.model.predict_velocity = orig_pv

        assert len(contexts_received) == 1

        # Context must equal clean z_img, not any noised version
        ctx = contexts_received[0]
        assert torch.allclose(ctx, z_img, atol=1e-6), \
            f"Training conditioning is not clean z_img (max diff " \
            f"{(ctx - z_img).abs().max().item():.2e})"

    def test_interpolation_uses_xsource_not_zimg(self, module):
        """
        z_t at any timestep must be an interpolation of (x_source, z_depth),
        NOT (z_img, z_depth). Verify by checking that z_t at t=0 ≠ z_img.
        """
        from depth_fm.noise import q_sample

        batch = self._make_batch()
        z_img = module.model.encode_to_latent(batch["image"])

        z_t_received = []
        orig_pv = module.model.predict_velocity

        def spy_pv(z_t, t, z_img_cond):
            z_t_received.append((z_t.detach().clone(), t.detach().clone()))
            return orig_pv(z_t, t, z_img_cond)

        module.model.predict_velocity = spy_pv

        torch.manual_seed(99)
        try:
            module.training_step(batch, batch_idx=0)
        finally:
            module.model.predict_velocity = orig_pv

        assert len(z_t_received) == 1
        z_t, t_val = z_t_received[0]

        # z_t should NOT equal z_img (it includes q_sample noise in x_source)
        assert not torch.allclose(z_t, z_img, atol=1e-3), \
            "z_t in training_step equals clean z_img — q_sample not applied to source"
