"""Lightweight T5 text encoder helpers for stage-2 alignment."""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn

try:
    from transformers import AutoConfig, AutoModel, AutoTokenizer, T5EncoderModel
except ImportError as exc:  # pragma: no cover - optional runtime dependency
    raise ImportError(
        "transformers is required for stage_b text alignment. "
        "Install it with `uv add transformers`."
    ) from exc


def _load_pretrained_with_local_fallback(
    loader: Callable[..., Any],
    model_name: str,
) -> Any:
    """Retry Hugging Face loads from the local cache when network checks fail."""
    try:
        return loader(model_name)
    except Exception as exc:
        try:
            return loader(model_name, local_files_only=True)
        except Exception:
            raise exc


class T5Encoder(nn.Module):
    """Text encoder wrapper exposing pooled sequence embeddings."""

    def __init__(
        self,
        model_name: str = "google-t5/t5-base",
        max_length: int = 128,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        self.max_length = int(max_length)
        self.tokenizer = _load_pretrained_with_local_fallback(AutoTokenizer.from_pretrained, model_name)
        config = _load_pretrained_with_local_fallback(AutoConfig.from_pretrained, model_name)
        if getattr(config, "model_type", "") == "t5":
            # T5 is encoder-decoder; for embedding use we only need encoder states.
            self.model = _load_pretrained_with_local_fallback(T5EncoderModel.from_pretrained, model_name)
        else:
            self.model = _load_pretrained_with_local_fallback(AutoModel.from_pretrained, model_name)
        for parameter in self.model.parameters():
            parameter.requires_grad = bool(trainable)

    @staticmethod
    def _masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (hidden * weights).sum(dim=1) / denom

    def tokenize(self, texts: list[str], *, device: torch.device) -> dict[str, torch.Tensor]:
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        )
        return {key: value.to(device) for key, value in tokens.items()}

    def forward(self, texts: list[str], *, device: torch.device) -> tuple[torch.Tensor, dict[str, Any]]:
        tokens = self.tokenize(texts, device=device)
        output = self.model(**tokens)
        embedding = self._masked_mean(output.last_hidden_state, tokens["attention_mask"])
        return embedding, tokens


if __name__ == "__main__":  # pragma: no cover
    encoder = T5Encoder()
    with torch.no_grad():
        emb, batch = encoder(["Hello, Mars!"], device=torch.device("cpu"))
    print("token shape:", batch["input_ids"].shape)
    print("embedding shape:", emb.shape)
