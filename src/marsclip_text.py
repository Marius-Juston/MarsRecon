"""Text composition, tokenization, and batching utilities for MarsCLIP."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-/:.]*")


def compose_rationale_text(
    rationale_raw: str,
    rationale_expanded: str | None = None,
    *,
    text_mode: str = "raw_plus_expanded",
) -> str:
    """Compose the training text from raw and optionally expanded rationale."""
    raw = str(rationale_raw).strip()
    expanded = None if rationale_expanded is None else str(rationale_expanded).strip()

    if text_mode == "raw":
        return raw
    if text_mode == "expanded":
        return expanded or raw
    if text_mode == "raw_plus_expanded":
        if expanded:
            return f"{raw}\n\nExpanded context: {expanded}"
        return raw
    raise ValueError(f"Unsupported text_mode: {text_mode}")


@dataclass
class SimpleTextTokenizer:
    """A lightweight whitespace tokenizer for early MarsCLIP prototyping."""

    vocab: dict[str, int]
    pad_token: str = "[PAD]"
    unk_token: str = "[UNK]"
    bos_token: str = "[BOS]"
    eos_token: str = "[EOS]"

    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 1) -> "SimpleTextTokenizer":
        counts: dict[str, int] = {}
        for text in texts:
            for token in cls._tokenize(text):
                counts[token] = counts.get(token, 0) + 1

        vocab = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[BOS]": 2,
            "[EOS]": 3,
        }
        for token in sorted(counts):
            if counts[token] >= min_freq and token not in vocab:
                vocab[token] = len(vocab)
        return cls(vocab=vocab)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [tok.lower() for tok in _TOKEN_RE.findall(str(text))]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.vocab[self.pad_token]

    @property
    def unk_token_id(self) -> int:
        return self.vocab[self.unk_token]

    @property
    def bos_token_id(self) -> int:
        return self.vocab[self.bos_token]

    @property
    def eos_token_id(self) -> int:
        return self.vocab[self.eos_token]

    def encode(self, text: str, max_length: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one text string into padded token ids and attention mask."""
        tokens = self._tokenize(text)
        ids = [self.bos_token_id]
        ids.extend(self.vocab.get(tok, self.unk_token_id) for tok in tokens)
        ids.append(self.eos_token_id)
        ids = ids[:max_length]

        attention = [1] * len(ids)
        if len(ids) < max_length:
            pad_len = max_length - len(ids)
            ids.extend([self.pad_token_id] * pad_len)
            attention.extend([0] * pad_len)

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(attention, dtype=torch.bool),
        )

    def batch_encode(
        self,
        texts: list[str],
        max_length: int = 64,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a list of texts into batched token ids and attention masks."""
        ids, masks = zip(*(self.encode(text, max_length=max_length) for text in texts))
        return torch.stack(list(ids)), torch.stack(list(masks))


def build_tokenizer_from_manifest(
    manifest: pd.DataFrame,
    *,
    text_mode: str = "raw_plus_expanded",
    min_freq: int = 1,
) -> SimpleTextTokenizer:
    """Build a tokenizer from manifest raw/expanded rationale text."""
    texts = [
        compose_rationale_text(
            row["rationale_desc"],
            row.get("rationale_expanded"),
            text_mode=text_mode,
        )
        for _, row in manifest.iterrows()
    ]
    return SimpleTextTokenizer.build(texts, min_freq=min_freq)


class MarsCLIPBatchCollator:
    """Batch MarsCLIPDataset samples and tokenize their text fields."""

    def __init__(
        self,
        tokenizer: SimpleTextTokenizer,
        *,
        text_mode: str = "raw_plus_expanded",
        max_length: int = 64,
    ) -> None:
        self.tokenizer = tokenizer
        self.text_mode = text_mode
        self.max_length = max_length

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [
            compose_rationale_text(
                sample["rationale_raw"],
                sample.get("rationale_expanded"),
                text_mode=self.text_mode,
            )
            for sample in samples
        ]
        input_ids, attention_mask = self.tokenizer.batch_encode(
            texts,
            max_length=self.max_length,
        )

        images = torch.stack([sample["image"] for sample in samples])
        valid_masks = torch.stack([sample["valid_mask"] for sample in samples])
        geo_features = torch.stack([sample["geo_features"] for sample in samples])
        scale_features = torch.stack([sample["scale_features"] for sample in samples])
        viewing_features = torch.stack(
            [sample["metadata"]["viewing_features"] for sample in samples]
        )
        band_presence_mask = torch.stack(
            [sample["metadata"]["band_presence_mask"] for sample in samples]
        ).to(torch.float32)
        band_valid_fraction = torch.stack(
            [sample["metadata"]["band_valid_fraction"] for sample in samples]
        )
        overall_valid_fraction = torch.tensor(
            [[float(sample["metadata"]["overall_valid_fraction"])] for sample in samples],
            dtype=torch.float32,
        )
        quality_features = torch.cat(
            [band_presence_mask, band_valid_fraction, overall_valid_fraction],
            dim=1,
        )

        return {
            "image": images,
            "valid_mask": valid_masks,
            "geo_features": geo_features,
            "scale_features": scale_features,
            "viewing_features": viewing_features,
            "quality_features": quality_features,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "rationale_raw": [sample["rationale_raw"] for sample in samples],
            "rationale_expanded": [sample.get("rationale_expanded") for sample in samples],
            "text": texts,
            "metadata": [sample["metadata"] for sample in samples],
        }
