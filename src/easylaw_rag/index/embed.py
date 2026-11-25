"""BGE-M3 encoder producing dense + sparse (lexical) vectors in one pass."""

from __future__ import annotations

from dataclasses import dataclass

from easylaw_rag import config  # noqa: F401  (sets HF_HOME before transformers is imported)
from easylaw_rag.config import EMBED_FP16, EMBED_MAX_TOKENS, EMBED_MODEL


@dataclass
class Encoded:
    dense: list[list[float]]
    sparse: list[dict[int, float]]  # token id -> weight


def embedding_text(context: str, text: str) -> str:
    """What gets embedded: the clean citation/title header, then the (noisy) OCR chunk."""
    return f"{context}\n{text}".strip()


class Encoder:
    def __init__(self, model_name: str = EMBED_MODEL, device: str | None = None, fp16: bool = EMBED_FP16):
        import torch
        from FlagEmbedding import BGEM3FlagModel

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = BGEM3FlagModel(model_name, use_fp16=fp16 and device == "cuda", devices=device)

    def encode(self, texts: list[str], batch_size: int = 16, max_length: int = EMBED_MAX_TOKENS) -> Encoded:
        out = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        sparse = [{int(k): float(v) for k, v in w.items()} for w in out["lexical_weights"]]
        return Encoded(dense=out["dense_vecs"].tolist(), sparse=sparse)
