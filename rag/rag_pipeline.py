#!/usr/bin/env python3
"""
RAG & Multi-Modal RAG Evaluation Toolkit (single-file)
======================================================

[UPDATED for Italian + robustness]
- Safer JSON extraction with bracket matching.
- Multilingual embeddings by default (E5) with query/passage prefixes.
- Accent/punctuation-insensitive normalization for EM/F1 (better Italian handling).
- Image OCR supports Italian (configurable via --ocr_langs).
- Fixed undefined _answer_image_legacy and List type checks.
- Improved vision-model detection with optional --vision_models override.
- Simpler pdfminer import path.
- Fallback when tokenizer lacks apply_chat_template.

Backwards compatible with the original CLI unless noted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import math
import time
import re
import glob
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

# at top of your RAG script
from api_models import (
    OpenAITextLLM,
    DeepSeekTextLLM,
    QwenTextLLM,
    MistralTextLLM,
    GeminiTextLLM,
)


import numpy as np

# --- Dependencies that may be optionally missing are imported lazily ---

# tiktoken for better text chunking
try:
    import tiktoken
except ImportError:
    tiktoken = None

# Utility: deterministic seed
import random
random.seed(42)
np.random.seed(42)

# ---------------------- Helpers ----------------------

# ---------------------- Model-specific chat templates (fallbacks) ----------------------

# Minimal SmolLM3 chat template: system/user/assistant + generation prompt.
SMOLLM3_FALLBACK_CHAT_TEMPLATE = """
{% for message in messages %}
{% if message.role == "system" %}
<|im_start|>system
{{ message.content }}
<|im_end|>
{% elif message.role == "user" %}
<|im_start|>user
{{ message.content }}
<|im_end|>
{% elif message.role == "assistant" %}
<|im_start|>assistant
{{ message.content }}
<|im_end|>
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
<|im_start|>assistant
{% endif %}
"""

# Minimal Apertus chat template: system (optional) + user/assistant + generation prompt.
APERTUS_FALLBACK_CHAT_TEMPLATE = """
{{ bos_token if bos_token is defined else "" }}
{% set system_msgs = messages | selectattr("role", "equalto", "system") | list %}
{% if system_msgs %}
<|system_start|>
{% for m in system_msgs %}
{{ m.content }}
{% endfor %}
<|system_end|>
{% endif %}
{% for message in messages if message.role != "system" %}
{% if message.role == "user" %}
<|user_start|>
{{ message.content }}
<|user_end|>
{% elif message.role == "assistant" %}
<|assistant_start|>
{{ message.content }}
<|assistant_end|>
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
<|assistant_start|>
{% endif %}
"""




def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# Accent/punctuation tolerant normalization (better for Italian)
import unicodedata
WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+", re.UNICODE)


def _strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if not unicodedata.combining(ch))


def normalize_text(s: str) -> str:
    s = _strip_accents(s)
    s = s.casefold()
    toks = WORD_RE.findall(s)
    return " ".join(toks)


def token_f1(pred: str, gold: str) -> float:
    p_toks = normalize_text(pred).split()
    g_toks = normalize_text(gold).split()
    if not p_toks and not g_toks:
        return 1.0
    if not p_toks or not g_toks:
        return 0.0
    from collections import Counter
    pc = Counter(p_toks)
    gc = Counter(g_toks)
    overlap = sum((pc & gc).values())
    prec = overlap / max(1, sum(pc.values()))
    rec = overlap / max(1, sum(gc.values()))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _regex_salvage_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Last-resort heuristic to salvage a JSON object from noisy text.

    Strategy:
    - Strip common Markdown fences like ```json.
    - Use a regex to grab the first {...} block.
    - Try to json.loads() it.
    - Return a dict if parsing succeeds, else None.
    """
    # Remove code fences that often wrap JSON
    cleaned = re.sub(r"```(?:json)?", "", text)

    # Grab the largest {...} span we can find
    m = re.search(r"\{[\s\S]*\}", cleaned)
    if not m:
        return None

    candidate = m.group(0)
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None



def extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON extractor: direct load, ```json fenced, or first balanced {...}."""
    # 1) direct
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # 2) fenced
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # 3) first balanced {...}
    stack = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if stack == 0:
                start = i
            stack += 1
        elif ch == "}":
            if stack > 0:
                stack -= 1
            if stack == 0 and start != -1:
                candidate = text[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    break  # candidate is not valid JSON, we'll try regex salvage

    # 4) LAST RESORT: regex salvage
    salvaged = _regex_salvage_json(text)
    if salvaged is not None:
        return salvaged

    return None



# ---------------------- FAISS index (legacy corpora mode) ----------------------

class FaissIndex:
    def __init__(self, dim: int):
        self.dim = dim
        try:
            import faiss
        except Exception as e:
            raise RuntimeError("faiss is required: pip install faiss-cpu") from e
        self.faiss = faiss
        self.index = self.faiss.IndexFlatIP(dim)  # cosine via normalized vectors
        self.vectors = None  # (n, dim)
        self.meta: List[Dict[str, Any]] = []

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-10
        return x / norms

    def add(self, X: np.ndarray, meta: List[Dict[str, Any]]):
        assert X.shape[1] == self.dim
        Xn = self._l2_normalize(X.astype("float32"))
        if self.vectors is None:
            self.vectors = Xn
        else:
            self.vectors = np.vstack([self.vectors, Xn])
        self.index.add(Xn)
        self.meta.extend(meta)

    def search(self, q: np.ndarray, k: int) -> List[Dict[str, Any]]:
        qn = self._l2_normalize(q.astype("float32"))
        D, I = self.index.search(qn, k)
        out = []
        for row_i, (dists, idxs) in enumerate(zip(D, I)):
            hits = []
            for d, idx in zip(dists, idxs):
                if idx < 0 or idx >= len(self.meta):
                    continue
                item = dict(self.meta[idx])
                item["score"] = float(d)
                hits.append(item)
            out.append(hits)
        return out


# ---------------------- Embeddings ----------------------

class TextEmbedder:
    def __init__(self, name: str = "intfloat/multilingual-e5-base", device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name, device=device)
        self.name = name
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        embs = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        return embs.astype("float32")


class CLIPEmbedder:
    def __init__(self, name: str = "ViT-H-14", pretrained: str = "laion2b_s32b_b79k", device: Optional[str] = None):
        import torch
        import open_clip
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained, device=self.device)
        self.tokenizer = open_clip.get_tokenizer(name)
        self.dim = self.model.text_projection.shape[1]

    def encode_text(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        feats = []
        for i in range(0, len(texts), batch_size):
            toks = self.tokenizer(texts[i : i + batch_size]).to(self.device)
            with self.torch.no_grad():
                txt = self.model.encode_text(toks)
            feats.append(txt.cpu().numpy())
        X = np.concatenate(feats, axis=0)
        return X.astype("float32")

    def encode_images(self, images: List[Any], batch_size: int = 8) -> np.ndarray:
        feats = []
        for i in range(0, len(images), batch_size):
            ims = [self.preprocess(img).unsqueeze(0) for img in images[i : i + batch_size]]
            ims = self.torch.cat(ims, dim=0).to(self.device)
            with self.torch.no_grad():
                im = self.model.encode_image(ims)
            feats.append(im.cpu().numpy())
        X = np.concatenate(feats, axis=0)
        return X.astype("float32")


# ---------------------- Token-based chunking utilities ----------------------


def _get_tiktoken_encoder():
    if tiktoken is None:
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def _whitespace_chunk(text: str, max_tokens: int, overlap: int) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    step = max(1, max_tokens - overlap)
    while i < len(words):
        j = min(len(words), i + max_tokens)
        chunk = " ".join(words[i:j])
        if chunk.strip():
            chunks.append(chunk)
        if j == len(words):
            break
        i += step
    return chunks


def chunk_text(text: str, max_tokens: int = 300, overlap: int = 60) -> List[str]:
    """Chunk text with tiktoken if available; fallback to whitespace chunks.
    Ensures forward progress even if overlap >= max_tokens.
    """
    enc = _get_tiktoken_encoder()
    if enc is None:
        return _whitespace_chunk(text, max_tokens, overlap)

    try:
        tokens = enc.encode(text)
        chunks = []
        i = 0
        step = max(1, max_tokens - overlap)
        while i < len(tokens):
            j = min(len(tokens), i + max_tokens)
            chunk_tokens = tokens[i:j]
            chunk_text_ = enc.decode(chunk_tokens)
            if chunk_text_.strip():
                chunks.append(chunk_text_)
            if j == len(tokens):
                break
            i += step
        return chunks
    except Exception:
        # conservative fallback
        return _whitespace_chunk(text, max_tokens, overlap)


def rank_chunks_by_query(embedder: TextEmbedder, query: str, chunks: List[str], top_k: int = 5) -> List[str]:
    """Return top-k chunk strings by cosine similarity to the query.
    If using E5 models, we apply query/passage prefixes.
    """
    if not chunks:
        return []
    use_e5 = "e5" in getattr(embedder, "name", "").lower()
    q_text = f"query: {query}" if use_e5 else query
    c_texts = [f"passage: {c}" for c in chunks] if use_e5 else chunks

    # Encode
    q_emb = embedder.encode([q_text])  # (1, d)
    c_embs = embedder.encode(c_texts)  # (n, d)
    # Normalize
    qn = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)
    cn = c_embs / (np.linalg.norm(c_embs, axis=1, keepdims=True) + 1e-10)
    # Cosine scores
    scores = (cn @ qn.T).squeeze(-1)  # (n,)
    idx = np.argsort(-scores)[:top_k]
    return [chunks[int(i)] for i in idx]


# ---------------------- Corpora ingestion (legacy mode) ----------------------


class TextCorpus:
    def __init__(self, embedder: TextEmbedder):
        self.embedder = embedder
        self.index = FaissIndex(embedder.dim)
        self.chunks: List[str] = []
        self.chunk_meta: List[Dict[str, Any]] = []
        self.tokenizer = _get_tiktoken_encoder()
        if self.tokenizer is None:
            print("[WARN] tiktoken not installed or failed to load. Falling back to basic whitespace chunker.")

    def _extract_text(self, path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext in [".txt", ".md"]:
            return open(path, "r", encoding="utf-8", errors="ignore").read()
        if ext == ".pdf":
            try:
                from pdfminer_high_level import extract_text  # type: ignore
            except Exception:
                try:
                    from pdfminer.high_level import extract_text  # type: ignore
                except Exception:
                    return ""  # pdfminer not installed
            try:
                return extract_text(path)
            except Exception:
                return ""
        return ""

    def _chunk(self, text: str, max_tokens: int = 300, overlap: int = 80) -> List[str]:
        return chunk_text(text, max_tokens=max_tokens, overlap=overlap)

    def ingest_dir(self, root: str, max_tokens: int = 300, overlap: int = 80):
        paths = [
            p
            for p in glob.glob(os.path.join(root, "**/*"), recursive=True)
            if os.path.isfile(p) and os.path.splitext(p)[1].lower() in [".txt", ".md", ".pdf"]
        ]
        texts = []
        metas = []
        for p in paths:
            txt = self._extract_text(p)
            if not txt:
                continue
            for ch in self._chunk(txt, max_tokens=max_tokens, overlap=overlap):
                texts.append(ch)
                metas.append({"doc_path": p, "type": "text", "chunk": ch[:200]})
        if not texts:
            return
        X = self.embedder.encode(texts)
        self.index.add(X, metas)
        self.chunks.extend(texts)
        self.chunk_meta.extend(metas)

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        use_e5 = "e5" in getattr(self.embedder, "name", "").lower()
        q_text = f"query: {query}" if use_e5 else query
        q = self.embedder.encode([q_text])
        hits = self.index.search(q, k=k)[0]
        return hits


class ImageCorpus:
    def __init__(self, clip: CLIPEmbedder, ocr_langs: Optional[List[str]] = None):
        from PIL import Image
        self.Image = Image
        self.clip = clip
        self.index = FaissIndex(clip.dim)
        self.images: List[str] = []
        self.image_meta: List[Dict[str, Any]] = []
        self.captions: Dict[str, str] = {}
        self.ocr_text: Dict[str, str] = {}
        self.ocr_langs = ocr_langs or ["it", "en"]

    def _load_image(self, path: str):
        try:
            return self.Image.open(path).convert("RGB")
        except Exception:
            return None

    def _make_captioner(self):
        # optional captioner via HF pipeline (BLIP)
        try:
            from transformers import pipeline
            return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        except Exception:
            return None

    def _make_ocr(self):
        # optional OCR via easyocr with configurable languages
        try:
            import easyocr
            return easyocr.Reader(self.ocr_langs, gpu=False)
        except Exception:
            return None

    def ingest_dir(self, root: str, caption: bool = True, ocr: bool = True):
        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        paths = [
            p
            for p in glob.glob(os.path.join(root, "**/*"), recursive=True)
            if os.path.isfile(p) and os.path.splitext(p)[1].lower() in img_exts
        ]
        if not paths:
            return
        images = []
        metas = []
        for p in paths:
            img = self._load_image(p)
            if img is None:
                continue
            images.append(img)
            # document id = parent folder (sequence grouping)
            doc_id = os.path.basename(os.path.dirname(p))
            if doc_id == "" or doc_id == os.path.basename(root):
                doc_id = os.path.splitext(os.path.basename(p))[0]
            metas.append({"doc_path": p, "doc_id": doc_id, "type": "image"})
        if not images:
            return
        X = self.clip.encode_images(images)
        self.index.add(X, metas)
        for p, m in zip(paths, metas):
            self.images.append(p)
            self.image_meta.append(m)

        # optional enrichments
        cap = self._make_captioner() if caption else None
        reader = self._make_ocr() if ocr else None
        if cap is not None or reader is not None:
            print("Running image enrichments (caption/OCR)...")
            for p in paths:
                img = self._load_image(p)
                if img is None:
                    continue
                if cap is not None:
                    try:
                        out = cap(img)
                        self.captions[p] = out[0]["generated_text"] if out else ""
                    except Exception:
                        self.captions[p] = ""
                if reader is not None:
                    try:
                        import numpy as _np
                        res = reader.readtext(_np.array(img))
                        self.ocr_text[p] = " ".join([t[1] for t in res])
                    except Exception:
                        self.ocr_text[p] = ""
            print("Enrichments complete.")

    def retrieve(self, query: str, k: int = 6) -> List[Dict[str, Any]]:
        q = self.clip.encode_text([query])
        hits = self.index.search(q, k=k)[0]
        # attach captions/OCR if available
        for h in hits:
            p = h.get("doc_path", "")
            h["caption"] = self.captions.get(p, "")
            h["ocr"] = self.ocr_text.get(p, "")
        return hits


# ---------------------- Model wrappers ----------------------

@dataclass
class GenerateConfig:
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 0.95


class TextLLM:
    def __init__(self, name: str, device_map: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.torch = torch

        # tokenizer first
        self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)

        # --- model-specific flags ---
        lname = name.lower()
        self.is_smollm3 = "smollm3-3b" in lname or "smollm3" in lname
        self.is_apertus = "apertus-8b-instruct-2509" in lname or (
            "apertus-8b" in lname and "instruct" in lname
        )

        # --- model loading (special-case Apertus) ---
        if self.is_apertus:
            # Follow HF snippet: no torch_dtype, no device_map
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = AutoModelForCausalLM.from_pretrained(name).to(device)
        else:
            # Keep old behaviour for all other models
            self.model = AutoModelForCausalLM.from_pretrained(
                name,
                device_map=device_map,
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )


        # If HF didn’t ship a chat_template (e.g., some quantized forks), attach a minimal one.
        tmpl = getattr(self.tokenizer, "chat_template", None)
        if not tmpl:
            if self.is_smollm3:
                self.tokenizer.chat_template = SMOLLM3_FALLBACK_CHAT_TEMPLATE
            elif self.is_apertus:
                self.tokenizer.chat_template = APERTUS_FALLBACK_CHAT_TEMPLATE

    # Accepts messages and uses apply_chat_template if available; fallback to plain prompt
    def generate(self, messages: List[Dict[str, str]], gen: GenerateConfig) -> str:
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_id

        # Make a shallow copy so we don't mutate the caller's list when we tweak system prompts.
        local_messages = [dict(m) for m in messages]

        # For SmolLM3 we optionally prepend `/no_think` to the system message as a soft switch.
        if self.is_smollm3:
            for m in local_messages:
                if m.get("role") == "system":
                    content = m.get("content", "")
                    if "/think" not in content and "/no_think" not in content:
                        m["content"] = "/no_think\n\n" + content
                    break
            else:
                # No system message present -> add a small one to disable thinking.
                local_messages.insert(0, {"role": "system", "content": "/no_think"})

        # Try chat template first
        try:
            chat_kwargs = {
                "add_generation_prompt": True,
                "return_tensors": "pt",
            }

            # For SmolLM3 and Apertus we explicitly disable thinking / deliberation if supported.
            if self.is_smollm3 or self.is_apertus:
                chat_kwargs["enable_thinking"] = False

            applied = self.tokenizer.apply_chat_template(
                local_messages,
                **chat_kwargs,
            )

            if isinstance(applied, self.torch.Tensor):
                inputs = {"input_ids": applied.to(self.model.device)}
            else:
                # BatchEncoding -> dict of tensors
                inputs = {k: v.to(self.model.device) for k, v in dict(applied).items()}

        except TypeError:
            # e.g., template doesn't accept enable_thinking; retry without it.
            try:
                applied = self.tokenizer.apply_chat_template(
                    local_messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                if isinstance(applied, self.torch.Tensor):
                    inputs = {"input_ids": applied.to(self.model.device)}
                else:
                    inputs = {k: v.to(self.model.device) for k, v in dict(applied).items()}
            except Exception:
                # fall through to plain prompt fallback
                inputs = None
        except Exception:
            inputs = None

        if inputs is None:
            # Fallback: join system+user to a single prompt (no chat template)
            sys_txt = "\n\n".join([m["content"] for m in local_messages if m.get("role") == "system"]) or ""
            usr_txt = "\n\n".join([m["content"] for m in local_messages if m.get("role") == "user"]) or ""
            prompt = (sys_txt + "\n\n" + usr_txt).strip()
            enc = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in dict(enc).items()}

        # Ensure 2D batch shape
        input_ids = inputs["input_ids"]
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            inputs["input_ids"] = input_ids
        input_token_len = input_ids.shape[-1]

        if "attention_mask" not in inputs:
            inputs["attention_mask"] = self.torch.ones_like(inputs["input_ids"])


        gen_kwargs = {
            "max_new_tokens": gen.max_new_tokens,
            "pad_token_id": pad_id,
            "eos_token_id": eos_id,
        }

        # Only use sampling flags when temperature > 0
        if gen.temperature > 0:
            gen_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": gen.temperature,
                    "top_p": gen.top_p,
                }
            )

        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                **gen_kwargs,
            )


        # Slice new tokens robustly
        if out.dim() == 2:
            new_tokens = out[0, input_token_len:]
        else:
            new_tokens = out[input_token_len:]

        try:
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        except Exception:
            text = self.tokenizer.batch_decode(new_tokens.unsqueeze(0), skip_special_tokens=True)[0]
        return text.strip()


class VisionLLM:
    """Wrapper for VLMs that support HF AutoProcessor with images in chat template (e.g., Qwen2-VL, LLaVA-OneVision)."""

    def __init__(self, name: str, device_map: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        self.name = name
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            name,
            device_map=device_map,
            trust_remote_code=True,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

    def generate(self, prompt: str, images: List[Any], gen: GenerateConfig) -> str:
        messages = [
            {
                "role": "user",
                "content": ([{"type": "text", "text": prompt}] + [{"type": "image", "image": img} for img in images]),
            }
        ]

        tok = self.processor.tokenizer
        eos_id = getattr(tok, "eos_token_id", None)
        if eos_id is None:
            # fallback if tokenizer lacks eos id
            eos_id = tok.convert_tokens_to_ids(tok.eos_token) if getattr(tok, "eos_token", None) else None
        pad_id = getattr(tok, "pad_token_id", None)
        if pad_id is None and eos_id is not None:
            pad_id = eos_id

        try:
            # Text (chat) side
            applied = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
            if isinstance(applied, self.torch.Tensor):
                text_inputs = {"input_ids": applied.to(self.model.device)}
            else:
                text_inputs = {k: v.to(self.model.device) for k, v in dict(applied).items()}

            # Vision side
            vision_inputs = self.processor(images=images, return_tensors="pt")
            # Some processors return BatchEncoding with .to(); otherwise move tensors manually
            if hasattr(vision_inputs, "to"):
                vision_inputs = vision_inputs.to(self.model.device)
            else:
                vision_inputs = {k: v.to(self.model.device) for k, v in dict(vision_inputs).items()}

            # Ensure 2D input_ids
            input_ids = text_inputs["input_ids"]
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                text_inputs["input_ids"] = input_ids
            input_token_len = input_ids.shape[-1]

            # Build generation kwargs depending on whether we sample or go greedy
            gen_kwargs = {
                "max_new_tokens": gen.max_new_tokens,
                "pad_token_id": pad_id,
                "eos_token_id": eos_id,
            }
    
            if gen.temperature > 0:
                # Sampling mode: use temperature + top_p
                gen_kwargs.update(
                    {
                        "do_sample": True,
                        "temperature": gen.temperature,
                        "top_p": gen.top_p,
                    }
                )
            else:
                # Greedy mode: no sampling flags
                gen_kwargs.update(
                    {
                        "do_sample": False,
                    }
                )
    
            with self.torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    **gen_kwargs,
                )

            new_tokens = out[0, input_token_len:] if out.dim() == 2 else out[input_token_len:]
            # Prefer tokenizer.decode from processor
            text = tok.decode(new_tokens, skip_special_tokens=True)
        except Exception:
            # Fallback path if chat template or joint processing fails
            proc = self.processor(text=prompt, images=images, return_tensors="pt")
            if hasattr(proc, "to"):
                proc = proc.to(self.model.device)
            else:
                proc = {k: v.to(self.model.device) for k, v in dict(proc).items()}

            # Establish input length from input_ids
            input_ids = proc["input_ids"]
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                proc["input_ids"] = input_ids
            input_token_len = input_ids.shape[-1]

            with self.torch.no_grad():
                out = self.model.generate(
                    **proc,
                    max_new_tokens=gen.max_new_tokens,
                    do_sample=(gen.temperature > 0),
                    temperature=gen.temperature,
                    top_p=gen.top_p,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                )
            new_tokens = out[0, input_token_len:] if out.dim() == 2 else out[input_token_len:]
            try:
                text = tok.decode(new_tokens, skip_special_tokens=True)
            except Exception:
                from transformers import AutoTokenizer
                alt_tok = AutoTokenizer.from_pretrained(self.name)
                text = alt_tok.decode(new_tokens, skip_special_tokens=True)

        return text.strip()


# ---------------------- RAG prompts ----------------------

TEXT_SYSTEM_PROMPT = """
Sei un assistente estrattivo. Rispondi SOLO usando il CONTENUTO nel blocco "Context".
Se l’informazione non è nel contesto, rispondi esattamente: "Non presente nel contesto".
Usa l’italiano. NON inventare nulla. NON fare deduzioni esterne.

DEVI RESTITUIRE SOLO JSON **VALIDO** (senza testo aggiuntivo prima o dopo) con questo schema ESATTO:
{
  "answer": "<risposta concisa>",
  "evidence": [
    {"chunk_id": <numero>, "quote": "<frase esatta dal contesto>"}
  ],
  "chunks_used": [<numeri>],
  "status": "ok|non_present|ambiguous"
}

Regole:
- Cita frasi brevi esatte dal testo come "quote".
- I "chunk_id" si riferiscono alle etichette [Chunk N] nel Context.
- Se trovi più valori in conflitto, usa "status":"ambiguous" e includi tutte le citazioni rilevanti.
- Se non trovi nulla, usa "status":"non_present" e "answer":"Non presente nel contesto".
"""

VISION_SYSTEM_PROMPT = (
    "You are a helpful assistant. You will see one or more images and optional captions/OCR text. "
    "Answer the user's question using the visual evidence and textual hints. If unclear, say you don't know."
)


def build_text_prompt_messages(
    question: str, contexts: List[str], prompt_template: Optional[str] = None
) -> Tuple[List[Dict[str, str]], str]:
    """Build a chat message list for text models from contexts and question.
    If `prompt_template` is provided, it may contain `{context}` and/or `{question}` placeholders.
    Returns: (messages, user_prompt_text)
    """
    ctx = "\n\n".join([f"[Chunk {i+1}]\n" + c for i, c in enumerate(contexts)])

    if prompt_template:
        try:
            user_prompt = prompt_template.format(context=ctx, question=question)
        except Exception:
            user_prompt = f"Context:\n{ctx}\n\nQuestion: {question}"
    else:
        user_prompt = f"Context:\n{ctx}\n\nQuestion: {question}"

    messages = [
        {"role": "system", "content": TEXT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return messages, user_prompt


def build_vision_prompt_from_text(question: str, hint_text: Optional[str]) -> str:
    hint = f"\nHints:\n{hint_text}" if (hint_text and hint_text.strip()) else ""
    return f"{VISION_SYSTEM_PROMPT}Question: {question}{hint}"


# ---------------------- Evaluation orchestration ----------------------

@dataclass
class QAResult:
    qid: str
    model: str
    modality: str  # text or image
    question: str
    gold: str
    answer: str
    em: float
    f1: float
    contexts: List[str]


class RagasScorer:
    def __init__(self, use: bool, judge_model: Optional[str] = None):
        self.enabled = use
        self.judge_model = judge_model
        if self.enabled:
            try:
                from ragas.metrics import (
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                )
                try:
                    from ragas.metrics import (
                        multimodal_relevance,
                        multimodal_faithfulness,
                    )  # type: ignore
                    self.has_mm = True
                    self.mm_metrics = [multimodal_relevance, multimodal_faithfulness]
                except Exception:
                    self.has_mm = False
                    self.mm_metrics = []
                self.metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
            except Exception as e:
                print("[WARN] ragas not available or incompatible:", e)
                print("[INFO] To use RAGAS, run: pip install ragas==0.1.7")
                self.enabled = False

    def score_batch(self, batch: List[QAResult]) -> Dict[str, Dict[str, float]]:
        """Returns mapping: (qid,model,modality)-> metric dict"""
        if not self.enabled:
            return {}
        try:
            import pandas as pd
            from datasets import Dataset
            from ragas import evaluate

            rows = []
            for r in batch:
                if r.modality == "text":
                    contexts = r.contexts
                else:
                    contexts = r.contexts
                rows.append(
                    {
                        "question": r.question,
                        "answer": r.answer,
                        "contexts": contexts,
                        "ground_truth": r.gold,
                        "id": f"{r.qid}::{r.model}::{r.modality}",
                    }
                )
            ds = Dataset.from_list(rows)

            metrics = self.metrics
            if getattr(self, "has_mm", False):
                print("[INFO] Multimodal RAGAS metrics detected (may require custom judge setup).")
                metrics += self.mm_metrics

            result = evaluate(ds, metrics=metrics)
            df = result.to_pandas()
            out = {}
            for _, row in df.iterrows():
                key = row["id"]
                out[key] = {m.__name__: float(row[m.__name__]) for m in metrics if m.__name__ in row}
            return out
        except Exception as e:
            print("[WARN] ragas evaluation failed:", e)
            return {}


# ---------------------- Runner ----------------------


class Runner:
    def __init__(self, args):
        self.args = args
        self.text_embedder: Optional[TextEmbedder] = None  # lazy
        self.clip_embedder: Optional[CLIPEmbedder] = None  # lazy
        self.text_corpus = None
        self.image_corpus = None
        self.gen = GenerateConfig(max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p)
        self.ragas = RagasScorer(args.use_ragas, args.judge_model)
        # New flags
        self.fail_on_missing_text = args.fail_on_missing_text
        self.fail_on_missing_question = args.fail_on_missing_question
        self.debug = args.debug
        self.vision_overrides = set(args.vision_models or [])

    def _ensure_text_embedder(self):
        if self.text_embedder is None:
            self.text_embedder = TextEmbedder(self.args.text_embedder)

    def _ensure_clip_embedder(self):
        if self.clip_embedder is None:
            self.clip_embedder = CLIPEmbedder(self.args.clip_backbone, self.args.clip_weights)

    def load_corpora(self):
        # Legacy mode: directories
        if self.args.text_corpus:
            print(f"Ingesting text corpus from: {self.args.text_corpus}")
            self._ensure_text_embedder()
            self.text_corpus = TextCorpus(self.text_embedder)
            self.text_corpus.ingest_dir(
                self.args.text_corpus,
                max_tokens=self.args.text_chunk_tokens,
                overlap=self.args.text_chunk_overlap,
            )
            print("Text ingestion complete.")
        if self.args.image_corpus:
            print(f"Ingesting image corpus from: {self.args.image_corpus}")
            self._ensure_clip_embedder()
            self.image_corpus = ImageCorpus(self.clip_embedder, ocr_langs=self.args.ocr_langs)
            self.image_corpus.ingest_dir(
                self.args.image_corpus, caption=not self.args.no_caption, ocr=not self.args.no_ocr
            )
            print("Image ingestion complete.")

    def _is_vision_model(self, name: str) -> bool:
        # Prefer explicit override if provided
        if name in self.vision_overrides:
            return True
        lname = name.lower()
        # Heuristic without the overly broad "-v"
        return any(x in lname for x in ["-vl", "onevision", "idefics", "vision"])  # heuristic

    def _make_model(self, name: str):
        # Closed-source APIs selected by prefixes in --models

        # OpenAI:   --models openai:gpt-4o-mini
        if name.startswith("openai:"):
            base = name.split(":", 1)[1]
            return OpenAITextLLM(base)

        # DeepSeek: --models deepseek:deepseek-chat
        if name.startswith("deepseek:"):
            base = name.split(":", 1)[1] if ":" in name else "deepseek-chat"
            return DeepSeekTextLLM(base)

        # Qwen:     --models qwen:qwen-plus
        if name.startswith("qwen:"):
            base = name.split(":", 1)[1] if ":" in name else "qwen-plus"
            return QwenTextLLM(base)

        # Mistral:  --models mistral:mistral-large-latest
        if name.startswith("mistral:"):
            base = name.split(":", 1)[1] if ":" in name else "mistral-large-latest"
            return MistralTextLLM(base)

        # Gemini:   --models gemini:gemini-2.5-flash
        if name.startswith("gemini:"):
            base = name.split(":", 1)[1] if ":" in name else "gemini-2.5-flash"
            return GeminiTextLLM(base)

        # Otherwise, fall back to HF models (your original logic)
        if self._is_vision_model(name):
            return VisionLLM(name)
        else:
            return TextLLM(name)




    # ---------- NEW: dataset validator ----------
    def _validate_qa(self, qa: List[Dict[str, Any]]):
        """Validate required fields in the QA JSONL before running."""
        missing_q = []
        missing_t = []
        for idx, row in enumerate(qa, 1):
            q = row.get("question", "")
            t = row.get("text", "")
            if not isinstance(q, str) or not q.strip():
                missing_q.append(idx)
            # 'text' required only if no legacy --text_corpus is provided
            if self.args.text_corpus is None and (not isinstance(t, str) or not t.strip()):
                missing_t.append(idx)

        # Hard fail according to flags
        if self.fail_on_missing_question and missing_q:
            raise ValueError(
                f"{len(missing_q)} row(s) missing/empty 'question' "
                f"(first indices: {missing_q[:10]}{', ...' if len(missing_q) > 10 else ''})"
            )
        if self.fail_on_missing_text and missing_t:
            raise ValueError(
                f"{len(missing_t)} row(s) missing/empty 'text' and no --text_corpus provided "
                f"(first indices: {missing_t[:10]}{', ...' if len(missing_t) > 10 else ''})"
            )

        # Always warn (even if not failing)
        if missing_q:
            print(f"[WARN] {len(missing_q)} row(s) missing/empty 'question'. These rows may be skipped or error.")
        if missing_t and self.args.text_corpus is None:
            print(
                f"[WARN] {len(missing_t)} row(s) missing/empty 'text' and no --text_corpus. These rows may error or be skipped."
            )

    # ---- New per-row TEXT answering with JSON parsing ----
    def _answer_text_per_row(
        self, model: TextLLM, q: Dict[str, Any]
    ) -> Tuple[str, List[str], Dict[str, Any], str, List[Dict[str, Any]], List[int], str]:
        """Returns (ans, contexts, prompt_log, status, evidence, chunks_used_ids, raw_output)."""
        self._ensure_text_embedder()
        raw_text = q.get("text", "") or ""
        if not isinstance(raw_text, str):
            raw_text = str(raw_text)
        chunks = chunk_text(
            raw_text, max_tokens=self.args.text_chunk_tokens, overlap=self.args.text_chunk_overlap
        )
        contexts = rank_chunks_by_query(self.text_embedder, q["question"], chunks, top_k=self.args.k)

        if self.debug:
            print("\n[DEBUG] QID:", q.get("id", q.get("qid", "<no-id>")))
            print("[DEBUG] QUESTION:", q.get("question", ""))
            for i, c in enumerate(contexts, 1):
                print(f"[DEBUG] CONTEXT {i} (first 500 chars):\n{c[:500]}\n---")

        messages, user_prompt = build_text_prompt_messages(
            q["question"], contexts, prompt_template=q.get("prompt_template")
        )
        raw = model.generate(messages, self.gen)
        parsed = extract_json_obj(raw)

        if isinstance(parsed, dict) and "answer" in parsed:
            ans = str(parsed.get("answer", "")).strip()
            status = str(parsed.get("status", "") or "ok")
            evidence = parsed.get("evidence", [])
            chunks_used_ids = parsed.get("chunks_used", [])
        else:
            # Fallback if model didn't follow the schema
            ans = raw.strip()
            status = "unstructured"
            evidence = []
            chunks_used_ids = []

        prompt_log = {
            "chat_messages": messages,
            "user_prompt": user_prompt,
            "parsed": parsed if isinstance(parsed, dict) else None,
        }
        return ans, contexts, prompt_log, status, evidence, chunks_used_ids, raw

    # ---- Legacy answering using corpora with JSON parsing ----
    def _answer_text_legacy(
        self, model: TextLLM, q: Dict[str, Any]
    ) -> Tuple[str, List[str], Dict[str, Any], str, List[Dict[str, Any]], List[int], str]:
        hits = self.text_corpus.retrieve(q["question"], k=self.args.k)
        contexts = [h.get("chunk", h.get("text", "")) for h in hits]
        messages, user_prompt = build_text_prompt_messages(q["question"], contexts)
        raw = model.generate(messages, self.gen)
        parsed = extract_json_obj(raw)

        if isinstance(parsed, dict) and "answer" in parsed:
            ans = str(parsed.get("answer", "")).strip()
            status = str(parsed.get("status", "") or "ok")
            evidence = parsed.get("evidence", [])
            chunks_used_ids = parsed.get("chunks_used", [])
        else:
            ans = raw.strip()
            status = "unstructured"
            evidence = []
            chunks_used_ids = []

        prompt_log = {
            "chat_messages": messages,
            "user_prompt": user_prompt,
            "parsed": parsed if isinstance(parsed, dict) else None,
        }
        return ans, contexts, prompt_log, status, evidence, chunks_used_ids, raw

    # ---- New per-row IMAGE answering ----
    def _parse_image_paths(self, q: Dict[str, Any]) -> List[str]:
        paths = []
        for key in ("img_path", "img_path_copied"):
            val = q.get(key)
            if not val:
                continue
            if isinstance(val, list):  # FIX: use list, not typing.List
                paths.extend([p for p in val if isinstance(p, str)])
            elif isinstance(val, str):
                # support comma/semicolon separated
                parts = re.split(r"[,;]", val)
                paths.extend([p.strip() for p in parts if p.strip()])
        # filter existing files only
        paths = [p for p in paths if os.path.isfile(p)]
        return paths

    def _answer_image_per_row(self, model: VisionLLM, q: Dict[str, Any]) -> Tuple[str, List[str], str]:
        """Returns (answer, contexts(textual hints), prompt_str)."""
        from PIL import Image
        img_paths = self._parse_image_paths(q)
        images = []
        for p in img_paths:
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception:
                pass
        if not images:
            raise RuntimeError("No valid image files found for row.")
        hint_text = q.get("text") if isinstance(q.get("text"), str) else None
        prompt = build_vision_prompt_from_text(q["question"], hint_text)
        ans = model.generate(prompt, images, self.gen)
        # we return contexts as textual hints only (for ragas)
        contexts = [hint_text] if hint_text else []
        return ans, contexts, prompt

    # ---- NEW: Legacy image answering (previously undefined) ----
    def _answer_image_legacy(self, model: VisionLLM, q: Dict[str, Any]) -> Tuple[str, List[str], str]:
        """Retrieve top-K images from legacy corpus and answer. Returns (ans, contexts, prompt)."""
        from PIL import Image
        hits = self.image_corpus.retrieve(q["question"], k=self.args.k)
        img_paths = [h.get("doc_path") for h in hits if h.get("doc_path")]
        images = []
        hints: List[str] = []
        for h in hits:
            p = h.get("doc_path")
            if not p:
                continue
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception:
                pass
            if h.get("caption"):
                hints.append(h.get("caption", ""))
            if h.get("ocr"):
                hints.append(h.get("ocr", ""))
        if not images:
            raise RuntimeError("No valid images retrieved from legacy image corpus.")
        hint_text = " ".join([t for t in hints if t]) or None
        prompt = build_vision_prompt_from_text(q["question"], hint_text)
        ans = model.generate(prompt, images, self.gen)
        contexts = [hint_text] if hint_text else []
        return ans, contexts, prompt

    def run(self):
        # Load data
        qa = read_jsonl(self.args.qa)
        if not qa:
            print("[WARN] Empty QA file.")
            return
        total_examples = len(qa) * len(self.args.models)
        processed_examples = 0    

        # Validate the dataset before proceeding
        self._validate_qa(qa)

        # Try legacy corpora only if explicitly provided
        if self.args.text_corpus or self.args.image_corpus:
            self.load_corpora()

        results: List[QAResult] = []
        rows_out: List[Dict[str, Any]] = []

        for model_name in self.args.models:
            print(f"\n=== Running model: {model_name} ===")
            model = self._make_model(model_name)

            for q in qa:
                qid = str(q.get("id", q.get("qid", "")))
                question = q.get("question", "")
                gold_val = q.get("answer")
                gold = "" if gold_val is None else str(gold_val)

                # Fail-fast per-row if question is missing and user requested strictness
                if self.fail_on_missing_question and (not isinstance(question, str) or not question.strip()):
                    raise ValueError(
                        f"Row with id={qid or '<no-id>'} missing 'question' and --fail_on_missing_question was set."
                    )

                did_any = False

                # TEXT path: prefer per-row `text`; else legacy corpus if provided
                try:

                    # Treat any non-vision model as a TEXT model (HF or API)
                    if not isinstance(model, VisionLLM) and isinstance(q.get("text", None), str) and q.get("text"):
                        ans, contexts, prompt_log, status, evidence, chunks_used_ids, raw_output = \
                            self._answer_text_per_row(model, q)

                        em = 1.0 if normalize_text(ans) == normalize_text(str(gold)) else 0.0
                        f1 = token_f1(ans, str(gold))

                        results.append(
                            QAResult(
                                qid=qid,
                                model=model_name,
                                modality="text",
                                question=question,
                                gold=str(gold),
                                answer=ans,
                                em=em,
                                f1=f1,
                                contexts=contexts,
                            )
                        )
                        row = dict(q)
                        row.update(
                            {
                                "model": model_name,
                                "modality": "text",
                                "model_answer": ans,
                                "em": em,
                                "f1": f1,
                                "chunks_used": json.dumps(contexts, ensure_ascii=False),
                                "question_used": question,
                                "full_prompt": json.dumps(prompt_log.get("chat_messages", []), ensure_ascii=False),
                                "raw_model_output": raw_output,
                                "status": status,
                                "evidence": json.dumps(evidence, ensure_ascii=False),
                                "chunks_used_ids": json.dumps(chunks_used_ids, ensure_ascii=False),
                            }
                        )
                        rows_out.append(row)
                        did_any = True

                    elif not isinstance(model, VisionLLM) and self.text_corpus is not None:
                        ans, contexts, prompt_log, status, evidence, chunks_used_ids, raw_output = \
                            self._answer_text_legacy(model, q)

                        em = 1.0 if normalize_text(ans) == normalize_text(str(gold)) else 0.0
                        f1 = token_f1(ans, str(gold))

                        results.append(
                            QAResult(
                                qid=qid,
                                model=model_name,
                                modality="text",
                                question=question,
                                gold=str(gold),
                                answer=ans,
                                em=em,
                                f1=f1,
                                contexts=contexts,
                            )
                        )
                        row = dict(q)
                        row.update(
                            {
                                "model": model_name,
                                "modality": "text",
                                "model_answer": ans,
                                "em": em,
                                "f1": f1,
                                "chunks_used": json.dumps(contexts, ensure_ascii=False),
                                "question_used": question,
                                "full_prompt": json.dumps(prompt_log.get("chat_messages", []), ensure_ascii=False),
                                "raw_model_output": raw_output,
                                "status": status,
                                "evidence": json.dumps(evidence, ensure_ascii=False),
                                "chunks_used_ids": json.dumps(chunks_used_ids, ensure_ascii=False),
                            }
                        )
                        rows_out.append(row)
                        did_any = True
                    
                    elif isinstance(model, TextLLM) and self.text_corpus is None:
                        # Per-row text is required when no legacy text corpus is provided
                        raise ValueError("Missing 'text' for this row and no --text_corpus provided.")
                except Exception as e:
                    print(f"[ERR][text] Failed on qid {qid or 'N/A'}: {e}")
                    import traceback
                    traceback.print_exc()

                # IMAGE path: prefer per-row images if present and model is vision;
                # else legacy image corpus if provided and model is vision
                try:
                    has_row_images = bool(self._parse_image_paths(q))
                    if isinstance(model, VisionLLM) and has_row_images:
                        ans, contexts, prompt_repr = self._answer_image_per_row(model, q)
                        em = 1.0 if normalize_text(ans) == normalize_text(str(gold)) else 0.0
                        f1 = token_f1(ans, str(gold))
                        results.append(
                            QAResult(
                                qid=qid,
                                model=model_name,
                                modality="image",
                                question=question,
                                gold=str(gold),
                                answer=ans,
                                em=em,
                                f1=f1,
                                contexts=contexts,
                            )
                        )
                        row = dict(q)
                        row.update(
                            {
                                "model": model_name,
                                "modality": "image",
                                "model_answer": ans,
                                "em": em,
                                "f1": f1,
                                # NEW LOGGING FIELDS
                                "chunks_used": json.dumps(contexts, ensure_ascii=False),  # textual hints (if any)
                                "question_used": question,
                                "full_prompt": prompt_repr,  # string used alongside images
                            }
                        )
                        rows_out.append(row)
                        did_any = True
                    elif isinstance(model, VisionLLM) and self.image_corpus is not None:
                        ans, contexts, prompt_repr = self._answer_image_legacy(model, q)
                        em = 1.0 if normalize_text(ans) == normalize_text(str(gold)) else 0.0
                        f1 = token_f1(ans, str(gold))
                        results.append(
                            QAResult(
                                qid=qid,
                                model=model_name,
                                modality="image",
                                question=question,
                                gold=str(gold),
                                answer=ans,
                                em=em,
                                f1=f1,
                                contexts=contexts,
                            )
                        )
                        row = dict(q)
                        row.update(
                            {
                                "model": model_name,
                                "modality": "image",
                                "model_answer": ans,
                                "em": em,
                                "f1": f1,
                                # NEW LOGGING FIELDS
                                "chunks_used": json.dumps(contexts, ensure_ascii=False),
                                "question_used": question,
                                "full_prompt": prompt_repr,
                            }
                        )
                        rows_out.append(row)
                        did_any = True
                except Exception as e:
                    print(f"[ERR][image] Failed on qid {qid or 'N/A'}: {e}")
                    import traceback
                    traceback.print_exc()

                if not did_any:
                    # No modality applicable; skip silently but note once per row
                    pass

                # ---- PROGRESS PRINT ----
                processed_examples += 1
                remaining = total_examples - processed_examples
                print(
                    f"[PROGRESS] Processed {processed_examples}/{total_examples} examples "
                    f"({remaining} left)...",
                    end="\r",
                    flush=True,
                )                

        # Save CSV
        import pandas as pd
        if not rows_out:
            print("[WARN] No results were generated. Exiting.")
            return

        df = pd.DataFrame(rows_out)
        out_csv = self.args.out
        df.to_csv(out_csv, index=False)
        print(f"\nSaved results to {out_csv}")

        # Optional Ragas
        if self.ragas.enabled and results:
            print("Running Ragas metrics (this may take a while)...")
            ragas_scores = self.ragas.score_batch(results)
            # Merge and save
            extra = []
            for r in results:
                key = f"{r.qid}::{r.model}::{r.modality}"
                metrics = ragas_scores.get(key, {})
                extra.append({"id": key, **metrics})
            import pandas as pd
            df2 = pd.DataFrame(extra)
            out_ragas = os.path.splitext(out_csv)[0] + "_ragas.csv"
            df2.to_csv(out_ragas, index=False)
            print(f"Saved Ragas metrics to {out_ragas}")


# ---------------------- CLI ----------------------


def build_argparser():
    p = argparse.ArgumentParser(
        description="Evaluate text & multimodal RAG across models from a JSONL with per-row context."
    )
    p.add_argument('--models', nargs='+', required=True, help='Hugging Face model ids (text and/or vision).')
    p.add_argument('--qa', required=True, help='Path to questions JSONL file.')

    # Legacy corpora options (optional)
    p.add_argument('--text_corpus', default=None, help='Directory of text/pdf files for legacy text RAG (optional).')
    p.add_argument('--image_corpus', default=None, help='Directory of images for legacy multimodal RAG (optional).')

    p.add_argument('--k', type=int, default=5, help='Top-k context items to retrieve (per-row text).')
    p.add_argument('--max_new_tokens', type=int, default=256)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--top_p', type=float, default=0.95)

    p.add_argument('--text_chunk_tokens', type=int, default=300, help='Max tokens per chunk.')
    p.add_argument('--text_chunk_overlap', type=int, default=80, help='Token overlap between adjacent chunks.')
    p.add_argument('--text_embedder', default='intfloat/multilingual-e5-base')
    p.add_argument('--clip_backbone', default='ViT-H-14')
    p.add_argument('--clip_weights', default='laion2b_s32b_b79k')

    p.add_argument('--no_caption', action='store_true', help='Disable BLIP captioning during legacy image ingestion.')
    p.add_argument('--no_ocr', action='store_true', help='Disable EasyOCR during legacy image ingestion.')
    p.add_argument('--ocr_langs', default='it,en', help='Comma-separated OCR languages (e.g., it,en,fr).')

    p.add_argument('--use_ragas', action='store_true', help='Also compute RAGAS metrics if available.')
    p.add_argument('--judge_model', default=None, help='Optional: judge model for RAGAS (configure per ragas docs).')

    # ----- NEW FLAGS -----
    p.add_argument('--fail_on_missing_text', action='store_true',
                   help="Error out if any row lacks a non-empty 'text' when no --text_corpus is provided.")
    p.add_argument('--fail_on_missing_question', action='store_true',
                   help='Error out if any row lacks a non-empty \"question\".')
    p.add_argument('--debug', action='store_true',
                   help='Print selected contexts and questions for quick inspection.')

    p.add_argument('--vision_models', nargs='*', default=[], help='Explicit list of model IDs to treat as vision models.')

    p.add_argument('--out', default='rag_eval_results.csv')
    return p


def main():
    args = build_argparser().parse_args()
    # Parse OCR langs into list
    if isinstance(args.ocr_langs, str):
        args.ocr_langs = [x.strip() for x in args.ocr_langs.split(',') if x.strip()]
    runner = Runner(args)
    runner.run()


if __name__ == '__main__':
    main()
