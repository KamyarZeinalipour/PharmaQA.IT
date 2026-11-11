#!/usr/bin/env python3
"""
RAG CSV Evaluator + LLM-as-a-Judge
==================================

Usage:
  python eval_rag_csv.py --csv rag_eval_results.csv
  # optional:
  python eval_rag_csv.py --csv rag_eval_results.csv --group_by model modality --out_prefix scores/my_run
  # with LLM-as-a-judge:
  python eval_rag_csv.py --csv rag_eval_results.csv --llm_judge

What it does:
- Loads the CSV produced by your RAG runner (expects columns: 'model_answer' and 'answer').
- Computes many metrics per row (safe even if libs are missing):
  * em_exact (strict exact match after .strip() on both sides)
  * em_norm (exact match after normalization: lowercase + punctuation/space collapse)
  * token_precision / token_recall / token_f1
  * jaccard_token
  * edit_distance, edit_sim_ratio (difflib or python-Levenshtein if available)
  * rouge1/rouge2/rougeL (requires 'rouge-score')
  * bleu, chrf in [0,1] (requires 'sacrebleu'; uses sentence_bleu / sentence_chrf)
  * meteor (requires 'nltk')
  * bertscore_P/R/F1 (requires 'bert-score')
  * sbert_cosine (requires 'sentence-transformers')
  * coverage vs. contexts (if 'chunks_used' exists in CSV)
  * retrieval_hit: 1.0 if normalized gold answer appears in the normalized concatenated contexts (chunks_used),
                  NaN if no contexts or no gold.
  * OPTIONAL: LLM-as-a-judge metrics using DeepSeek (if --llm_judge and DEEPSEEK_API_KEY set):
      - llm_judge_score           (0.0–1.0)
      - llm_judge_label           ("CORRECT", "PARTIALLY_CORRECT", "INCORRECT")
      - llm_judge_label_score     (1.0, 0.5, 0.0)
      - llm_judge_explanation     (short text)
      - llm_judge_raw             (full raw model output)
      - llm_judge_user_prompt     (exact prompt sent as user message)

- Supports multiple gold references:
  - If 'answer' cell looks like JSON list (e.g. ["x","y"]) or uses '||', it picks the best reference per metric.
- Writes:
  - <out_prefix>_scored.csv  (row-level metrics appended)
  - <out_prefix>_summary.csv (aggregated by group(s), default: model, modality if present)
  - <out_prefix>_summary.json (same summary as JSON)

CLI flags:
  --csv               Path to input CSV
  --gold_column       Defaults to 'answer'
  --pred_column       Defaults to 'model_answer'
  --group_by          Column names to group summaries (default: model modality if present)
  --language          Language hint for some metrics (default: 'en'; for Italian use --language it)
  --bert_score_model  HF model for bert-score (default: None -> lang mode)
  --sbert_model       SBERT model for cosine similarity (default: 'sentence-transformers/all-MiniLM-L6-v2')
  --llm_judge         If set, use DeepSeek as an LLM judge (requires DEEPSEEK_API_KEY, openai>=1.x)
  --out_prefix        Output prefix (default: input name without extension)
"""

import argparse
import ast
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

# ---------- Optional deps ----------
HAS_SACREBLEU = False
HAS_ROUGE = False
HAS_NLTK_METEOR = False
HAS_BERTSCORE = False
HAS_SENTENCE_TRANSFORMERS = False
HAS_LEVENSHTEIN = False
HAS_OPENAI = False

try:
    import sacrebleu  # type: ignore
    HAS_SACREBLEU = True
except Exception:
    pass

try:
    from rouge_score import rouge_scorer  # type: ignore
    HAS_ROUGE = True
except Exception:
    pass

try:
    from nltk.translate.meteor_score import meteor_score  # type: ignore
    HAS_NLTK_METEOR = True
except Exception:
    pass

try:
    from bert_score import score as bert_score  # type: ignore
    HAS_BERTSCORE = True
except Exception:
    pass

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    import numpy as np
    HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    np = None  # type: ignore

try:
    import Levenshtein  # type: ignore
    HAS_LEVENSHTEIN = True
except Exception:
    from difflib import SequenceMatcher  # type: ignore
    Levenshtein = None  # type: ignore

try:
    from openai import OpenAI  # type: ignore
    HAS_OPENAI = True
except Exception:
    OpenAI = None  # type: ignore


# ---------- Text normalization & helpers ----------

_punct_re = re.compile(r"[^\w\s]", flags=re.UNICODE)
_spaces_re = re.compile(r"\s+")


def normalize(s: str) -> str:
    s = s.strip().lower()
    s = _punct_re.sub(" ", s)
    s = _spaces_re.sub(" ", s).strip()
    return s


def tokenize(s: str) -> List[str]:
    return normalize(s).split()


def exact_match_norm(pred: str, gold: str) -> float:
    return 1.0 if normalize(pred) == normalize(gold) else 0.0


def exact_match_strict(pred: str, gold: str) -> float:
    """Strict exact match after trimming leading/trailing whitespace only."""
    return 1.0 if pred.strip() == gold.strip() else 0.0


def prf_tokens(pred: str, gold: str) -> Tuple[float, float, float]:
    p = tokenize(pred)
    g = tokenize(gold)
    if not p and not g:
        return (1.0, 1.0, 1.0)
    if not p or not g:
        return (0.0, 0.0, 0.0)
    from collections import Counter

    pc = Counter(p)
    gc = Counter(g)
    overlap = sum((pc & gc).values())
    prec = overlap / max(1, sum(pc.values()))
    rec = overlap / max(1, sum(gc.values()))
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return (prec, rec, f1)


def jaccard_tokens(pred: str, gold: str) -> float:
    ps = set(tokenize(pred))
    gs = set(tokenize(gold))
    if not ps and not gs:
        return 1.0
    if not ps or not gs:
        return 0.0
    inter = len(ps & gs)
    union = len(ps | gs)
    return inter / union if union else 0.0


def edit_distance_and_ratio(pred: str, gold: str) -> Tuple[int, float]:
    if HAS_LEVENSHTEIN:
        d = Levenshtein.distance(pred, gold)  # type: ignore
        m = max(len(pred), len(gold))
        ratio = 1.0 - (d / m) if m else 1.0
        return d, ratio
    else:
        sm = SequenceMatcher(None, pred, gold)  # type: ignore
        ratio = sm.ratio()
        m = max(len(pred), len(gold))
        d = int(round((1.0 - ratio) * m))
        return d, ratio


def safe_mean(vals: Sequence[float]) -> float:
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def try_parse_json_list(s: str) -> Optional[List[str]]:
    """Detect if gold is multi-reference: JSON list or 'x || y'."""
    s = s.strip()
    if not s:
        return None
    # JSON list?
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple)):
                return [str(x) for x in obj]
        except Exception:
            pass
    # || separated?
    if "||" in s:
        parts = [p.strip() for p in s.split("||") if p.strip()]
        if parts:
            return parts
    return None


def best_over_refs(pred: str, refs: List[str], fn) -> Tuple[Any, str]:
    """Return (best_val, best_ref) over multiple gold references using comparator fn(pred, ref)->scalar."""
    best_v = None
    best_r = ""
    for r in refs:
        v = fn(pred, r)
        if best_v is None or (v is not None and v > best_v):
            best_v, best_r = v, r
    return best_v, best_r


# ---------- Metric runners using optional libs ----------

def compute_rouge(pred: str, gold: str) -> Dict[str, float]:
    if not HAS_ROUGE:
        return {}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    s = scorer.score(gold, pred)
    return {
        "rouge1_f": float(s["rouge1"].fmeasure),
        "rouge2_f": float(s["rouge2"].fmeasure),
        "rougeL_f": float(s["rougeL"].fmeasure),
        "rougeL_p": float(s["rougeL"].precision),
        "rougeL_r": float(s["rougeL"].recall),
    }


def compute_sacrebleu(pred: str, golds: List[str]) -> Dict[str, float]:
    """
    Compute sentence-level BLEU and chrF using sacrebleu, with scores normalized to [0,1].
    Supports multiple references.
    """
    if not HAS_SACREBLEU or not golds:
        return {}
    # sentence_bleu / sentence_chrf return scores in [0, 100]
    bleu_score = sacrebleu.sentence_bleu(pred, golds).score
    chrf_score = sacrebleu.sentence_chrf(pred, golds).score
    return {
        "bleu": float(bleu_score / 100.0),
        "chrf": float(chrf_score / 100.0),
    }


def compute_meteor(pred: str, gold: str) -> Dict[str, float]:
    if not HAS_NLTK_METEOR:
        return {}
    hyp_tokens = tokenize(pred)
    ref_tokens = tokenize(gold)
    try:
        return {"meteor": float(meteor_score([ref_tokens], hyp_tokens))}
    except TypeError as e:
        print(f"[WARN] METEOR failed on a row: {e}")
        return {}


def compute_bertscore(pred: str, golds: List[str], model: Optional[str], lang: str) -> Dict[str, float]:
    """
    Compute BERTScore. For multi-reference, we pick the best reference based on token-F1,
    to avoid shape issues with bert_score API.
    """
    if not HAS_BERTSCORE or not golds:
        return {}

    # pick best reference according to token-F1 (consistent with other metrics)
    best_ref = golds[0]
    best_f1 = float("-inf")
    for r in golds:
        _, _, f1 = prf_tokens(pred, r)
        if f1 > best_f1:
            best_f1 = f1
            best_ref = r

    P, R, F1 = bert_score(
        [pred],
        [best_ref],
        lang=lang if model is None else None,
        model_type=model if model else None,
        verbose=False,
    )
    return {
        "bertscore_P": float(P.mean().item()),
        "bertscore_R": float(R.mean().item()),
        "bertscore_F1": float(F1.mean().item()),
    }


def compute_sbert_cosine(pred: str, gold: str, model: Optional["SentenceTransformer"]) -> Dict[str, float]:
    if not (HAS_SENTENCE_TRANSFORMERS and model is not None and np is not None):
        return {}
    emb = model.encode([pred, gold], convert_to_numpy=True, normalize_embeddings=True)
    cos = float((emb[0] @ emb[1].T))
    return {"sbert_cosine": cos}


# ---------- Coverage vs retrieved contexts (if present) ----------

def coverage_against_context(text: str, contexts: List[str]) -> float:
    """Fraction of text tokens found in the concatenated contexts."""
    if not contexts:
        return float("nan")
    ctx = normalize(" ".join([c for c in contexts if isinstance(c, str)]))
    tokens = tokenize(text)
    if not tokens:
        return 1.0
    hit = sum(1 for t in tokens if t in ctx)
    return hit / len(tokens)


def retrieval_hit_metric(gold: str, contexts: List[str]) -> float:
    """
    Binary retrieval indicator:
    1.0 if the normalized gold answer appears inside the normalized concatenated contexts,
    NaN if there are no contexts or no gold.
    """
    if not contexts:
        return float("nan")
    gold_norm = normalize(gold)
    if not gold_norm:
        return float("nan")
    ctx_norm = normalize(" ".join(str(c) for c in contexts))
    return 1.0 if gold_norm in ctx_norm else 0.0


# ---------- LLM-as-a-judge ----------

def llm_as_judge(question: str, pred: str, gold: str, lang: str = "it") -> Dict[str, Any]:
    """
    Usa un LLM (DeepSeek) come giudice per valutare quanto la risposta `pred` è corretta
    rispetto alla risposta di riferimento `gold`, per la domanda `question`.

    Ritorna un dict con:
      - llm_judge_score           (0.0–1.0)
      - llm_judge_label           ("CORRECT", "PARTIALLY_CORRECT", "INCORRECT")
      - llm_judge_label_score     (1.0, 0.5, 0.0)
      - llm_judge_explanation     (spiegazione breve in italiano)
      - llm_judge_raw             (output grezzo del modello)
      - llm_judge_user_prompt     (prompt mandato all'LLM)
    """
    if not HAS_OPENAI:
        return {}

    # Use DeepSeek as judge: requires DEEPSEEK_API_KEY and base_url
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("[WARN] DEEPSEEK_API_KEY non impostata; salto LLM judge.", file=sys.stderr)
        return {}

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    system_msg = (
        "Sei un valutatore esperto di sistemi di domanda-risposta in LINGUA ITALIANA. "
        "Riceverai: una domanda, una risposta generata dal modello e una risposta di riferimento (gold). "
        "Devi valutare quanto la risposta del modello è CORRETTA, COMPLETA e NON CONTRADDITTORIA "
        "rispetto alla risposta di riferimento. "
        "Non aggiungere conoscenze esterne: considera solo ciò che è scritto nella domanda e nelle risposte."
    )

    user_msg = f"""
DOMANDA (in italiano):
{question}

RISPOSTA DEL MODELLO (da valutare):
{pred}

RISPOSTA DI RIFERIMENTO (gold, in italiano):
{gold}

LINEE GUIDA PER LA VALUTAZIONE (segui attentamente):

1. Lavora SOLO con le informazioni presenti sopra. NON usare conoscenza esterna.
2. Confronta la risposta del modello con la risposta gold e valuta:
   - Correttezza dei fatti.
   - Completezza delle informazioni ESSENZIALI presenti nel gold.
   - Assenza di contraddizioni rispetto al gold.
3. Assegna uno score numerico tra 0.0 e 1.0:
   - 1.0: la risposta è semanticamente equivalente al gold; tutte le informazioni importanti sono presenti;
          nessun errore sostanziale.
   - 0.7–0.9: risposta in gran parte corretta; mancano solo dettagli secondari oppure ci sono piccole imprecisioni.
   - 0.4–0.6: risposta parzialmente corretta; contiene alcune informazioni corrette ma manca una parte
              significativa di contenuto importante oppure ci sono errori rilevanti.
   - 0.1–0.3: risposta quasi tutta sbagliata; solo pochi dettagli sono corretti.
   - 0.0: risposta completamente sbagliata, fuori tema o in forte contraddizione con il gold.
4. Assegna anche un'etichetta categoriale:
   - "CORRECT": la risposta del modello è sostanzialmente equivalente al gold
                (tutte le informazioni chiave sono presenti, al massimo piccoli dettagli mancanti).
   - "PARTIALLY_CORRECT": la risposta contiene parti corrette ma è incompleta oppure mischia errori e contenuto giusto.
   - "INCORRECT": la risposta è per lo più sbagliata, fuori tema o contraddice il gold.
5. Se la risposta del modello introduce informazioni non citate nel gold:
   - Se sono plausibili ma NON verificabili dal gold, considerale sospette e penalizza lo score.
   - Se sono chiaramente in conflitto con il gold, considera la risposta INCORRECT.
6. Scrivi una spiegazione BREVE in ITALIANO che:
   - Indichi quali parti sono corrette.
   - Indichi quali parti mancano o sono sbagliate.

Restituisci SOLO JSON valido, senza nessun testo extra, con questo schema ESATTO:
{{
  "score": <float tra 0.0 e 1.0>,
  "label": "CORRECT" | "PARTIALLY_CORRECT" | "INCORRECT",
  "explanation": "<spiegazione breve in italiano>"
}}
"""

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            # response_format={"type": "json_object"},  # if supported
        )
    except Exception as e:
        print(f"[WARN] LLM judge call failed: {e}", file=sys.stderr)
        return {}

    content = resp.choices[0].message.content or ""

    # Default
    score = float("nan")
    label = "UNKNOWN"
    expl = content.strip()

    # Prova a fare il parse come JSON "pulito"
    try:
        obj = json.loads(content)
        score = float(obj.get("score", 0.0))
        raw_label = str(obj.get("label", "")).upper().strip()
        if raw_label in {"CORRECT", "PARTIALLY_CORRECT", "INCORRECT"}:
            label = raw_label
        expl = str(obj.get("explanation", "")).strip()
    except Exception:
        # Fallback robusto: cerca i campi con regex se il modello ha aggiunto testo extra
        m_score = re.search(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', content)
        if m_score:
            try:
                score = float(m_score.group(1))
            except Exception:
                pass
        else:
            m_generic = re.search(r"([01](?:\.\d+)?)", content)
            if m_generic:
                try:
                    score = float(m_generic.group(1))
                except Exception:
                    pass

        m_label = re.search(r'"label"\s*:\s*"([^"]+)"', content)
        if m_label:
            raw_label = m_label.group(1).upper().strip()
            if raw_label in {"CORRECT", "PARTIALLY_CORRECT", "INCORRECT"}:
                label = raw_label

        m_expl = re.search(r'"explanation"\s*:\s*"([^"]*)"', content, flags=re.S)
        if m_expl:
            expl = m_expl.group(1).strip()

    label_score_map = {
        "CORRECT": 1.0,
        "PARTIALLY_CORRECT": 0.5,
        "INCORRECT": 0.0,
    }
    label_score = label_score_map.get(label, float("nan"))

    return {
        "llm_judge_score": float(score),
        "llm_judge_label": label,
        "llm_judge_label_score": float(label_score),
        "llm_judge_explanation": expl,
        "llm_judge_raw": content.strip(),
        "llm_judge_user_prompt": user_msg.strip(),
    }


# ---------- Row metric computation ----------

def compute_all_metrics_row(
    pred: str,
    gold_cell: str,
    contexts_cell: Optional[str],
    question: str,
    opts,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}

    # Prepare references (support multi-gold)
    golds = try_parse_json_list(gold_cell)
    if golds is None:
        golds = [gold_cell]

    # --- Exact matches (best-over-refs) ---
    em_exact_v, em_exact_ref = best_over_refs(pred, golds, lambda p, g: exact_match_strict(p, g))
    em_norm_v, em_norm_ref = best_over_refs(pred, golds, lambda p, g: exact_match_norm(p, g))

    # --- Simple overlaps (use best-F1 reference for PRF/Jaccard/edit/ROUGE/etc.) ---
    prf_best_v, prf_ref = best_over_refs(pred, golds, lambda p, g: prf_tokens(p, g)[2])  # by F1
    jac_v, _ = best_over_refs(pred, golds, lambda p, g: jaccard_tokens(p, g))

    prec, rec, f1 = prf_tokens(pred, prf_ref)
    metrics.update(
        {
            "em_exact": float(em_exact_v or 0.0),
            "em_norm": float(em_norm_v or 0.0),
            "token_precision": float(prec),
            "token_recall": float(rec),
            "token_f1": float(f1),
            "jaccard_token": float(jac_v or 0.0),
        }
    )

    # Edit distance/similarity on best-F1 reference
    ed, ed_ratio = edit_distance_and_ratio(pred, prf_ref)
    metrics.update({"edit_distance": int(ed), "edit_sim_ratio": float(ed_ratio)})

    # ROUGE (single reference: best-F1 reference)
    metrics.update(compute_rouge(pred, prf_ref))

    # sacrebleu / chrF (multi-ref, normalized to [0,1])
    metrics.update(compute_sacrebleu(pred, golds))

    # METEOR (single reference: best-F1 reference)
    metrics.update(compute_meteor(pred, prf_ref))

    # BERTScore (multi-ref, but internally picks best reference)
    if HAS_BERTSCORE:
        try:
            metrics.update(compute_bertscore(pred, golds, opts.bert_score_model, opts.language))
        except Exception as e:
            print(f"[WARN] bert-score failed on a row: {e}", file=sys.stderr)

    # SBERT cosine (single reference: best-F1 reference)
    if HAS_SENTENCE_TRANSFORMERS:
        try:
            metrics.update(compute_sbert_cosine(pred, prf_ref, opts._sbert))
        except Exception as e:
            print(f"[WARN] sbert failed on a row: {e}", file=sys.stderr)

    # Length & novelty/overlap
    p_toks = tokenize(pred)
    g_toks = tokenize(prf_ref)
    novelty = len([t for t in p_toks if t not in g_toks]) / (len(p_toks) or 1)
    overlap = len([t for t in p_toks if t in g_toks]) / (len(p_toks) or 1)
    metrics.update(
        {
            "pred_len_tok": int(len(p_toks)),
            "gold_len_tok": int(len(g_toks)),
            "novelty_rate": float(novelty),
            "overlap_rate": float(overlap),
        }
    )

    # Context coverage if contexts available (expects 'chunks_used' JSON in cell)
    contexts: List[str] = []
    if contexts_cell and isinstance(contexts_cell, str) and contexts_cell.strip():
        try:
            ctx = json.loads(contexts_cell)
            if isinstance(ctx, list):
                contexts = [str(x) for x in ctx]
        except Exception:
            pass
    metrics.update(
        {
            "pred_coverage_in_context": float(coverage_against_context(pred, contexts))
            if contexts
            else float("nan"),
            "gold_coverage_in_context": float(coverage_against_context(prf_ref, contexts))
            if contexts
            else float("nan"),
        }
    )

    # Binary retrieval indicator: does the (best-F1) gold reference appear in contexts?
    metrics["retrieval_hit"] = (
        float(retrieval_hit_metric(prf_ref, contexts)) if contexts else float("nan")
    )

    # Remember which references were best
    metrics["matched_gold_ref_f1"] = prf_ref
    metrics["matched_gold_ref_em_exact"] = em_exact_ref
    metrics["matched_gold_ref_em_norm"] = em_norm_ref

    # --- LLM-as-a-judge metric (optional, DeepSeek) ---
    if getattr(opts, "llm_judge", False):
        try:
            judge_metrics = llm_as_judge(question, pred, prf_ref, lang=opts.language)
            metrics.update(judge_metrics)
        except Exception as e:
            print(f"[WARN] LLM judge failed on a row: {e}", file=sys.stderr)

    return metrics


# ---------- Aggregation ----------

def agg_df(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    metric_cols = [c for c in df.columns if c not in set(group_cols + ["_row_idx"])]
    summary = df.groupby(group_cols, dropna=False)[metric_cols].mean(numeric_only=True).reset_index()
    summary["count"] = df.groupby(group_cols, dropna=False).size().values
    return summary


# ---------- Main ----------

def build_argparser():
    ap = argparse.ArgumentParser(description="Evaluate QA metrics on a RAG CSV.")
    ap.add_argument("--csv", required=True, help="Path to the CSV produced by the RAG runner.")
    ap.add_argument(
        "--gold_column",
        default="answer",
        help="Column name for gold/reference answers.",
    )
    ap.add_argument(
        "--pred_column",
        default="model_answer",
        help="Column name for model predictions.",
    )
    ap.add_argument(
        "--group_by",
        nargs="*",
        default=None,
        help="Columns to group for summary (default: model modality if present).",
    )
    ap.add_argument(
        "--language",
        default="en",
        help="Language hint for some metrics (bertscore when model not set). For Italian, use 'it'.",
    )
    ap.add_argument(
        "--bert_score_model",
        default=None,
        help="HF model for bert-score (e.g., roberta-large). If omitted and bert-score is installed, lang-based defaults are used.",
    )
    ap.add_argument(
        "--sbert_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model for sbert_cosine.",
    )
    ap.add_argument(
        "--llm_judge",
        action="store_true",
        help="If set, use DeepSeek as an LLM judge to score answers (requires DEEPSEEK_API_KEY).",
    )
    ap.add_argument(
        "--out_prefix",
        default=None,
        help="Output prefix. Default: <csv_basename_without_ext>.",
    )
    return ap


@dataclass
class Opts:
    language: str
    bert_score_model: Optional[str]
    _sbert: Optional[Any]  # SentenceTransformer or None
    llm_judge: bool = False


def main():
    args = build_argparser().parse_args()

    # Load CSV
    df = pd.read_csv(args.csv)

    if args.pred_column not in df.columns or args.gold_column not in df.columns:
        raise SystemExit(
            f"Columns not found. Need '{args.pred_column}' (pred) and '{args.gold_column}' (gold)."
        )

    # Identify question column if present
    question_col = "question" if "question" in df.columns else None

    # ---- REMOVE DUPLICATES (question + answer/gold) BEFORE EVALUATION ----
    if question_col is not None:
        before = len(df)
        df = df.drop_duplicates(
            subset=[question_col, args.gold_column],  # 'question' + gold column (default: 'answer')
            keep="first",
        )
        df = df.reset_index(drop=True)
        removed = before - len(df)
        print(
            f"[INFO] Removed {removed} duplicate rows based on ({question_col}, {args.gold_column}). "
            f"Remaining rows: {len(df)}",
            file=sys.stderr,
        )
    else:
        print("[INFO] No 'question' column found; skipping duplicate removal.", file=sys.stderr)


    # Prepare SBERT model (optional)
    sbert_model = None
    if HAS_SENTENCE_TRANSFORMERS and args.sbert_model:
        try:
            sbert_model = SentenceTransformer(args.sbert_model)
        except Exception as e:
            print(f"[WARN] Failed to load SBERT model '{args.sbert_model}': {e}", file=sys.stderr)

    opts = Opts(
        language=args.language,
        bert_score_model=args.bert_score_model,
        _sbert=sbert_model,
        llm_judge=args.llm_judge,
    )

    # Compute metrics row-wise
    metric_rows: List[Dict[str, Any]] = []
    scored_rows: List[Dict[str, Any]] = []

    # Identify contexts column if present
    contexts_col = "chunks_used" if "chunks_used" in df.columns else None
    # Identify question column if present (adjust name if your CSV uses another)
    question_col = "question" if "question" in df.columns else None

    total = len(df)
    print(f"[INFO] Starting evaluation for {total} examples...", file=sys.stderr)

    for i, (idx, row) in enumerate(df.iterrows(), start=1):
        pred = str(row.get(args.pred_column, "") or "")
        gold_cell = str(row.get(args.gold_column, "") or "")
        question = str(row.get(question_col, "")) if question_col else ""

        row_metrics = compute_all_metrics_row(
            pred,
            gold_cell,
            row.get(contexts_col) if contexts_col else None,
            question,
            opts,
        )
        metric_rows.append({"_row_idx": idx, **row_metrics})

        merged = dict(row)
        merged.update(row_metrics)
        scored_rows.append(merged)

        # ----- Progress printing: how many done / left -----
        left = total - i
        print(f"[PROGRESS] Done {i}/{total} examples, {left} left.", file=sys.stderr)

    mdf = pd.DataFrame(metric_rows)
    scored_df = pd.DataFrame(scored_rows)

    # Choose grouping
    if args.group_by:
        group_cols = [c for c in args.group_by if c in scored_df.columns]
    else:
        default_cols = [c for c in ["model", "modality"] if c in scored_df.columns]
        group_cols = default_cols if default_cols else []

    summary_df = (
        agg_df(
            mdf.merge(
                df.reset_index().rename(columns={"index": "_row_idx"}),
                on="_row_idx",
                how="left",
            ),
            group_cols,
        )
        if group_cols
        else pd.DataFrame()
    )

    # Outputs
    base = args.out_prefix or os.path.splitext(os.path.basename(args.csv))[0]
    out_scored = f"{base}_scored.csv"
    out_summary = f"{base}_summary.csv"
    out_summary_json = f"{base}_summary.json"

    scored_df.to_csv(out_scored, index=False)
    print(f"[OK] Wrote row-level metrics to: {out_scored}", file=sys.stderr)

    if not summary_df.empty:
        summary_df.to_csv(out_summary, index=False)
        print(f"[OK] Wrote grouped summary to: {out_summary}", file=sys.stderr)

        with open(out_summary_json, "w", encoding="utf-8") as f:
            json.dump(summary_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
        print(f"[OK] Wrote grouped summary JSON to: {out_summary_json}", file=sys.stderr)
    else:
        print("[INFO] No grouping columns found; skipped summary files.", file=sys.stderr)

    # Final hints re: optional metrics
    missing = []
    if not HAS_ROUGE:
        missing.append("rouge-score")
    if not HAS_SACREBLEU:
        missing.append("sacrebleu")
    if not HAS_NLTK_METEOR:
        missing.append("nltk (meteor)")
    if not HAS_BERTSCORE:
        missing.append("bert-score")
    if not HAS_SENTENCE_TRANSFORMERS:
        missing.append("sentence-transformers")
    if args.llm_judge and not HAS_OPENAI:
        missing.append("openai")
    if missing:
        print("[INFO] Some metrics were skipped. To enable more, install:", ", ".join(missing), file=sys.stderr)


if __name__ == "__main__":
    main()
