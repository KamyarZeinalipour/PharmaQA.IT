# PharmaQA.IT

**An Italian question-answering benchmark over pharmaceutical regulatory documents**

[![Paper](https://img.shields.io/badge/EACL_2026-Industry_Track-b31b1b)](https://aclanthology.org/2026.eacl-industry.70/)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97_Dataset-PharmaQA.IT-yellow)](https://huggingface.co/datasets/Kamyar-zeinalipour/PharmaQA.IT)
[![Collection](https://img.shields.io/badge/%F0%9F%A4%97_Collection-PharmaQA.IT-orange)]({{COLLECTION_URL}})
[![License](https://img.shields.io/badge/code-Apache_2.0-blue)](LICENSE)

In Italy the *Riassunti delle Caratteristiche del Prodotto* (RCP) issued by the Italian Medicines
Agency (AIFA) are the main regulatory source on medicines, but no question-answering dataset
existed for them. **PharmaQA.IT** is the first: 861 extractive question–answer pairs grounded in
497 RCP documents, each answer taken from the document text.

This repository accompanies [*PharmaQA.IT: an Italian dataset for Q&A in the pharmaceutical
domain*](https://aclanthology.org/2026.eacl-industry.70/) (EACL 2026, Industry Track) and contains the dataset, the retrieval-augmented
evaluation pipeline and the scoring scripts.

## The data

```python
import pandas as pd

qa = pd.read_json("data/pharmaqa_it.jsonl.gz", lines=True)
row = qa.iloc[0]
print(row.question)
print(row.answer)
```

Also on the Hugging Face Hub as [`Kamyar-zeinalipour/PharmaQA.IT`](https://huggingface.co/datasets/Kamyar-zeinalipour/PharmaQA.IT).

| Field | Description |
|---|---|
| `id` | Row id |
| `question` | The question, in Italian (median 14 words) |
| `answer` | The answer, extracted from the document (median 4 words) |
| `document_text` | Full text of the source RCP document |
| `page` | Page of the document the answer comes from |
| `source_url` | AIFA record for the medicine |

Questions were proposed by a multimodal LLM prompted on page images with professional personas,
then validated by experts.

## Running the benchmark

```bash
pip install -r requirements.txt
cp .env.example .env     # add the API keys you need

# retrieval-augmented QA over the documents
python rag/rag_pipeline.py --help

# score the predictions (ROUGE, exact match, token F1, LLM judge)
python evaluation/evaluate.py --csv results.csv --out_prefix scores/run
```

The pipeline chunks each document, retrieves the top passages with multilingual E5 embeddings and
asks the model to answer from them. Chunk size and overlap are configurable; the paper reports
64–16, 128–32, 256–64 and 512–128.

## Results

**Open-weight models**, ROUGE-L F1 by chunk configuration:

| Model | 64–16 F1 | 128–32 F1 | 256–64 F1 | 512–128 F1 |
|---|---:|---:|---:|---:|
| Apertus-8B | 0.479 | 0.488 | 0.526 | 0.546 |
| Qwen3-30B | 0.464 | 0.522 | 0.577 | 0.609 |
| Mistral-Nemo | 0.456 | 0.525 | 0.597 | 0.610 |
| Mistral-Small-24B | 0.417 | 0.492 | 0.581 | **0.656** |
| SmolLM3-3B | 0.381 | 0.406 | 0.472 | 0.449 |
| Gemma-3-12B | 0.369 | 0.412 | 0.441 | 0.480 |
| EuroLLM-9B | 0.366 | 0.338 | 0.235 | 0.020 |
| DeepSeek-7B | 0.298 | 0.222 | 0.173 | 0.005 |
| Minerva-7B | 0.220 | 0.203 | 0.177 | 0.033 |
| DeepSeek-V2-Lite | 0.154 | 0.137 | 0.169 | 0.174 |

**Closed models** (512–128):

| Model | F1 (ROUGE-L) | EM |
|---|---:|---:|
| GPT-4.1 | **0.636** | **0.367** |
| Mistral-Large | 0.477 | 0.237 |
| Gemini-2.5-Flash | 0.329 | 0.005 |

**Retrieval quality**:

| Chunk configuration | Retrieval hit@10 |
|---|---:|
| 64–16 | 0.436 |
| 128–32 | 0.511 |
| 256–64 | 0.580 |
| 512–128 | 0.615 |

## Citation

```bibtex
@inproceedings{zeinalipour-etal-2026-pharmaqa,
    title = "{P}harma{QA}.{IT}: an {I}talian dataset for {Q}{\&}{A} in the pharmaceutical domain",
    author = "Zeinalipour, Kamyar  and
      Zugarini, Andrea  and
      Zanollo, Asya  and
      Rigutini, Leonardo",
    editor = {Matusevych, Yevgen  and
      Eryi{\u{g}}it, G{\"u}l{\c{s}}en  and
      Aletras, Nikolaos},
    booktitle = "Proceedings of the 19th Conference of the {E}uropean Chapter of the {A}ssociation for {C}omputational {L}inguistics (Volume 5: Industry Track)",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.eacl-industry.70/",
    doi = "10.18653/v1/2026.eacl-industry.70",
    pages = "937--947",
    ISBN = "979-8-89176-384-5"
}
```

## License

Code: [Apache License 2.0](LICENSE). The RCP documents are public regulatory texts published by
AIFA; the dataset is released for research use.
