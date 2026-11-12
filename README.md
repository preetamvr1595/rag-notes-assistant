# RAG Notes Assistant — Chat With Your Study Notes (Citations, Out-of-Scope Refusal, Optional Llama 3)

A **Retrieval-Augmented Generation (RAG) chatbot** that answers questions **only from your own notes**, shows the **sources**, and says *"I couldn't find that in your notes"* when the topic isn't covered. Built and **measured** with a hand-written evaluation set — including the parts that don't work well yet.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B) ![RAG](https://img.shields.io/badge/RAG-BM25%20%2B%20char%20n--grams-purple) ![Ollama](https://img.shields.io/badge/LLM-Ollama%20%2F%20OpenAI--compatible-black) ![tests](https://img.shields.io/badge/tests-25%20passing-brightgreen) ![License-MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![v0.2.0](https://img.shields.io/badge/version-0.2.0-blue)

> Project 2 of 5 in my ML-engineer portfolio series (pipeline → **RAG** → LLM fine-tuning → monitoring/drift → multimodal).

## What it does

```
question ──► retrieve top passages (BM25 + character n-grams, fused) ──► refusal gate ──► answer
                                                                       │ not covered ──► "I couldn't find that in your notes."
                                                                       └ covered ──► offline excerpts  OR  LLM (Ollama / OpenAI-compatible) with [n] citations
```

**Corpus (included):** my own NumPy, Pandas, Matplotlib, Seaborn and Scikit-learn notebooks (exported to Markdown), a pandas interview Q&A set, and the public *100 NumPy exercises* list — 27 files, 32k words, 582 searchable passages. **Drop in your own `.md / .txt / .ipynb / .pdf` files** (course notes, past papers) and rebuild.

## Results (honest version)

Evaluation set: **62 hand-written in-scope questions** (paraphrased, so keyword matching gets no free pass) + **24 out-of-scope questions**, split into dev (tuning) and test (looked at after the settings were frozen). Small sample: one question ≈ 3 points; **confidence intervals are about ±15 points**.

| Retrieval on the 31 test questions | hit@3 | hit@5 | MRR |
|---|---|---|---|
| BM25 (keywords) | 0.74 | 0.77 | 0.69 |
| Character n-grams | 0.68 | 0.71 | 0.63 |
| LSA (embedding trained on the notes) | 0.71 | 0.84 | 0.72 |
| **Default: BM25 + char n-grams (RRF fusion)** | **0.71** | **0.74** | **0.68** |

- **No method is provably better** with this sample size; on the dev split plain BM25 had the best MRR. The fused retriever was frozen before looking at test, and the notebooks say so.
- Real problem found and fixed: tiny *exercise* chunks crowded out real answers (fixed by down-weighting; they are now ~1% of top-3 results).
- Remaining misses are mostly **vocabulary mismatch** ("rescale to 0–1" vs `MinMaxScaler`). Neural embeddings are the usual fix — `scripts/eval_minilm.py` measures them on your machine; **I did not run it**.

| Refusal gate (default threshold 0.65) | dev | test |
|---|---|---|
| In-scope questions answered | 74% | 90% |
| Out-of-scope questions refused | 92% | 92% (11 of 12) |

End to end (answered **and** correct passage in the top 3): 45% dev, 68% test. A crude word-overlap gate is a known weak spot: it refused 3 good test questions and answered 1 off-topic one.

## What is and is not verified

| Part | Status |
|---|---|
| Chunking, retrieval, evaluation, refusal gate | ✅ run end to end, 3 notebooks with 0 errors |
| 18 unit/integration tests, Streamlit app (headless smoke test) | ✅ passing |
| LLM backend (Ollama / OpenAI-compatible) | ⚠️ tested only against a **local fake server**; **no real Llama 3 / Mistral was run** (no model access where I built it) |
| MiniLM embeddings | ⚠️ code + script provided, **not run** |
| Docker / cloud deploy | ❌ not included |

## Project layout

| Path | Purpose |
|---|---|
| `01_corpus_and_chunking.ipynb` (55 cells) | corpus stats, heading-aware chunking, chunk-size trade-offs |
| `02_retrieval_experiments.ipynb` (64 cells) | eval set, ablations on dev, frozen test results, bootstrap CIs, failure analysis |
| `03_answers_and_refusal.ipynb` (44 cells) | grounding prompt, citations, refusal-gate calibration, LLM plumbing |
| `rag_notes/` | the library: `loader`, `chunker`, `tokenize`, `retrievers`, `index`, `answer`, `evaluate`, `cli` |
| `app/streamlit_app.py` | chat UI (series filter, strictness slider, LLM or offline mode) |
| `eval/golden.json` | the question set (+ `make_golden.py`) |
| `tests/` | 18 tests, run in CI |

## Quick start

```bash
git clone https://github.com/preetamvr1595/rag-notes-assistant.git
cd rag-notes-assistant
pip install -r requirements.txt
python -m rag_notes.cli build                 # index data/notes  (about 1 second)
python -m rag_notes.cli ask "How do I join two tables on a common column?"
streamlit run app/streamlit_app.py            # chat UI
```

### Use a real LLM (Llama 3 via Ollama)
```bash
ollama pull llama3
# Windows PowerShell:  $env:RAG_BACKEND="ollama"      (Linux/macOS: export RAG_BACKEND=ollama)
python -m rag_notes.cli ask "What is the difference between ravel and flatten?"
```
Any OpenAI-compatible server works: set `RAG_BACKEND=openai`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `RAG_MODEL`. If the server is down, the app falls back to note excerpts instead of crashing.

### Use your own notes
Put files under `data/notes/<subject>/` (`.md`, `.txt`, `.ipynb`, `.pdf`), then run `python -m rag_notes.cli build`. A folder named `interview-qa` or `exercises` gets special treatment (see `rag_notes/loader.py`).

## Design decisions worth knowing
- **Heading-aware, code-fence-aware chunks** — fixed windows would cut code blocks in half 36% of the time on this corpus.
- **Contextual header** (`series | file > section`) prepended to indexed text.
- **Honest evaluation:** dev/test split, answer-key validation, bootstrap intervals, failure analysis.
- **Refusal first:** an assistant that invents answers is worse than none for studying.

## Limitations
Small corpus and eval set; keyword-based retrieval; crude refusal gate; extractive mode shows excerpts instead of fluent prose; if a note is wrong, the bot repeats it (with a citation).

## FAQ
**What is RAG?** Retrieve relevant passages from your documents first, then answer from them — instead of trusting a model's memory.
**Does it need an API key or internet?** No — default mode is fully offline. An LLM is optional.
**Can I use PDFs of past exam papers?** Yes — put them in `data/notes/` and rebuild (text-based PDFs; scanned PDFs need OCR first).

## Series
[student-dropout-prediction](https://github.com/preetamvr1595/student-dropout-prediction) · [imdb-sentiment-analysis](https://github.com/preetamvr1595/imdb-sentiment-analysis) · [numpy-mastery](https://github.com/preetamvr1595/numpy-mastery) · [pandas](https://github.com/preetamvr1595/pandas) · [matplotlib](https://github.com/preetamvr1595/matplotlib) · [seaborn](https://github.com/preetamvr1595/seaborn)

If this helped you, please ⭐ star the repo.

*Keywords: RAG, retrieval augmented generation, chatbot, BM25, Streamlit, Ollama, Llama 3, question answering, study assistant, NLP, information retrieval*
