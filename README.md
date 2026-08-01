# AIC 2026 — Multimodal Video Search & Retrieval Engine

> **City AI Challenge (AIC) 2026 — Ho Chi Minh City**  
> An end-to-end multimodal retrieval system for large-scale video databases supporting **Textual KIS**, **Visual Q&A**, and **TRAKE (Temporal Event Alignment)** queries.

---

## 📌 1. Competition Summary & Challenge Overview

The competition evaluates a system's ability to locate precise moments, extract information, and align chronological events within a massive repository of video data.

### Query Types
1. **Textual Known Item Search (Textual KIS):** Find exact moments based on a natural language text description.
   * *Output:* `<video_id>, <frame_id>`
2. **Visual Question Answering (Q&A):** Locate a relevant video moment and extract a specific textual/numerical answer[cite: 1].
   * *Output:* `<video_id>, <frame_id>, <answer>`[cite: 1]
3. **Temporal Alignment of Key Events (TRAKE):** Identify a single video and align a sequence of $N$ sequential semantic keyframes corresponding to event stages[cite: 1].
   * *Output:* `<video_id>, <frame_id_1>, <frame_id_2>, ..., <frame_id_N>`[cite: 1]

### Evaluation Metric: Final Score
Submissions allow up to 100 answers per query[cite: 1]. Scores are measured using **R-Score** (0.0 to 1.0) per submission[cite: 1]. The **Final Score** prioritizes correct predictions at top ranks[cite: 1]:

$$\text{Final Score} = \frac{R@1 + R@5 + R@20 + R@50 + R@100}{5}$$

*where $R@k = \max_{1 \le i \le k} \{R\text{-Score}(r_i)\}$[cite: 1]. Placing correct answers at **Rank 1** is crucial to maximize points[cite: 1].*

---

## 🛠️ 2. Proposed Tech Stack & Architecture

                   ┌──────────────────────────────┐
                   │        INPUT QUERIES         │
                   └──────────────┬───────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
[ Textual KIS ]               [ Visual Q&A ]              [ TRAKE ]
│                         │                         │
▼                         ▼                         ▼
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│  Vector Search +     │  │ Vector Retrieval +   │  │ Temporal DP / DTW    │
│  BM25 Keyword Search │  │ VLM Inference Engine │  │ Multi-Stage Sequence │
└───────────┬──────────┘  └───────────┬──────────┘  └───────────┬──────────┘
│                         │                         │
└─────────────────────────┼─────────────────────────┘
▼
┌───────────────────────────┐
│   RERANKING & TOP-K RANK   │
│   (Streamlit / Gradio)    │
└───────────────────────────┘


| Component | Technology / Model | Purpose |
| :--- | :--- | :--- |
| **Visual Embeddings** | `OpenCLIP-ViT-H/14` / `InternVL` | Cross-modal text-to-image semantic matching |
| **Object Detection** | `YOLOv8` / `YOLOWorld` / `Faster R-CNN` | Filtering keyframes by object labels & bounding boxes[cite: 1] |
| **OCR (Text in Video)** | `PaddleOCR` / `Qwen-VL` | Detecting on-screen text, signage, and overlays |
| **ASR (Audio Transcripts)**| `Whisper-large-v3-turbo` | Speech-to-text conversion for spoken queries |
| **Visual LLM (Q&A)** | `Qwen2-VL-7B` / `LLaVA-NeXT` / `Gemini 1.5 Flash API` | Answering visual reasoning questions |
| **Vector Database** | `Qdrant` / `Milvus` / `FAISS` | Sub-50ms similarity search over millions of vectors |
| **Text Database** | `Meilisearch` / `Elasticsearch` | BM25 keyword matching on OCR & Whisper transcripts[cite: 1] |
| **GUI Interface** | `Streamlit` / `Gradio` | Interactive human-in-the-loop review and export |
