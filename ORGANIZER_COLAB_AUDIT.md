# Organizer Colab audit: AIC 2026 FiftyOne baseline

Audit date: 2026-08-17
Source: [official organizer Colab](https://colab.research.google.com/drive/1kwhlNPBSVL5gVqt2YV5iazGgfHD-dqJz)
Downloaded filename: `AIC-2026-Baseline-v1.ipynb`
SHA-256: `8cbcfae98ff904632875500b1f7282a0e11769393509a2c26c4554cce1fda11d`

## Executive conclusion

The linked notebook is a 33-cell, unexecuted FiftyOne data-exploration tutorial. It is not an
official KIS/Visual Q&A/TRAKE evaluator and does not define a submission format. It confirms the
organizer keyframe, object-detection, and CLIP ViT-B/32 artifact layout, but it contains no sample
query, top-k setting, scoring formula, ground truth, CSV contract, or frame-map semantics.

Historical adapters in this repository therefore remain explicitly historical/provisional. They
must not be renamed or represented as AIC 2026 rules until a separate official rule pack is
published.

## Evidence classification

| Dimension | Status | Evidence |
|---|---|---|
| Notebook identity | Confirmed | 33 cells: 16 code, 17 Markdown, zero executed cells and no outputs |
| Image layout | Confirmed | Cell 5 loads `KeyFrames_L01/keyframes` recursively |
| Video/frame fields | Confirmed | Cell 12 derives `video` and `frameid` from the Windows path and JPEG stem |
| Object schema | Confirmed | Cell 16 reads entities, boxes, and scores; converts TF boxes to FiftyOne boxes |
| Object threshold | Confirmed | Cell 16 keeps detections only when `score > 0.4` |
| Feature layout | Confirmed | Cell 25 loads one `clip-features/<video>.npy` per video |
| Feature row mapping | Confirmed | Cells 21-25 lexicographically sort JPEG stems and zip rows by position |
| Visual model label | Confirmed | Cell 19 says CLIP ViT-B/32; cell 29 uses `clip-vit-base32-torch` |
| Vector dimension/dtype | Verified from supplied pack | 512-D `float16`; the notebook itself does not assert either |
| Vector normalization | Verified from supplied pack | sampled norms are approximately 1.0; notebook performs no explicit validation |
| Similarity backend | Conditional | no backend is passed; current FiftyOne defaults to sklearn, but config can override it |
| Distance metric | Conditional | current FiftyOne sklearn examples use cosine; notebook does not pass a metric |
| KIS query contract | Absent | no query text, preprocessing, search call, depth, fusion, or reranking |
| Visual Q&A | Absent | no event/question/answer schema or normalization |
| TRAKE | Absent | no stages, chronology, or alignment logic |
| Evaluation/submission | Absent | no CSV, filename, interval, R@k, R-Score, MRR, or evaluator |

Current FiftyOne documentation describes sklearn as the default similarity backend, while also
documenting that the default is configurable. That behavior is not a notebook invariant because
the notebook runs unpinned `pip install fiftyone`: [FiftyOne Brain similarity
documentation](https://docs.voxel51.com/brain/index.html).

## Cell inventory

| Cells | Behavior |
|---|---|
| 0-1 | Vietnamese introduction; unpinned installs of FiftyOne and PyTorch packages |
| 3-5 | Imports and recursive image-dataset creation from a Windows path |
| 7-10 | Launches the FiftyOne UI on `localhost:5151` |
| 11-14 | Stores `video` and JPEG-stem `frameid` fields on every sample |
| 15-16 | Loads per-frame object JSON and saves detections above 0.4 confidence |
| 17-25 | Discovers keyframes, sorts stems, and maps per-video `.npy` rows to stems |
| 26-29 | Builds an embedding list and registers FiftyOne similarity index `img_sim` |
| 30-31 | Suggests manual search/filter; UMAP visualization is commented out |

## Exact artifact schemas

### Keyframes

```text
KeyFrames_L01/keyframes/<video>/<frameid>.jpg
```

`frameid` is a string filename stem. The notebook never loads the organizer frame-map CSV and
does not prove that the stem is the source-video frame ordinal. Submission code must continue to
use canonical `source_frame_idx`, not this JPEG stem.

### Object detections

```json
{
  "detection_class_entities": ["..."],
  "detection_boxes": [[0.0, 0.0, 1.0, 1.0]],
  "detection_scores": [0.9]
}
```

The actual pack also contains aligned `detection_class_names` and
`detection_class_labels`. Input box order is `[ymin, xmin, ymax, xmax]`; the tutorial converts it
to `[x, y, width, height]`. All arrays are positional and only strict scores above 0.4 are used.

### CLIP features

```text
clip-features/<video>.npy
```

Rows are assumed to match the lexicographically sorted JPEG stems. There is no check for row
count, dimension, dtype, finite values, normalization, or duplicate stems. The current reference
pack independently verifies:

- 873 map files and 177,321 map rows;
- 873 feature files and 177,321 feature rows;
- 177,321 object JSON files;
- no per-video map/feature row mismatch;
- 512-D `float16` features with approximately unit-length rows.

## Gap analysis

| Evaluation dimension | Organizer Colab | Current AIC2026 repository | Optimization decision |
|---|---|---|---|
| Visual embedder | Supplied CLIP ViT-B/32 features registered in FiftyOne | Verified organizer OpenCLIP 512-D channel plus native SigLIP2 768-D and optional ViT-L/14 | Preserve isolated spaces; add optional recomputed B/32 native channel only behind an explicit gate |
| Feature integrity | Positional mapping without validation | Exact map/JPEG/vector/object count and fingerprint checks | Add an executable baseline manifest/parity report |
| Vector search | FiftyOne default backend/config, not pinned | Versioned Qdrant collections with dimension/count/readiness checks | Compare exact Qdrant results against streamed NumPy cosine before measuring ANN latency |
| Query routing | Manual UI search; no preprocessing shown | Offline VI-to-EN translation, encoder-aware routing, OCR/ASR/metadata and RRF | Keep defaults; tune only with labeled queries |
| Temporal ranking | None | Shot-aware aggregation, diversity, raw-frame refinement | Retain; no accuracy claim without ground truth |
| Visual Q&A | Absent | Decoupled KIS retrieval and single-frame Gemini/Qwen reasoning | No Colab-driven provider change |
| TRAKE | Absent | Strict monotonic DP and persistent mmap trajectories | Wire configured global weight and harden canonical timestamp validation |
| Submission | Absent | Strict historical/provisional KIS, QA, and TRAKE exporters | Add one explicit-spec validator; lock unavailable `official_2026` |
| Ingestion | Reads extracted files and saves per sample | Resumable archive/loose-video ingestion with bounded GPU inference | Wire declared queue/batch controls and add process/scratch safety |

## Baseline weaknesses

1. Dependencies and FiftyOne backend behavior are unpinned.
2. Windows-only path parsing is embedded in notebook cells.
3. Lexicographic sorting is safe only when stems use a consistent fixed width.
4. Feature rows are trusted without alignment, dimension, norm, or finite checks.
5. Object JSON files and FiftyOne samples are opened/saved one at a time.
6. All embeddings are materialized through nested dictionaries and a second list.
7. No source-frame map is loaded, so JPEG ordinals cannot safely be submitted.
8. No benchmark or evaluation data is supplied.

## Prioritized roadmap

### P0: evidence and correctness

- Publish the machine-readable notebook manifest and exact baseline parity harness.
- Correct the configured reference count to 177,321.
- Add a strict, explicit-spec submission validator that refuses unavailable official rules.
- Require canonical timestamps and membership for TRAKE export.
- Add a cross-process ingestion lock and wire existing batch-size configuration.

### P1: measured performance

- Compare exact NumPy and exact Qdrant organizer rankings before ANN measurements.
- Add bounded CPU-to-GPU queues and scratch byte leases for archive members.
- Expose a separately fingerprinted native OpenCLIP B/32 channel, disabled by default.
- Tune RRF only after labeled queries and intervals are provided.

### P2: validation-gated experiments

- Activate the multilingual Meilisearch v2 shadow indexes only after labeled gains.
- Evaluate optional second-encoder fusion and TRAKE raw refinement.
- Add an immutable AIC 2026 rule pack when the organizer publishes one.

## Claims this audit does not make

- The Colab does not define official AIC 2026 scoring or CSV submission rules.
- It contains no sample queries to benchmark accuracy.
- Operational parity and lower latency do not prove higher retrieval accuracy.
- Existing historical 2025 metrics are not substitutes for a future official evaluator.
