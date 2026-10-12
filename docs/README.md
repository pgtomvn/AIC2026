# Documentation

| Folder | Contents |
|---|---|
| [guides/](guides/) | How to install, operate, and run the system |
| [design/](design/) | Design notes for retrieval features |
| [batches/](batches/) | Per-batch ingestion contracts and release notes (B01, B02) |
| [audits/](audits/) | Audits of external/organizer material |

## Guides

- [Operations guide](guides/operations.md): full architecture, data operations, model
  preparation, HTTP API, verification gates, and the generated-state policy.
- [Hướng dẫn cài đặt cho thành viên (Tiếng Việt)](guides/team_setup_vi.md): step-by-step
  Windows setup and daily startup.
- [Cloud feature extraction](guides/cloud_feature_extraction.md): ZIP-streamed SigLIP2/OCR/ASR
  extraction on Google Colab or Kaggle.

## Design notes

- [Interactive retrieval](design/interactive_retrieval.md)
- [Fine-grained Textual KIS](design/kis_finegrained_retrieval.md)
- [KIS accuracy evaluation and activation](design/kis_accuracy_evaluation.md)
- [Metadata-only retrieval batching](design/retrieval_metadata_batching.md)
- [Cross-batch search (B01 + B02)](design/all_batches_search.md)

## Batch notes

- [B01 query hardening](batches/b01_query_hardening.md)
- [B02 P0: source and keyframe-time contract](batches/b02_p0_contract.md)
- [B02 P1: organizer-reference ingestion](batches/b02_p1_reference.md)
- [B02 TRAKE release](batches/b02_trake.md)
- [B02 OCR (VietOCR) staging](batches/b02_ocr_vietocr.md)
- [B02 ASR CPU pilot](batches/b02_asr_ms_cpu.md)

## Audits

- [Organizer Colab baseline audit](audits/organizer_colab_audit.md)

Batch notes are dated engineering records. When a note and the code disagree, the code and
`config/settings.yaml` are authoritative.
