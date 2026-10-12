# B02 P0: source and keyframe-time contract

This document records the read-only Batch 2 pre-ingestion gate. It does not
authorize extraction, indexing, ASR, OCR, or replacement of Batch 1 data.

## Source observations (2026-09-24)

- Local reference archives in the repository root: 21 `Keyframes_M*`,
  `Keyframes_N*`, and `Keyframes_S01` ZIPs; `aic26-b2-map-keyframes.zip`;
  `aic26-b2-clip-features.zip`; `aic26-b2-media-info.zip`; and three `Objects_*`
  ZIPs. Do not expand the 84.87 GiB keyframe set onto the local disk.
- The 614 canonical video IDs and 220,270 keyframe ordinals split into
  M (304 videos / 94,677 frames), N (298 / 88,453), and S (12 / 37,140).
  Map, feature, image, and object inventories cover the same videos and frame
  ordinals. Objects are detector metadata, **not OCR text**.
- The media-info archive covers M and S (316 videos); it contains no N media
  records. This is a coverage gap, not evidence that N has no audio stream.
- The accessible Drive `AIC2026/B02/raw` folder lists 21 video ZIPs, but
  `Videos_M010.zip` is **zero bytes**. The other 20 have nonzero sizes. The
  31 M10 keyframe videos therefore cannot yet be confirmed against source
  video. Do not treat an old 583-video ASR plan as complete for B02.
- No Batch 2 video ZIP was found at the repository root or in the canonical
  local `data/archives` tree. Do not download or extract all source videos to
  solve this: preserve bounded per-video scratch processing for later phases.

The structural audit does not read every compressed image or recompute member
CRCs. A separate [local SHA-256 baseline](../outputs/checkpoints/b02-p0/archives.sha256)
does hash the complete bytes of all 27 local B02 reference/object/keyframe
ZIPs. These hashes detect later changes to the local files; without an
organizer-published digest they do not, by themselves, prove agreement with
the upstream source. Video ZIPs have not been hashed locally.

## Canonical identity and time

For every B02 map row, keep:

1. `batch_id` and the *exact organizer video ID* (including `-` versus `_`);
2. the one-based keyframe ordinal `n` as the stable keyframe identity;
3. `timestamp_ms = round_half_even(Decimal(pts_time) * 1000)`;
4. `frame_idx` as advisory provenance only for N001–N100.

Deduplicate by `(batch_id, video_id, n)`, not by `frame_idx` or timestamp.
For N001–N100 traffic-camera videos, never derive a submission timestamp from
`frame_idx / fps`; KIS and Q&A submit the PTS-derived millisecond timestamp.
N001–N100 are ineligible for TRAKE. M/S are not subject to this exclusion.

Before any B02 import, audit each map for finite, nonnegative PTS with no
decrease; contiguous ordinal sequence; valid `frame_idx`; and exact
agreement of ordinal sets with image/object members and feature row counts.
Repeated N `frame_idx` with distinct PTS is a reportable collision, not a
reason to discard either keyframe. The existing B01 canonical repository and
fusion key use source-frame identity, so they must not be pointed at N videos
until a B02 ordinal-aware path has been implemented and tested.

The P0 audit identified nine pairs in `M08_V001` with identical PTS *and*
`frame_idx`, but different JPEG, feature vectors, and object data. Preserve
both ordinals. Do not deduplicate, invent a timestamp, or use either member of
such a pair as distinct ordered TRAKE proof until the ambiguity is resolved.
The [P0 manifest v3](../outputs/checkpoints/b02-p0/b02-p0-manifest-v3.json)
records all nine pairs and has `status=blocked` for strict temporal acceptance.
The [targeted evidence report](../outputs/checkpoints/b02-p0/m08-v001-pts-collision-evidence.json)
records the JPEG ZIP CRC/size, vector-row comparison, and object JSON ZIP
CRC/size for each pair. The audit CLI deliberately exits with code 2 while
these unresolved pairs remain; do not reinterpret that as an archive-count
failure.

## Go/no-go for P1

P1 may start only after the P0 audit produces a deterministic manifest with
zero map/image/object/feature alignment errors and documents exceptions.
`M08_V001` temporal ambiguity must be explicitly quarantined, not silently
treated as clean. The missing M10 video source blocks a claim of 614/614
source-video or ASR coverage, but does not block read-only keyframe/reference
preparation.
Feature dimensionality alone does not prove encoder compatibility; P1 must
verify the supplied vectors against the matching text/image encoder before
activating text-to-visual retrieval.

Drive source folder: the organizer-provided B02 share (link intentionally not published).
