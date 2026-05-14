# Technical notes

Deep-dive reference material maintained alongside the codebase. These pages are kept in sync
with `MarsHiRISE_Technical_Reference.md` and `MarsHiRISE_DTM_Technical_Reference.md` at the
repository root via [pymdown-snippets](https://facelessuser.github.io/pymdown-extensions/extensions/snippets/),
so the markdown source has a single home.

- [HiRISE RDR reference](hirise-rdr-reference.md) — PDS3 labels, calibration, RED/IRB
  channels, footprint conventions.
- [HiRISE DTM reference](hirise-dtm-reference.md) — stereo DTM IMG format, nodata semantics,
  orthoimage alignment.
- [Refactor notes](refactor-notes.md) — load-bearing files >1000 LOC with known split
  candidates; test patches that need updating after the structural refactor.
