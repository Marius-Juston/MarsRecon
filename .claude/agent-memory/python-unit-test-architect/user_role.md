---
name: user-role
description: User is closing unit-test coverage gaps in MarsRecon src/dataset toward ~100%
metadata:
  type: user
---

User leads a focused effort to push `src/dataset/` unit-test coverage as close to
100% as reasonable with deterministic, isolated tests. Works file-by-file
(base.py, rdr.py, dtm.py, sampler.py, geometry.py).

Preferences observed and validated:
- OK with `# pragma: no cover` on genuinely defensive/unreachable lines
  (abstract `...` bodies, `width<0` on `.bounds`, stride>0 guards) **provided
  each is justified in the report**.
- Prefers reviving stale `@pytest.mark.skip` classes (fixing the assertion that
  rotted) over writing parallel duplicates — several skips were just stale
  logger-message / signature assumptions, not real incompatibilities.
- Wants the final report under 400 words: per-file numbers, remaining uncovered
  lines w/ one-line reason, pragmas added w/ justification, total.

See [[project-test-infra]] for the environment traps.
