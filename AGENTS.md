## Agent skills

### Issue tracker

Issues and specs for this repo are tracked using GitHub Issues. See `docs/agents/issue-tracker.md`.

### Triage labels

The default set of triage labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`) are used in this repo. See `docs/agents/triage-labels.md`.

### Domain docs

This is a single-context repository (`GLOSSARY.md` + `docs/adr/` at the repo root). See `docs/agents/domain.md`.

### Coding standards & review

Review standards and artifact conventions are documented in `CODING_STANDARDS.md`.

### Invariant checks

Run `bash tests/check_invariants.sh` to verify frozen runtime core invariants, unit tests, prebuilt harness, and glossary domain terminology before proposing or merging changes.
