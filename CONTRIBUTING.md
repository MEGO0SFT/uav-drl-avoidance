# Contributing

[← Back to README](README.md) · [Roadmap](ROADMAP.md)

Thanks for looking. This repository is primarily a personal research archive that
is also maintained as a working project, so the most useful contributions are the
ones that improve **reproducibility**, not feature count.

## Most welcome right now

1. **Portability.** The Stage-1 code was written on Windows with hard-coded LAN
   IPs and Windows-style paths. Reports of (and fixes for) Linux/WSL runs, AirSim
   version pinning, and dependency conflicts are valuable.
2. **Benchmark harness.** Anything that makes the evaluation suite cheaper to run
   or harder to misconfigure — a scenario registry, a fixed seed set, a smaller
   smoke-test subset for CI.
3. **Documentation of the failure modes.** If you reproduce a collision pattern
   that `failure_analysis.txt` does not classify, that is a genuinely new data
   point.
4. **Isaac Lab pointers.** Stage-2 is in progress; concrete notes on PhysX raycast
   LiDAR throughput or block-sparse map kernels in PyTorch save real time.

Less useful: style-only refactors of the archived Stage-1 scripts, or new
features for a simulator other than Isaac Lab.

## Ground rules

- **Never commit binary artefacts.** `git ls-files` must stay text. Model weights,
  archives and videos belong in a GitHub Release. Run the guard before pushing:

  ```bash
  python scripts/repo_hygiene.py --all
  ```

- **Keep the archival branch intact.** `origin_26-0364` is the frozen snapshot
  matching the published paper. Do not rewrite its history, and do not "fix" the
  numbers there — if something needs correcting, do it on `main` and document the
  difference.
- **Numbers must be traceable.** If a change affects a reported metric, say which
  table or figure it changes and why. `benchmarks/` is the single source of truth
  for published values.
- **One logical change per pull request**, with a message that explains the
  *reason*, not just the diff. `CHANGELOG.md` entries are welcome in the same PR.

## Local checks

```bash
python scripts/repo_hygiene.py --all     # no oversized / binary files
python -m compileall -q RL_scripts methods scripts   # syntax
ruff check .                             # lint (once configured)
```

## Reporting a problem

Please include: the branch, your OS + Python version, the exact command, the full
traceback, and — if it is a simulation issue — the AirSim version and the modified
`settings.json`. Most Stage-1 failures are configuration, not code.

## Contact

Open an issue. For anything related to the published article, use the DOI in
[`paper/README.md`](paper/README.md).
