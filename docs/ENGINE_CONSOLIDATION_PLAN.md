# Engine consolidation and controlled comparison

Status: staged plan approved; public-safe Cobra cleanup committed as dd67e04.
Archived v18.3 reference and evaluation-first implementation approved 2026-11-06.
The evaluation port is implemented; search alignment and release gates remain.

## Reference selection (2026-11-06)

Recommend the archived **v18.3.0 deployment source**, not today's root source,
as the behavioral reference. The deployment changelog records a 110-day first
run, a 1722 peak, and 1661 stable rating. These are historical observations,
not proof that it beats every later version in a controlled match.

Recover the reference from commit
`98706258b8bd093b144f11d286df8b9d31d9df7d`, directory
`lichess/engines/V7P3R_v18.3_20251229/src/`.
Do NOT use that commit's root `src/`: its engine header and UCI identity still
say v18.2. The archived engine and UCI Git blobs match the local archive;
the fast evaluator also matches the commit's root evaluator.

Reference SHA256 fingerprints:

| File | SHA256 |
|---|---|
| v7p3r.py | 3A41C478DB7ED96AB30856EB95052AF4C080FE6A2DD293F180384DDB49E9E2AD |
| v7p3r_fast_evaluator.py | AF554D64D6E9482F03740409A69E50182379EAE1113F98FF585F1CC2B04ACE35 |
| v7p3r_uci.py | 025FAABB78D6196BE8DABD7358AEDE81BEB0F2A67C7924595649E528D3D6B265 |

Use the complete archived source directory for dependencies, not a mixture
of current and historical Python modules. Preserve it unchanged.

Reasons not to choose alternatives automatically:

- v18.4 was rolled back for declining results.
- v18.3's second deployment had a documented 37.5% time-forfeit rate (3/8).
  Timing/dead-code failures must not be blindly reproduced in C#.
- v18.5 strips overhead, but this changelog provides no corresponding results.
- v18.6.3 is marked deploying/test without a completed performance record.
- The changelog's active-version fields conflict and its last-updated date
  predates some entries. It cannot establish current production identity.
- The success-analysis document has a peak-date inconsistency and discusses
  v18.4 aspiration windows; do not infer v18.3 behavior from those examples.

### First Cobra v4 implementation slice

Continue in the existing Cobra branch and root source, as requested. No new
branch, reset, deployment, or automatic commit is needed.

1. Establish build and UCI baselines before edits.
2. Port the archived fast evaluator's material values, exact PST orientation,
   phase predicates, and strategic terms, plus engine-level adjustments on
   the active evaluation path. Do not retain extra Cobra terms invisibly:
   this would compare different heuristics rather than language implementations.
3. Add deterministic component and total-score parity fixtures covering both
   sides to move, opening/middlegame/endgame boundaries, and asymmetric positions.
4. Update Cobra identity/version and development documentation for
   `4.0.0` as an unreleased candidate; do not claim a finished twin or release.
5. After evaluation parity, align search incrementally: ordering, TT semantics,
   quiescence/check handling, mate distance, repetition, and deadline enforcement.
   Existing Cobra search checks time only between iterations; resolve this before
   timed live matches. Preserve board state on search interruption.
6. Document intentional corrections separately from literal parity. Run
   fixed-depth and equal-time matches only after the relevant checks pass.

Approval requested: selected archived v18.3 reference and evaluation-first
implementation slice. Existing history and the cleanup commit remain the
rollback points; no additional engine freeze unless requested.

Approval received with a requested freeze. No existing freeze utility was found.
A hash-verified source-only snapshot was created outside both repositories at
`C0BR4 Chess Engine/engine-freezes/pre-v4-20261106-082249`.
Runtime credentials, active games, and VM state were not copied.

First-slice result: Cobra Release build passed; 7,832 Python-oracle evaluations
matched each component and total. Default search and UCI eval use the new
evaluator. Search parity is not complete. Existing UCI `perft 3` returned 20
instead of 8,902, recorded as an unresolved release blocker, not a passed check.

## Goal

Realign Cobra v4 with a pinned V7P3R reference, validate both locally, then
operate both bots for 7-14 days. Select one engine backend for `v7p3r_bot`.
Retire the `c0br4_bot` service after explicit approval, preserving its source,
builds, and rollback configuration. Do not delete the Lichess account.

Neither language choice nor nodes per second establishes playing strength.
The comparison must measure playing strength, reliability, and resource use.

## Verified local starting point

- V7P3R is on `main` with no worktree changes at inspection.
- The V7P3R source header identifies v18.6.3 as a test build. This is not
  evidence of the engine version currently running in production.
- The root `deployment_log.json` required by repository instructions is absent.
- The PowerShell deployment script uses a legacy local destination, skips
  validation while reporting it as passed, and leaves production upload manual.
  It reads an external token but also copies a token-bearing configuration into
  its staging package. Do not use this as a verified release process yet.
- Cobra is on `lichess-classical-v3.4`; its project declares version 3.4.0
  and targets .NET 6. Generated historical build files are modified and some
  tracked deployment files are deleted. Preserve these changes before cleanup.
- Both `dotnet` and `gcloud` are available locally. Cloud authentication,
  actual VM topology, running versions, capacity, and service health remain
  unverified.

## Phase 1: preservation and read-only inventory

1. Use dedicated feature branches in both repositories if approved.
2. Record commit IDs and preserve uncommitted Cobra changes separately; history
   does not preserve uncommitted edits. Do not reset or discard the worktree.
3. Offer the existing engine-freeze procedure before changes. Locate and inspect
   it before running it; never put credentials in a tracked snapshot.
4. Inspect existing match tooling before adding another harness.
5. Inventory cloud instances, containers, engine paths, service configuration,
   resource limits, and running UCI identities using read-only commands.
   Never print token values or unfiltered secret-bearing configuration.
6. Reconcile deployed versions with source before choosing the reference.

## Phase 2: reproducible and secret-safe local operation

- Repair the existing deployment entry point rather than add competing scripts.
- Separate source-only engine packages from private runtime credentials.
  Never bundle credentials into release archives or public build artifacts.
- Require explicit engine/backend, version, destination, and validation status.
- Add a local validation/dry-run mode with no cloud mutations.
- Record source commit, package hash, runtime, UCI identity, and test results.
  Failed or skipped checks must never be reported as passed.
- Test UCI readiness, legal moves, search deadline handling, process exit,
  failure reporting, and rollback selection before deployment.
- Establish the required deployment log without inventing historical entries.
- Keep old engines locally. Ignore rules alone do not untrack existing files:
  review any index cleanup and preserve the local copies.

## Phase 3: Cobra v4 alignment

- Pin V7P3R behavior from the selected reference, not historical feature claims.
- Map evaluation terms, score perspective, phase calculation, move ordering,
  transposition bounds, quiescence/check handling, repetition, mate scoring,
  principal variation, and time control operation by operation.
- Port incrementally, preserving Cobra's verified board/move-generation core.
- Require matching-position evaluation and fixed-depth search checks, with
  documented tie-breaking differences, before calling the engines twins.
- Compare both equal-work search behavior and equal-time playing strength.
  Different speed naturally produces different depth at equal time.
- Update Cobra version references and change documentation for v4 candidates.
  Keep the V7P3R reference unchanged unless a separately approved fix is needed.

## Phase 4: local acceptance and live trial

- Pass the repository regression and release gates before production changes.
- Use paired openings with colors reversed, identical time controls, equal
  resource budgets, and controlled book settings in local matches.
- Run at least 50 games, with color and time-control coverage; preserve PGNs,
  failures, exact versions, settings, and elapsed/resource measurements.
- Only then deploy approved candidates with private per-account credentials,
  a tested rollback path, and verified UCI/service readiness.
- Start the 7-14 day observation window after both candidates are healthy.
  Pin versions throughout; restart the window after material engine changes.
- Use scheduled direct challenges with concurrency limits and backoff.
  Do not interrupt active games or overwhelm shared VM resources.
- Treat public rating changes as supporting evidence, not a controlled result:
  opponent mix and account rating history differ.
- Report wins/draws/losses, uncertainty, time forfeits, illegal moves, crashes,
  move latency, CPU, peak memory, and actual infrastructure cost.
- Inconclusive results extend testing; do not automatically declare a winner.

## Phase 5: consolidation

After user approval of the evidence:

1. Configure `v7p3r_bot` to launch the winning UCI backend.
2. Verify account identity, engine identity, readiness, and initial games.
3. Drain active games and disable `c0br4_bot` auto-restart/challenge scheduling.
4. Stop only its identified service/container. Do not stop a shared VM.
5. Decommission a dedicated VM only after verifying no other workloads depend
   on it and receiving explicit approval. A stopped container alone does not
   eliminate the cost of a running VM.
6. Preserve private rollback assets and both public source histories.

## Approval boundary

Approve branch/rollback preparation and optional engine freeze first.
Implementation proceeds locally after that feedback. Cloud deployments,
paid resource creation, service shutdown, and account/token changes require
separate explicit approval. No live services have been changed by this plan.
