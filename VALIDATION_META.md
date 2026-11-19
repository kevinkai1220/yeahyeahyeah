# Executed validation and pilot results

Validated source: `validation/source_manifest.json`. Original v1/v2 source folders were preserved.

## Tests

```text
......................................................                   [100%]
54 passed in 30.93s
```

Full test command:

```bash
PYTHONPATH=.deps ../marl_strategic_exploration/.venv/bin/python -m pytest tests -q
```

Tests cover legacy behavior, reward/noise semantics, exact ordered combos, trap validation, nonterminal mode resets, coverage, train/test separation, checkpoint equality, recurrent PPO versions, downstream GAE, TD replay, frozen/world isolation, paired snapshots, recurrent opponents, interrupted optimizer resume, launcher retries/unique outputs/locks, optional ablations, and PNG/PDF generation.

## Executed commands

```bash
scripts/meta.sh launch --config configs/meta/smoke.json --out runs/meta_smoke --regimes B --seeds 0
scripts/meta.sh launch --config configs/meta/pilot.json --out runs/meta_pilot_verified --regimes A1 A2 A3 B --seeds 0 1 2 3 4
scripts/meta.sh hardware > validation/hardware.json
```

An earlier exploratory pilot remains in `runs/meta_pilot`; the final coverage-prioritized implementation and results below use **`runs/meta_pilot_verified`**. The final launcher finished with exit code 0.

## Pilot scope

- 20 completed jobs; no failed jobs; 4 regimes × 5 seeds.
- Per job: 6 training worlds, 6 pretraining episodes, 6 main episodes, hidden size 16, one PPO epoch.
- Pretraining B=18, K=24; test B in {9,18}, K=24; frozen and online modes; online interval 8.
- Six held-out worlds per job; mixed-world jobs have two worlds of each type. World identities overlap between single-type and mixed evaluations by design; these are not 120 independent cross-regime parameters.
- 1920 world/method/mode/budget metric records; 72000 logical test interaction steps across all methods.
- 54 PNG and 54 PDF figures; aggregate CSV/JSON included.
- History ranking prioritizes pair coverage. Selected informative pretraining histories cover all nine ordered pairs at B=18; hidden parameters are never consulted for this selection.

## Measured online exploitation mean reward

B=18, K=24. Each entry is a mean across 5 seeds × 6 held-out worlds for that regime. Bracketed values are hierarchical seed/world bootstrap 95% intervals (300 replicates). For B, the table aggregates all three types; per-type values are in the CSV.

| Regime | Single | Q-only | Dual Neural | Dual Q |
|---|---:|---:|---:|---:|
| A1 | 0.855 [0.526, 1.117] | 1.208 [0.746, 1.714] | 0.854 [0.544, 1.173] | 1.022 [0.692, 1.385] |
| A2 | 0.583 [0.317, 0.883] | 1.525 [1.187, 1.847] | 0.692 [0.364, 1.144] | 1.359 [1.032, 1.776] |
| A3 | 0.508 [0.285, 0.717] | 1.577 [1.265, 1.881] | 0.561 [0.292, 0.829] | 1.386 [0.988, 1.779] |
| B | 0.667 [0.391, 0.984] | 1.315 [0.974, 1.643] | 0.499 [0.127, 0.801] | 1.199 [0.878, 1.500] |

**These tiny runs validate the pipeline, not comparative learning performance.** Only six optimizer steps per neural stage are far too few to claim convergence or superiority. Q methods retain online updates in the frozen-neural evaluation condition.

## Costs and hardware

- Current runtime: PyTorch 2.15.0+cpu; CUDA build/availability absent; nvidia-smi unavailable. All actual jobs used CPU. GPU dispatch and retry logic were mocked in tests; no actual CUDA execution is claimed.
- Sum of measured training/evaluation job runtimes: 126.31 seconds. This excludes process startup, final plotting, tests, and earlier exploratory runs.
- Per job: 468 pretraining interactions (including unused history candidates); 252 logical training interactions per method; 900 physical main-training interactions including Q diagnostics and paired-prefix reuse.
- Per job: 144 pretraining optimized transitions; Single 252; Dual exploiter 144; explorer 108. Each neural stage performs 6 optimizer steps. Single has 1,892 trainable parameters; Dual Neural 3,784.
- Evaluation history-only reference adds diagnostic interactions, separately recorded in `evaluation/protocol.json`; these never update evaluated agents.

## Inspect results

- [Aggregated CSV](examples/pilot/aggregated.csv)
- [Aggregated JSON](examples/pilot/aggregated.json)
- [Mixed online adaptation](examples/pilot/adaptation_B_online_18_24_all.png)
- [Training progress](examples/pilot/training_progress.png)
- [Paired Type 3 history](examples/pilot/history_type3.png)
- [Hardware report](validation/hardware.json)
- [Test output](validation/pytest.txt)

## Not executed / remaining limitations

Full default 300-pretraining/1200-training-episode studies, large test sets, and actual multi-GPU jobs were not executed. This CPU-only environment was used for smoke and small pilots. No DDP implementation or exact recurrent-PPO-opponent oracle is claimed. Adaptive PPO-opponent support and optional Q-aware/privileged/discounted/K-only branches were unit-tested but were not used in the main pilot. A compatible opponent checkpoint is required to include learned opponents in a new study.

## GitHub / GPU-server packaging validation

The published repository adds portable `setup_gpu.sh` / `run_multi_gpu.sh`
entrypoints, a `--require-gpu` guard, and per-device CUDA probe synchronization.
The original pilot's learning implementation is unchanged by these packaging fixes.

- Full suite after packaging: **57 passed**, recorded in `validation/publish_tests.txt`.
- Bash syntax checks passed for all three entry scripts.
- Portable interpreter override, no-GPU rejection, and per-device synchronization
  are tested. Actual CUDA installation/execution still requires the GPU server.
- Compact measured results are bundled under `examples/pilot/`; raw trajectories,
  checkpoints, virtual environments, and installed dependency files are not uploaded.
- `validation/source_manifest.json` describes the original local pilot source;
  `validation/publish_manifest.json` describes the published source and entrypoints.

## Throughput revision

The first GPU launcher used one independent job per device and scalar recurrent
inference. The revised server wrapper defaults to two jobs per GPU and batches
shared frozen policies across test worlds (default batch size 32). Adaptive neural
weights/optimizers remain per world; training PPO batch/update schedules and logical
interaction budgets are unchanged. Scalar action probabilities, log-probabilities
and values now share one device-to-host transfer. Discarded terminal recurrent
states are no longer replayed after PPO updates.

- **63 tests passed** (`validation/throughput_tests.txt`), including scalar/batched
  trajectory equivalence on CPU, independent online adaptation, shared paired
  prefix identity, unchanged PPO weights when skipping terminal replay, and four
  concurrent jobs assigned to two mocked GPUs without output collisions.
- The updated launcher completed an end-to-end CPU smoke run with frozen batch
  size 2 (including a partial last batch), both evaluation modes, and PNG/PDF output.
- Measured CPU frozen-evaluation benchmark: 32 worlds, hidden=64, B=32/K=128,
  all four methods plus Q replay/reference/diagnostics, three timed repetitions:
  scalar median **7.127 s**, batch-32 median **2.330 s**, **3.059×** throughput.
  Details: `validation/throughput_cpu.json`. This excludes training and disk I/O;
  it is not a measured GPU speedup or GPU-utilization claim.
- GPU utilization/throughput must still be measured on the user's CUDA server.
  `python -m meta_exploration.benchmark --device cuda:0 --worlds 32` provides the
  same comparative benchmark. Raising worker count can hurt if CPU/memory bound.
- Runtime worker/batch controls are stored separately from scientific config,
  preserving compatibility with existing checkpoints. Batch floating-point rounding
  can differ slightly on GPU; per-world RNG streams, observation boundaries and
  the paired evaluation design remain unchanged.
