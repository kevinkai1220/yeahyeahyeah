# Three-action meta-exploration framework

This folder extends a copy of `marl_strategic_exploration_v2`. The source v1/v2
folders are untouched. Legacy modules, tests and notebooks
are retained; new experiments live in `meta_exploration/` and `runs/meta_*`.
The previous README is preserved as `README_V2.md`.

## GPU 서버에서 바로 실행

Linux/NVIDIA GPU 서버에서 아래 명령을 실행하세요. Python 3.11+를 권장합니다.

```bash
git clone https://github.com/kevinkai1220/yeahyeahyeah.git
cd yeahyeahyeah
bash scripts/setup_gpu.sh
bash scripts/run_multi_gpu.sh
```

`setup_gpu.sh`는 프로젝트 `.venv`를 만들고, 이미 설치된 CUDA PyTorch가 있으면
재사용합니다. 없으면 기본 CUDA 12.6 wheel과 나머지 의존성을 설치합니다.
서버 driver에 다른 CUDA wheel이 필요한 경우
[PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)에 맞춰 지정하세요:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 bash scripts/setup_gpu.sh
```

`run_multi_gpu.sh`는 모든 정상 **visible GPU**에 기본 2개 worker를 배치해
A1/A2/A3/Mixed × 5 seeds의 20개 job을 동적으로 실행합니다. 별도의 `torchrun`은
필요 없습니다. GPU가 하나도 정상 동작하지 않으면 CPU로 조용히 넘어가지 않고
오류를 냅니다. 실행 중 특정 GPU에 CUDA 오류가 생기면 해당 worker는 CPU로 재시도하고
`excluded_devices.jsonl`에 기록합니다. CUDA 실제 실행은 이 CPU 개발 환경에서
검증하지 못했으므로 서버의 연산 검사와 작은 pilot을 먼저 실행할 수 있습니다:

```bash
# 작은 20-job pilot
bash scripts/run_multi_gpu.sh configs/meta/pilot.json runs/gpu_pilot

# 전체 기본 실험
bash scripts/run_multi_gpu.sh configs/meta/default.json runs/full_study

# 특정 GPU만 사용
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_multi_gpu.sh

# SSH 종료 후에도 실행
nohup bash scripts/run_multi_gpu.sh > multi_gpu.log 2>&1 &
tail -f multi_gpu.log
```

같은 명령을 다시 실행하면 완료된 job은 건너뛰고 중단된 job은 checkpoint에서
재개합니다. 결과는 `runs/full_study/`, 그래프·CSV는 `runs/full_study/figures/`에
저장됩니다. 각 job의 진행 로그는 `REGIME/seed_N/worker_*.log`,
`training.jsonl`에 있습니다. `runs/`와 가상환경은 Git에 포함하지 않습니다.
작은 CPU pilot의 요약 결과와 대표 그림은 [`examples/pilot/`](examples/pilot/)에 포함했습니다.

## GPU 사용률 / throughput 조정

이 실험의 GRU는 작고 환경·Q-table 연산은 CPU에서 실행됩니다. GPU를 여러 개
배정하는 것만으로 높은 사용률이 보장되지는 않습니다. 현재 구현은 다음 비용을 줄입니다:

- Actor의 확률·log-prob·value를 step당 한 번에 CPU로 가져옵니다.
- Frozen explorer는 여러 test worlds를 함께 추론합니다. Frozen evaluation의
  Single/Dual Neural도 동일 weights를 공유하는 world들끼리 batch 추론합니다.
- Online adaptation은 world별 weights와 optimizer를 독립적으로 유지합니다.
  PPO episode/update 횟수, interaction budget, reward objective는 바꾸지 않습니다.
- Training episode 종료 후 버려질 recurrent state를 다시 계산하지 않습니다.
- GPU당 복수 job으로 CPU 환경·Q 연산 중 생기는 대기 구간을 겹칩니다.

기존 실행을 중단한 뒤 `git pull`하고 같은 output 경로로 재개할 수 있습니다.
Worker 수와 frozen batch size는 runtime 설정이므로 기존 scientific config와
checkpoint를 그대로 사용할 수 있습니다.

```bash
git pull --ff-only
WORKERS_PER_GPU=2 FROZEN_BATCH_SIZE=32 bash scripts/run_multi_gpu.sh

# 여유 GPU 메모리/CPU 코어가 충분할 때 비교해 볼 설정
WORKERS_PER_GPU=4 FROZEN_BATCH_SIZE=64 bash scripts/run_multi_gpu.sh

# 서버에서 scalar vs batched frozen evaluation의 실제 처리량 측정
.venv/bin/python -m meta_exploration.benchmark --device cuda:0 --worlds 32 --out runs/throughput_gpu.json
```

동시 worker를 무조건 늘리면 CPU 또는 메모리 경합으로 느려질 수 있습니다.
GPU 사용률만 보지 말고 `launcher.jsonl`의 job runtime과 `runtime.json`의
training/evaluation 시간을 비교하세요. `scheduling.json`과 evaluation의
`protocol.json`에 runtime 설정이 기록됩니다. Batch 연산은 부동소수점 반올림이
scalar와 미세하게 다를 수 있지만, world별 RNG/state 분리와 paired history는
유지됩니다. `FROZEN_BATCH_SIZE=1`로 scalar 경로를 사용할 수 있습니다.

## Other commands

```bash
# Hardware report / tests
bash scripts/meta.sh hardware
.venv/bin/python -m pytest tests -q

# CPU smoke test (general launcher permits CPU fallback)
bash scripts/meta.sh launch --config configs/meta/smoke.json --out runs/my_smoke --regimes B --seeds 0

# Individual stages: keep config/regime/seed/output identical
bash scripts/meta.sh pretrain --config configs/meta/default.json --regime B --seed 0 --out runs/custom/B/seed_0
bash scripts/meta.sh train --config configs/meta/default.json --regime B --seed 0 --out runs/custom/B/seed_0
bash scripts/meta.sh evaluate --config configs/meta/default.json --regime B --seed 0 --out runs/custom/B/seed_0
bash scripts/meta.sh plot --out runs/custom

# Inspect histories after the smoke test above
bash scripts/meta.sh inspect --history runs/my_smoke/B/seed_0/evaluation/frozen_B3_K5/test-type1-1_dual_neural.jsonl runs/my_smoke/B/seed_0/evaluation/frozen_B3_K5/test-type1-1_dual_q_table.jsonl --out runs/my_smoke/paired_history
```

For a CPU-only development installation, create a virtual environment and run
`python -m pip install -e '.[meta]'`. The GPU setup script intentionally requires
working CUDA. `META_PYTHON=/path/to/python bash scripts/meta.sh ...` selects an
existing interpreter without modifying it.

`train` automatically runs/resumes pretraining if needed. `run` performs training
and evaluations. `launch` additionally aggregates and plots. Configs are JSON,
merged with `meta_exploration/config.py` defaults. Each run saves the fully resolved
config; changing it requires a new output path.

## Modules and reuse

| Module | Role |
|---|---|
| `marl/models.py` | Existing encoder → GRU → actor/critic reused; optional dimensions added with legacy defaults intact |
| `meta_exploration/environment.py` | Three-action observation, stochastic worlds, trap validation, snapshots, adapted exact DP oracle |
| `meta_exploration/learning.py` | Existing sequence-replay/clipped-PPO design extended with GAE, context masks, online versions; extended tabular TD |
| `meta_exploration/training.py` | Observed-only history curriculum, shared pretraining, cumulative separate-policy training, checkpoint resume |
| `meta_exploration/evaluation.py` | Independent-world online/frozen adaptation, paired histories, diagnostics, metrics |
| `meta_exploration/hardware.py` | GPU probes, dynamic independent-job scheduling, retries, CPU fallback, resource logs |
| `meta_exploration/plotting.py` | Actual-log aggregation, hierarchical intervals, seven figure families, history inspection |
| `meta_exploration/__main__.py` | Pretraining/training/evaluation/launcher/plot/inspection CLI |

Legacy `marl.environment`, `marl.ppo`, and the other legacy entrypoints retain their
four-action behavior. The new package is an explicit experimental extension,
not a silent change to old experiment semantics. The new PPO implementation is
needed because the old dual network shared a GRU and had no online GAE/versioning.

## Environment semantics

- Simultaneous A/B/C actions. Agent chooses before observing the opponent's action.
- Strict R1 → R2 → R1, as in v2. No interruption rule existed, so none is invented.
- Only the configured B+K horizon terminates an episode. **R2 → R1 never stops TD,
  discounting or value bootstrap. B/K also does not reset the environment.**
- Observations: mode, previous agent/opponent actions, previous observed reward.
  The 11-vector contains two 4-way one-hot action encodings (3 is reset sentinel),
  2-way mode, and reward. On R1 after a pair, previous actions remain visible,
  preserving v2 semantics. Hidden parameters, type, IDs, mean and bonus are absent.
- Type 1: fixed current-action reward matrix, sampled dominant, conditional or
  risky preference pattern. There is no combo bonus.
- Type 2: positive bonuses only on same-action pairs in R2. Type 3: positive bonuses
  only on ordered different-action pairs in R2. Base reward applies in both modes.
- Combo constituent actions have **negative immediate reward**. Another action
  has positive immediate reward. The main combo beats that trap over two steps.
  Type 3 must also beat every fixed-action repeat. Type 2 intentionally allows
  repeating its optimal combo action. Rejection sampling checks these inequalities
  conservatively over every pair of opponent actions, rather than just iid means.
- Easy/medium/hard vary trap strength, combo advantage margin and number of bonuses.
  Sigma is independently configurable, default 1.0; zero is deterministic.
  Optional bounded opponent-dependent bonus perturbations are supported.
- Parameter RNG uses `(world seed, split tag)`; noise and opponent RNGs are separate.
  Means remain immutable. Agent/opponent rewards use independent Gaussian draws.
- Opponents: uniform iid, cyclic A/B/C, optional frozen recurrent PPO checkpoint.
  The PPO opponent receives the same observation schema and a symmetric reward
  function with roles swapped. It never receives the agent's current action.
  Main defaults use iid/cyclic because no compatible three-action opponent checkpoint
  was available. Legacy four-action checkpoints are rejected explicitly.

A new three-action policy is exported after pretraining as
`checkpoints/three_action_policy.pt`. It can be used as a frozen PPO opponent by
setting `environment.opponents` and `environment.opponent_checkpoint` in a **new**
run's config. Its pretraining worlds must satisfy that run's training-type scope;
never train it on test worlds. This enables the existing learned-opponent family
without pretending the incompatible legacy checkpoint can be reused.

## Training and history curriculum

A1/A2/A3 train on one type; B cycles through all three. All regimes use the same
total training-world count, episodes and interaction lengths. Default counts are
60 train worlds, 300 pretraining episodes, 1200 main episodes, 5 seeds, 100 test
worlds per seed/regime. Mixed samples are balanced to within one world. Per-type
world seeds are stable across regimes; training/test parameter hashes are disjoint.

Pretraining collects a pool of prefixes from coverage, random and narrow repeated
behavior. Coverage schedules all **nine ordered pairs** every 18 interactions;
shorter B cannot cover all pairs in one prefix, but pair permutations vary across
worlds/seeds. Every candidate starts an independent environment and its K continuation
uses that candidate's actual post-prefix state. Candidate interaction cost is counted,
including rejected candidates. Only the selected history conditions the exploiter.

Ranking prioritizes observed pair coverage; reward contrast divided by within-pair
variability breaks ties through a bounded score. With B>=18, the high stratum
contains full nine-pair histories (coverage collectors form one third of the pool). The pool is split into relative low/middle/high score strata, labelled
uninformative/ambiguous/informative. The default selection ratio is 0.4/0.3/0.3,
with each stratum visited first and weighted-deficit scheduling thereafter. These
are **observational strata, not a guarantee of hidden-optimal-combo evidence**.
Coverage counts, stratum and contrast are logged; true combo information is used
only in separate diagnostics. `privileged_curriculum=true` is an explicitly labelled
optional ablation that injects a known combo prefix into one pool candidate.

The exploiter conditions on the prefix, then receives PPO updates on its own K
on-policy actions. The trained state is saved at
`checkpoints/shared_exploitation_pretrain.pt`. Single and Dual Neural load identical
copies of this state; equality is asserted and saved in `identical_initialization.pt`.
Both have the same architecture. The explorer is a separate model with separate
parameters and Adam optimizer. Models persist across training worlds; recurrent
state and per-world Q tables do not.

The main objective is `lambda * sum(B rewards) + sum(K rewards)`, with neural
`gamma=1.0`. GAE lambda (default .95) is distinct from exploration reward lambda
(default .1). Legacy advantages are normalized within each updated rollout/policy segment.
The corrected config uses policy-specific multi-episode batch normalization; see the diagnostics section below.
The neural PPO minibatch is one complete recurrent trajectory, with configurable
PPO epochs. No time-shuffled recurrent minibatches are used.

| Method | Main training | Test |
|---|---|---|
| Single | Full B+K on-policy trajectory, weighted rewards, GAE across B/K | Same policy for all steps; frozen or interval PPO |
| Q-only | Training-world online-Q diagnostics; discard Q after every world, no transfer | New Q; epsilon-greedy TD throughout B+K |
| Dual Neural | One frozen-during-collection B+K rollout; B explorer and K exploiter receive separate PPO losses/optimizers | Frozen explorer produces B; independent exploiter replays history, then frozen/online K |
| Dual Q | Shares the Neural-trained explorer; paired training-world Q diagnostics, no extra neural training | Same B history/state as Dual Neural; reverse offline TD replay then online Q |

Legacy full-episode dual GAE uses the corresponding acting policy's value at each step,
including the exploiter value at the B/K boundary. Exploration actions therefore
receive downstream credit; exploitation never learns the explorer's actions as
its own PPO actions. Recurrent history replay can carry gradients through context,
without adding off-policy prefix actions to the exploiter surrogate.

Optional `q_aware=true` trains an additional explorer using actual downstream Q
return; it is a fifth ablation method and never replaces the main shared-prefix
comparison. `single_k_only=true` is the requested optional Single ablation.
`gamma=.99` is a separate discounted-objective ablation. Q gamma defaults to .99,
so its Bellman objective differs from the main undiscounted neural objective;
set `q_learning.gamma=1.0` for a matched-discount ablation.

## Test protocol and information boundaries

Each method receives B+K logical test interactions on identical held-out worlds.
All checkpoints/optimizers are freshly cloned per world, budget and evaluation
mode. Frozen evaluation changes only neural recurrent state, while Q methods
continue TD learning; Q results are labelled accordingly, not a frozen-weight Q
counterpart. The explorer is frozen at test time in both evaluation modes.

Online PPO collects a segment with a single fixed policy version. At a truncated
boundary it bootstraps the next value without advancing hidden state; at the true
horizon it uses zero bootstrap. It checks rollout versions before optimization,
discards that buffer afterward, increments the version and reconstructs recurrent
state from observed history under updated weights. No future actual K reward is
inserted into an earlier update. Updates after the last observed action cannot
change the recorded return and are discarded with that world's model. Policy
version numbers are local to each controller/world, not global checkpoint IDs.

Paired Dual Neural/Q: collect B **once**, clone mode, previous actions/rewards,
opponent hidden state, opponent RNG and noise RNG. Replay the same history into
the neural GRU and Q-table respectively. An odd B preserves R2. Future trajectories
may diverge with actions. Main four methods have equal logical budgets even though
paired prefix reuse saves physical simulation work. Separate
`paired_differences.json` stores neural-minus-Q outcomes by paired world.

The Q state is `(mode, previous_agent, previous_opponent)`, including retained
R1 observations. It never includes current opponent action. This state is adequate
for iid/cyclic opponents and reward dependence; a recurrent adaptive opponent
can require longer history, so the table is not asserted to be sufficient for its
latent dynamics. Reverse TD replay allows a delayed R2 reward to update R1 actions.
Prefix end is not terminal. Epsilon decay, alpha, gamma and replay passes are config.

## Metrics, plots and interpretation

All reward curves and tables come from executed JSONL trajectories. No synthetic
performance values or extrapolated training results are generated.

- AUC@k is the **sum** of rewards in the first k interactions; normalized AUC is also
  stored. k larger than the executed horizon is omitted, not silently truncated.
- Exploitation return is the K-window sum; `exploitation_mean` is its per-step mean.
- Oracle: exact finite-horizon DP for iid/cyclic opponent dynamics and mode. For
  recurrent PPO opponents oracle metrics are explicitly unavailable, not replaced
  with an iid approximation. Regret subtracts the realized trajectory's sum of
  conditional reward means, a Monte Carlo estimate of agent expected return; a
  single-world estimate can be negative. K-window oracle starts at the agent's
  actual post-B state. Oracle information never enters controllers.
- Discovery: attempted true positive-bonus pairs, positive observed reward, and
  repeated evidence are separate. Evidence heuristic requires at least two
  samples and a positive lower bound using 1.96 times standard error, with known
  experimental sigma as a diagnostic floor. This heuristic is not a calibrated
  multiple-testing discovery claim. False positive evidence is reported.
- Utilization: good pairs attempted in B and reused in K; separately, the best
  iid-mean two-step pair's K frequency. The latter is explicitly an iid-pair
  diagnostic, not an exact adaptive-opponent optimal policy statistic. Missing
  discovery gives undefined utilization rather than a misleading zero.
- History transfer gap: a reference uses only the same observed H, 50 reverse TD
  passes, and frozen greedy Q on a cloned environment. Its scored conditional-mean
  return minus the actual K conditional-mean return is reported. This empirical
  comparator **is not an upper bound** and can be worse than the agent. Extra
  reference interactions are reported separately and never feed either method.
- Seed/world hierarchical bootstrap gives 95% intervals; seed is the outer unit,
  worlds the inner. Mean, standard deviation, minimum, sample sizes and per-type
  summaries are exported. With only one seed these are world-only intervals.

Figure families, all PNG and PDF: adaptation mean/CI (per type and aggregate),
AUC comparisons, exploration-budget ablation, discovery/utilization, single-type
versus mixed generalization, cumulative training progress, representative paired
history panels. History panels show actions, reward, cumulative return, pair
frequencies, phase reward-sign counts, rolling action frequencies. A positive noisy
reward is labelled as such rather than called a true hidden-combo success.

## Outputs and costs

Each `REGIME/seed_N/` contains:

- `config.json`, `identity.json`, `costs.json`, `runtime.json`;
- `checkpoints/` shared initialization, current training, model and optimizer resume;
- `training.jsonl`, `training_histories/` for neural and Q training-world trajectories;
- `diagnostics/worlds.json` and separate true-mean/bonus history files;
- `evaluation/{mode}_B{B}_K{K}/` observed trajectories and metrics;
- `evaluation/metrics.json`, `paired_differences.json`, `protocol.json`.

`figures/aggregated.csv` and JSON summarize every method/regime/mode/B/K/type.
`costs.json` reports shared pretraining (including unused history candidates),
optimized transition counts, optimizer steps, parameters, logical training
interactions, Q-only diagnostic costs and shared-prefix savings. Dual has twice
the neural parameters of Single; the extra explorer updates are not hidden.
Q-only has no transferred parameters. Comparing test adaptation does not imply
identical total training cost. Training progress is separate from test metrics.

## Hardware and reliability

`hardware` records nvidia-smi, PyTorch/CUDA availability, visible-device mapping,
GPU memory and an actual matrix multiply on each visible GPU in isolated child
processes. Failed probes are excluded. A configurable number of independent workers per healthy GPU
pulls new jobs dynamically (server wrapper default: 2; general CLI default: 1). Small GRUs do not use DDP; this implementation does
not claim DDP support. With no healthy GPU it uses CPU workers. Config can force
CPU, select a CUDA device, or limit all-device usage. CUDA probes are real, not
inferred just from device count.

Workers write unique regime/seed directories and hold OS locks against accidental
concurrent writers. Writes to checkpoints/configs are atomic. Failed workers are
retried and resume model+optimizer state at the last checkpoint; unsaved episodes
are deterministically replayed. Evaluation reruns deterministically from its start.
Completed jobs are skipped after config validation. CPU/GPU runtime, per-worker
logs, failure reports and available GPU utilization/memory samples are saved.
No CUDA execution was possible on the supplied CPU-only runtime; scheduler GPU
behavior requires validation on a CUDA-enabled host.

## Validation and limitations

Run `tests/meta` for new semantics, or `tests` for both legacy and new tests.
The smoke config exercises odd B, both evaluation modes, all types and all four
methods. Pilot settings are deliberately tiny: six pretraining and six training
episodes per job; they validate execution and output, not research conclusions.
See `VALIDATION_META.md` for commands and actual run results.

Large experiments, robust adaptive-opponent studies and publication-level claims
require more training and a suitable compute environment. PPO opponent exact oracle
and DDP are not implemented. Main history categories measure observed informativeness;
no hidden-rule-independent collector can guarantee discovery of every hidden optimum
in an arbitrarily short noisy prefix.


## Corrected credit assignment and adaptation-speed diagnostics

Use `configs/meta/corrected.json` for new batch PPO training. The historical
default remains a legacy comparison; do not resume an old final checkpoint under
a changed objective. The corrected path batches independent episodes per policy,
normalizes advantages across that policy's batch, and uses the observed complete
suffix return for explorer boundary credit. Online test adaptation retains its
separate single-world normalization setting and never uses future rewards.

[Implementation, theory, tests, pilot commands and Phase 3–6 interfaces](docs/meta_learning_diagnostics.md)
include history interventions, finite-horizon oracle-normalized adaptation curves,
confirmed time-to-threshold, right censoring, RMST, paired bootstrap, and raw JSONL
backfill. Original full-study results are preserved. Full study is not started
by the diagnostic runner.

Actual CPU pilot results and limitations are recorded in
[the credit-diagnostics report](analysis/credit_diagnostics/REPORT.md).
Corrected training does not yet establish improved held-out neural performance;
CUDA execution remains unverified on the available CPU-only machine.

## History utilization and PPO optimization diagnostics

The checkpoint-only runner compares real, shuffled, reset and alternative observed
histories from a common post-exploration snapshot. It records preference switching,
ordered-pair probabilities/completion, observed-statistic probes and paired intervals.
A separate small Type 3 runner tests optimizer cadence and critic scaling without
changing historical defaults or overwriting existing runs.

See [server commands and experimental controls](docs/history_optimization_diagnostics.md)
and [executed CPU results and limitations](analysis/history_optimization_diagnostics/REPORT.md).
