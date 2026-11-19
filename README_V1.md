# MARL Strategic Exploration — Phase 1

고정된 `R1 → R2 → R1 → R2` 게임에서 reward 구조에 적응하는 전략을 비교합니다. 환경, generator, opponent, agent, PPO, oracle, 평가를 분리했습니다. Phase 2/3 및 다른 transition rule family는 구현하지 않습니다.

## 실행

Python 3.10 이상과 CPU PyTorch/NumPy를 사용합니다. 현재 작업 공간에는 `.venv`에 의존성이 설치되어 있습니다.

```bash
cd /home/kevinlees1220/marl_strategic_exploration
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m marl.cli run --config configs/smoke.json --output runs/smoke_new
```

새 환경에서는 다음처럼 설치합니다.

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e .
```

기본 실험 전체 실행:

```bash
.venv/bin/python -m marl.cli run --config configs/default.json --output runs/seed0 --seed 0
```

기본값은 `T=100`, `K=[10,25,50]`, training/evaluation task `256/128`, 업데이트당 `8 tasks × 8 rollouts`, 모델당 `1,000`회 업데이트입니다. PPO opponent를 먼저 학습하고, single-policy agent 한 개와 K별 dual-policy agent를 학습한 뒤 평가합니다. **Smoke 실행은 동작 검증이며 성능이나 수렴을 입증하지 않습니다.** 정식 실험은 상당한 CPU 시간이 필요합니다.

최종 연구 결과는 서로 다른 training seed로 최소 3회 반복합니다.

```bash
for seed in 0 1 2; do
  .venv/bin/python -m marl.cli run --config configs/default.json --output "runs/study_seed${seed}" --seed "$seed"
done
```

단계를 나누어 실행할 수도 있습니다. 후속 단계는 저장된 config를 사용합니다.

```bash
.venv/bin/python -m marl.cli prepare --config configs/default.json --output runs/staged
.venv/bin/python -m marl.cli pretrain --output runs/staged
.venv/bin/python -m marl.cli train --output runs/staged
.venv/bin/python -m marl.cli evaluate --output runs/staged
```

`prepare/run`은 비어 있지 않은 output을 덮어쓰지 않습니다. 기존 training log가 있는 모델의 재학습도 거부합니다. `evaluate` 재실행은 해당 디렉터리의 평가 결과를 다시 생성합니다. 학습 재개용 optimizer checkpoint는 범위에 포함하지 않습니다.

## 모듈

| 파일 | 역할 |
|---|---|
| `environment.py` | 외부 tensor 검증, 동시 행동, mode 전환, 제한된 observation |
| `generator.py` | 독립 train/eval task 생성, NPZ 저장·로드 |
| `models.py` | Encoder + GRU(128), single/dual actor 및 critic, checkpoint |
| `opponents.py` | Uniform Random, 순환 Fixed, frozen recurrent PPO |
| `qlearning.py` | rollout마다 초기화되는 ε-greedy tabular Q-learning |
| `rollout.py` | 고정 가중치로 rollout 수집 및 단계 전환 |
| `ppo.py` | 전체 sequence replay, MC return, head별 PPO/critic loss |
| `training.py` | opponent 사전학습 및 single/dual 학습, 안정성 기록 |
| `oracle.py` | Random/Fixed 상대의 정확한 유한 horizon 최적 기대 return |
| `evaluation.py` | 공통 비교 조건, 발견 시간, reward 곡선, oracle regret |
| `cli.py` | task 준비 → 사전학습 → 본 학습 → 평가 |

## 환경과 generator

Action `A/B/C/D`를 정수 `0/1/2/3`으로 표시합니다. 각 agent는 독립적인 reward tensor를 가지며, 모든 tensor의 action 축 순서는 전역 `(A agent, B agent)` 순서입니다. 아래는 generator 없이 외부 tensor를 사용하는 예시입니다.

```python
import numpy as np
from marl.environment import Task, Game

# 각 agent의 R1은 (4,4), R2는 (4,4,4,4).
task = Task(
    r1=np.stack([R1_A, R1_B]),
    r2=np.stack([R2_A, R2_B]),
)
game = Game(task, horizon=100)
observations = game.reset()
observations, rewards, done = game.step(0, 2)
```

기본 모드에서 R1 reward를 계산한 뒤 history-dependent 모드로 이동합니다. 다음 round에서 방금 전 R1 행동과 현재 행동으로 R2를 계산한 뒤 기본 모드로 복귀합니다. 다음 구간의 R1은 이전 행동을 reward 계산에 사용하지 않습니다. 과거 action은 observation 이력에는 남지만 다음 combo 판정에는 누적되지 않습니다.

Observation은 `직전 자신의 action, 직전 상대 action, 자신의 직전 reward, 현재 mode`입니다. Action은 sentinel 포함 각 5차원 one-hot, mode는 2차원 one-hot, reward는 scalar로 총 13차원입니다. 첫 observation은 이전 action sentinel `4`, reward `0`, 기본 mode입니다. Time, K, tensor, combo 위치 및 task ID는 모델 입력에 없습니다. 두 agent의 행동을 모두 결정한 뒤 환경을 진행합니다.

Generator는 agent별로 안전 action `s`와 hidden combo `(u,v)`를 고릅니다. `u != s`이며 `v`는 임의 action입니다.

| 조건 | 기본 보상 |
|---|---:|
| R1의 안전 action | 0.6 |
| R1의 hidden prefix | -3.0 |
| R1의 나머지 action | 0 |
| R2의 hidden combo 완성 | 9.0 |
| R2에서 combo 미완성, 안전 action | 0.6 |
| R2의 나머지 action | 0 |

Agent별 task scale `Uniform[0.8,1.2]`와 tensor 각 항의 작은 상호작용 `Uniform[-0.05,0.05]`를 적용합니다. Hidden combo가 우선 적용되며 상대 행동에 관계없이 발동합니다. 설정 검증은 최악의 상호작용에서도 hidden pair의 총보상이 안전 pair보다 높음을 보장합니다. 기본값에서 두 round의 안전 reward는 약 `1.2`, hidden reward는 약 `6.0`이며 prefix 비용은 큽니다. 이 값은 `config.generator`에서 변경할 수 있습니다.

Train/eval은 별도 RNG stream으로 생성하고 tensor fingerprint 중복을 검사합니다. Task 내에서는 모든 rollout에 같은 tensor를 사용합니다. `Task.combos`는 평가용 metadata이며 관측으로 전달하지 않습니다. 외부 tensor에 metadata가 없으면 발견 관련 지표는 `null`입니다. 외부 task 묶음은 `save_tasks`/`load_tasks`로 보관할 수 있습니다.

## 학습

Dual-policy는 shared encoder/GRU와 별도의 두 actor/critic을 사용합니다. `t < K`에는 exploration head, 이후에는 target head를 사용합니다. Rollout 시작 시 hidden/environment state를 초기화하고, 단계 전환 시에는 유지합니다. **K=25에서는 25번째 R1과 26번째 R2가 단계 경계를 가로질러 연결**되며 26번째 reward는 exploitation에 포함됩니다.

`γ=1`, GAE 없음, entropy coefficient `0`입니다. 0부터 시작하는 step index 기준:

```text
G_exploit = sum(reward[K:T])
exploration return target[t < K] = G_exploit
target return target[t >= K] = sum(reward[t:T])
```

Exploration 구간 reward는 observation과 전체 reward 평가에는 사용하지만 dual-policy return target에는 직접 포함하지 않습니다. 각 head의 advantage는 MC return에서 rollout 시점 critic 값을 뺀 뒤, 해당 head의 batch 내에서 정규화합니다. Head별 clipped actor loss와 critic MSE를 합칩니다. Critic coefficient `0.5`, clip `0.2`, Adam 학습률 `3e-4`, PPO epoch `4`, sequence minibatch `16`, gradient norm cap `0.5`가 기본값입니다.

Minibatch는 독립 rollout 단위로 섞습니다. 매 PPO 업데이트에서 현재 가중치로 첫 step부터 hidden state를 재계산하고, exploration/exploitation 경계에서 detach하지 않습니다. Exploitation loss가 exploration 구간의 GRU 연산까지 역전파됩니다. 수집한 trajectory는 해당 업데이트에서만 사용하며 오래된 rollout replay buffer는 없습니다.

Single-policy는 같은 encoder/GRU와 하나의 actor/critic으로 전체 T reward를 최적화합니다. K와 무관하므로 seed당 한 번 학습합니다. PPO opponent 역시 단일 recurrent PPO로 전체 reward를 최적화하지만, 사전학습 상대는 Random/Fixed 균등 혼합입니다. 본 학습에서는 Random/Fixed/저장된 PPO를 rollout마다 균등 선택합니다. 저장된 opponent의 가중치는 고정하며 hidden state만 갱신합니다.

Policy entropy, value/policy loss, approximate KL, clipping 비율, gradient norm을 기록합니다. Entropy `<0.1`은 issues log에 기록하며 알고리즘을 자동 변경하지 않습니다. 비유한 loss/gradient는 오류를 기록하고 중단합니다. 가중치 고정과 recurrent 구조만으로 실제 적응 능력을 보장하지는 않습니다.

## Online Q-learning

State는 `(mode, 직전 자신의 action, 직전 상대 action)`이며 Q-table shape는 `(2,5,5,4)`입니다. 모든 rollout은 Q=0으로 시작하며 task 간/rollout 간 Q를 공유하지 않습니다. 현재 action에 따른 reward를 받은 직후 다음 update를 수행합니다.

```text
Q(s,a) += alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))
```

종료 step에서는 bootstrap을 0으로 처리합니다. 기본 `alpha=0.2`, 고정 `epsilon=0.2`, `gamma=1`이고 동률 action은 무작위 선택합니다. 초기 optimistic reward 정보나 task 사전학습은 없습니다.

요청한 state에는 남은 시간이 없고, recurrent opponent의 내부 상태도 관찰되지 않습니다. 따라서 이 baseline에 유한 horizon 최적 수렴을 보장하지 않습니다. GRU 모델이 다수 task에서 사전학습한 경험과 Q-learning의 새 task 내 경험은 구분해서 해석해야 합니다.

## 평가와 결과

각 task·opponent 조합에서 8개 독립 rollout을 실행합니다. Agent별 환경과 hidden/Q state는 독립적입니다. 비교의 변동을 줄이기 위해 대응되는 평가에는 동일한 seed를 사용하되 agent/opponent RNG stream은 분리합니다. PPO opponent는 agent별 이력에 반응하므로 같은 seed에서도 행동이 달라질 수 있습니다.

각 K에서 다음 5개 조건을 평가합니다.

| 이름 | 처음 K rounds | 이후 |
|---|---|---|
| `dual_learned` | 학습된 exploration head | 같은 checkpoint의 target head |
| `dual_random` | 균등 random action, GRU 상태는 계속 갱신 | 같은 checkpoint의 target head |
| `dual_target` | target head | 같은 checkpoint의 target head |
| `single_gru` | 별도로 학습한 단일 policy | 동일 policy |
| `q_learning` | online Q-learning | 동일 online Q-learning |

Dual 3조건은 **같은 K checkpoint**를 사용합니다. Single/Q에는 K 기반 행동 전환이 없으며 같은 trajectory를 K별 cutoff로 평가합니다. Dual의 앞선 이력을 교체하는 실험은 학습 시와 다른 입력 이력을 만들 수 있습니다.

- 주요 K 비교 지표: 평균 exploitation reward. 전체 cumulative reward와 exploitation return도 저장합니다.
- Adaptation AUC: `mean_j(cumsum(reward[K:])[j] / (j+1))`. 구간 길이로 정규화하며 exploitation 초반 성능에 더 높은 가중치를 줍니다.
- Hidden combo 발견 시간: 처음으로 실제 R2 combo를 완성한 **1-based round 번호**. 이는 올바른 구조를 추론했음을 뜻하지는 않습니다.
- 미발견은 `first_combo_round=null`. 발견률, K까지 발견률, 발견한 경우만의 평균 시간, T에서 우측 검열한 제한 평균 시간을 함께 기록합니다.
- Oracle regret: 실제 K 종료 상태에서 얻을 수 있는 최적 **기대** exploitation return에서 실제 return을 뺍니다. Random/Fixed에 대해서만 계산하고 PPO에는 `null`을 기록합니다. Random의 개별 rollout에서는 표본 운에 의해 음수가 나올 수 있으므로 평균으로 해석합니다. 탐색 비용 전체나 다른 K 시작 상태의 손실을 포함하지 않습니다.
- 상대 reward와 자신의 reward 차이는 보조 지표입니다. 학습은 각자의 reward만 최대화합니다.

출력 파일:

```text
config.json, manifest.json             실제 설정, 버전, train/eval tensor fingerprint
training_tasks.npz, evaluation_tasks.npz
opponent.pt, single.pt, dual_k*.pt      저장 후 별도로 로드 가능한 모델
<model>.jsonl                         업데이트별 학습 지표
<model>_issues.jsonl                  안정성 문제 기록
evaluation/episodes.jsonl             task/상대/반복/K/조건별 원시 지표
evaluation/summary.json               각 조건의 평균 지표
evaluation/curves.csv                 round별 reward·누적 reward·평균 reward·상대 대비 차이
```

최종 비교에서는 task별 반복을 먼저 평균하고 training seed 간 변동도 보고해야 합니다. 요약 평균만으로 통계적 우위를 주장하지 않습니다.

## 검증 범위

단위 테스트는 tensor indexing, 두 round 경계, 정보 제한, hidden combo의 상대 독립성, split 분리, MC return, Q update, oracle과 완전탐색의 일치, rollout 중 가중치 고정, recurrent replay 일치, phase 경계를 넘는 gradient, 두 head 업데이트, checkpoint roundtrip, 지표 계산을 확인합니다. Smoke pipeline은 opponent 사전학습부터 세 종류 agent 및 다섯 평가 조건을 끝까지 실행합니다.

## Lambda / off-policy 추가 비교

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m scripts.run_comparison --smoke --output runs/comparison_smoke_new
.venv/bin/python -m scripts.run_comparison --output runs/comparison_new
.venv/bin/python -m scripts.summarize_comparison runs/comparison_new
```

`run_comparison`은 기존 `runs/pilot_seed0`의 task split과 frozen opponent/single을 복사합니다. 6조건(`ppo/vtrace × lambda=0/0.5/1`) 각각 K=10/25/50을 동일 initialization seed로 새로 학습합니다. 기존 pilot config를 사용하므로 현재 100 updates, seed=0입니다. 기존 결과는 덮어쓰지 않습니다. 출력 디렉터리는 비어 있어야 하며 optimizer resume는 지원하지 않습니다.

Exploration head의 target은 `lambda * sum(reward[t:K]) + sum(reward[K:T])`입니다. λ=0은 기존 objective, λ=1은 전체 reward-to-go입니다. λ는 exploitation head target을 변경하지 않습니다.

`vtrace` 조건은 exploration head에 fresh rollout PPO를, target head에 full-sequence replay V-trace actor-critic을 적용합니다. 탐색 행동도 target 학습에 포함하며 behavior log probability로 importance correction을 계산합니다. 두 head는 encoder/GRU를 공유합니다. Replay는 rollout 1,024개 FIFO, 균등 샘플링, update당 16 minibatches × 16 sequences이며 성공 trajectory 우선 샘플링은 없습니다. Hidden state는 현재 가중치로 episode 시작부터 재계산합니다. Gamma=1, terminal bootstrap=0, rho/c clipping=1, entropy bonus=0입니다. Lambda는 exploration PPO에만 적용되며 target은 모든 시점에서 원래 reward-to-go를 학습합니다. V-trace 기반 방식이지 PPO loss에 stale replay를 그대로 넣는 방식은 아닙니다. Reference: https://proceedings.mlr.press/v80/espeholt18a.html

두 조건의 환경 interaction 수는 같지만, PPO는 joint step 16회, V-trace는 exploration PPO 16회 + target replay 16회입니다. 따라서 알고리즘과 학습량 차이가 함께 포함됩니다. 독립 episode RNG와 배치 추론을 사용하므로 원래 pilot의 RNG 순서와는 다릅니다. Acquisition RNG를 optimization/replay RNG와 분리해 6조건 사이 task/opponent 선택을 맞춥니다.

평가는 기존 seed/task/opponent/rollout 규약을 유지하며 Single/Q 원시 결과를 공통 baseline으로 재사용합니다. 새로운 dual 결과에 콤보 발견 후 exploitation 구간의 재활용 기회·prefix·완성 횟수와 안전 행동률을 추가합니다. `comparison.csv`, `report.md`, `contrasts.json`은 task 단위 paired bootstrap을 포함합니다. Bootstrap은 학습 seed 변동을 추정하지 않으며 다중비교 보정도 없습니다. 본 결과는 단일 seed 파일럿입니다.

이번 실행의 최종 결과 경로는 `runs/lambda_offpolicy_seed0_corrected`입니다. 첫 실행의 V-trace 결과는 inactive head에 Adam momentum-only update가 적용되는 문제가 발견되어 폐기하고 재학습했습니다. 수정된 구현은 해당 head의 gradient를 `None`으로 설정해 optimizer step에서 제외합니다. Shared encoder/GRU는 양쪽 loss로 계속 학습합니다. PPO 결과는 이 수정의 영향을 받지 않아 재사용했습니다. `scripts/rerun_offpolicy.py`는 이 재실행의 provenance를 보존합니다.

## 학습 진행에 따른 발견 시점 / 행동 추적

본 학습을 tmux에서 직접 실행:

```bash
cd /home/kevinlees1220/marl_strategic_exploration
tmux new -s marl-track
set -o pipefail
.venv/bin/python -u -m scripts.run_tracked \
  --output runs/tracking_seed0 \
  --algorithms ppo vtrace --lambdas 0 0.5 1 --budgets 10 25 50 \
  --updates 100 --track-every 5 --diagnostic-tasks 16 --diagnostic-rollouts 2 \
  2>&1 | tee runs/tracking_seed0.log
```

`Ctrl-b`, `d`로 detach, `tmux attach -t marl-track`으로 복귀합니다. 최초 실행이며 기존 output 디렉터리는 덮어쓰지 않습니다. 다른 실행은 output과 log 이름을 바꾸세요. 본 runner는 CPU 단일 thread로 모델을 순차 학습합니다. 기본 source는 `runs/pilot_seed0`; 학습 seed, 기존 task split, frozen opponent/single을 재사용합니다. 학습 모델은 매번 새로 초기화합니다. 필요하면 `--algorithms vtrace --budgets 50`으로 범위를 줄일 수 있습니다.

학습 update와 episode round를 구분해 저장합니다.

- `learning_curves.csv`: 같은 진단 task/seed에서 update별 전체 return, exploitation 평균 보상, K까지 발견률, 미발견을 T에서 검열한 발견 시간, 발견 후 재활용률. 인접 checkpoint의 return 기울기와 발견 시간 변화도 같은 행에 있습니다.
- `discovery_curves.csv`: update마다 round별 발견 누적 비율, 그 round 직전 미발견 episode 수, 신규 발견 수.
- `event_aligned_rewards.csv`: 첫 발견을 lag=0으로 정렬한 두 라운드 pair 보상과 콤보 완성률, lag=-5..10. lag=0의 보상 급등은 정의상 생기므로 이후 lag에서 지속되는 개선을 봐야 합니다.
- `tracking_report.md`: 초기/마지막 발견 지표와 가장 큰 인접-checkpoint return 기울기 구간. 통계적 changepoint 판정이 아니라 기술적 요약입니다.
- `CASE/tracking/dual_kK/update_NNNNNN/*.steps.jsonl.gz`: round별 action, opponent action, explore/target/실제 behavior 확률, head, phase, reward, 누적 reward, 최근 10-round 평균, 첫 발견 flag. metadata는 evaluator만 사용하며 모델에 전달하지 않습니다.
- 같은 디렉터리의 `*.episodes.jsonl`: 첫 발견 시점, 첫 반복 시점/지연, 남은 horizon, 발견 이후 재활용 기회/횟수. 실패는 null/censored로 유지합니다.
- `CASE/checkpoints/dual_kK/update_NNNNNN.pt`: update 0 및 모든 진단 checkpoint를 따로 보관합니다. 기존 checkpoint를 덮어쓰지 않습니다. Optimizer resume는 지원하지 않습니다.

진단 task는 독립 RNG stream으로 생성하고 train/final-eval fingerprint와 중복을 검사합니다. 학습용 task·rollout RNG와 진단 RNG는 분리합니다. 진단은 update 0, `--track-every` 간격, 마지막 update에서 수행합니다. 더 정확한 상승 시점을 보려면 `--track-every 1`을 쓰되 평가/저장 비용이 늘어납니다. 고정 진단 집합은 추세 관찰용이며 최종 held-out 평가를 대체하지 않습니다. After-discovery 분석은 성공 episode와 남은 horizon에 조건부이므로 보상 변화의 인과성을 증명하지 않습니다.

진행 중 집계를 다시 만들려면:

```bash
.venv/bin/python -m scripts.summarize_tracking runs/tracking_seed0
```

작은 실행 경로 검증:

```bash
.venv/bin/python -m scripts.run_tracked --output runs/tracking_smoke_new \
  --algorithms vtrace --lambdas 0.5 --budgets 50 --smoke
```
