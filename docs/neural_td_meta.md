# Neural off-policy TD meta-exploration

별도 구현이며 기존 PPO / Tabular Q 코드는 변경하지 않는다. Q exploiter는 11차원 관측 → MLP → 3 Q-values 구조다. **GRU, context ingestion, history encoder가 없다.** 적응은 Adam의 TD parameter update로만 일어난다. 현재 observation에 포함된 마지막 action/reward는 그대로 허용한다.

## 모델 / objective

- `neural_td.QNetwork`: hidden64 기본, 2-layer ReLU MLP. 마지막 reward observation만 고정 scale10으로 나눈다. TD target/reward/objective는 원래 단위다. Hidden world type/bonus/oracle/remaining horizon은 input에 없다.
- `NeuralTD`: Double DQN, Huber loss, uniform replay with replacement, 기본 gamma=.99, Adam=.001, batch32, target Polyak tau=.05. `target_interval>0`이면 hard sync를 대신 사용한다. gamma=1 ablation 가능.
- `MetaExplorer`: **feedforward** actor/critic. 관측11 + Q-values3 + action counts3 + ordered pair counts9 + mode-conditioned reward means12/std12 + 최근 observed TD error1 = 51 features. 모두 관측에서 계산된다. Counts는 log1p, Q/reward statistics는 fixed /10 scale. TD error는 input일 뿐 intrinsic reward가 아니다.
- Outer objective는 `.1 * sum(B reward) + sum(K reward)`. 완료된 suffix의 실제 return을 포함한 **MC reward-to-go**를 prefix actor에 할당한다. 이 새 method는 GAE가 아닌 MC credit(=gamma1/lambda1)을 사용한다. Legacy config의 `gae_lambda`는 기존 PPO reference에 적용되며 새 outer target에는 적용되지 않는다.
- 같은 explorer policy version의 여러 independent worlds를 수집하고, 기존 `ppo_batch_update`로 **batch 전체 advantage normalization을 한 번만** 수행한다. TD를 미분하지 않는다. Recorded feature vectors는 rollout 당시 Q/statistics이며 PPO epoch에서 새로운 learner state로 재계산하지 않는다.
- `loo_baseline=true`, `same_world_rollouts>=2`는 optional ablation. 동일 world의 독립 sibling rollout objective 평균을 actor baseline으로 사용한다. Critic MC target은 그대로다. 모든 sibling B+K interactions가 episode/compute 비용에 포함된다. Batch/episode 수가 group size로 나누어져야 한다.

## Reset / replay / terminal

매 main-training episode 및 test world마다 동일한 pretrained theta0를 복사한다. Online Q/target Q, Adam momentum, replay, replay RNG, epsilon counter, statistics가 모두 새로 시작한다. Replay ownership ID는 cross-world insertion 검사에만 쓰며 network input에 들어가지 않는다. Meta-explorer weights는 train worlds 전체에서 공유되고 test에서는 고정된다.

R1→R2, R2→R1, B/K boundary는 terminal이 아니다. 실제 B+K horizon만 terminal이다. Pretraining budget의 마지막 partial segment는 잘라 수집해도 terminal을 인위적으로 만들지 않는다.

- `prefix_timing=online`: B의 각 transition을 저장한 뒤 `prefix_updates`회 update.
- `prefix_timing=batch`: 동일한 B transitions를 저장한 뒤 exploitation 직전에 동일 총 update 수를 수행.
- 기본 learning_starts=1이므로 두 방식은 정확히 B×prefix_updates optimizer steps와 동일 sampled-transition budget을 쓴다. 데이터가 이용 가능해지는 시점과 replay 분포는 다르다.
- `frozen_k`: B adaptation은 허용하고 K에서는 weights/replay/optimizer를 고정한다. Epsilon schedule과 current observation은 계속 진행한다.
- `online_k`: K transition마다 관측/replay/TD update를 계속한다.

기존 Single/Dual의 frozen은 B/K 모두 parameter freeze다. 기존 Tabular Q는 frozen 이름의 평가에서도 update한다. 비교 row의 `legacy_semantics`에 이 차이를 기록한다. 동일 의미의 frozen-weight 조건이라고 부르지 않는다.

## Theta0 pretraining 및 비용

Train worlds에서 coverage/random/repeat 관측만 수집한다. Candidate selection, hidden-label curriculum은 없다. Pretraining에서는 Q weights를 train worlds 사이에서 누적하고 **world마다 replay/target/optimizer를 새로 생성**한다. 초기 pretraining부터 world labels를 input으로 넣지 않는다. Pretraining이 끝나면 theta0를 고정하고 이후 outer training 내에서 수정하지 않는다. Joint initialization/explorer training은 구현하지 않았다.

Main config의 pretraining interactions=96,000, outer interactions=1,200×160=192,000이다. 기존 corrected protocol의 interaction budget에 맞췄지만 **TD optimizer compute는 PPO와 같지 않다**. Default outer episode마다 prefix128 + suffix128 =256 TD steps가 있고, each step에서 32 samples를 재사용한다. 실제 optimizer steps, sampled transitions/replay ratio, GPU event time, runtime을 별도로 기록한다. Random/Coverage/Repeat는 common theta0 pretraining 비용만 필요하며 Meta는 추가 outer training 비용이 든다. Existing Explorer의 기존 학습 비용도 별도 reference provenance로 남긴다.

World seed/split/opponents/noise는 기존 `world_set`과 `make_game`을 그대로 사용한다. 여러 explorer를 비교할 때 **하나의 prefix씩 실제로 실행**한다. 같은 initial environment/noise/opponent streams를 사용하지만 서로 다른 action에 따른 post-B state 차이는 정상적인 end-to-end exploration 효과다. 기존 Explorer를 사용한 TD와 기존 Dual 방법은 같은 prefix 행동을 재현한다. 다른 history를 하나의 anchor로 옮기는 counterfactual intervention 실험이 아니다.

## State aliasing / 진단 한계

Q input은 기존 current observation뿐이며 true reward world, remaining time, explicit cyclic phase는 없다. 따라서 hidden world는 theta adaptation으로 추정해야 하고, finite-horizon termination이나 history-dependent opponent에서는 aliasing이 남을 수 있다. 현재 random/constant/fixed cyclic opponent만 지원한다. Reward input이 불필요한 상태 구분을 만들어 함수 근사 generalization에 영향을 줄 가능성도 있다. 관측 가능한 step counter/phase augmentation은 후속 ablation이며 main에 조용히 추가하지 않았다.

`q_sampled_return_rmse`는 action 시점의 Q와 해당 suffix의 실제 discounted behavior return 차이다. 단일 noisy trajectory, epsilon behavior, online learning, finite horizon을 포함한다. **Stationary optimal Q의 정확도나 true Q* error가 아니다.** TD loss/error와 분리해 해석한다.

## 실행 순서

Repository root의 기존 CUDA PyTorch 환경에서 실행한다. CPU 개발환경에서는 `--device cpu`를 사용한다. 기존 checkpoint/runs를 덮어쓰지 않도록 fresh output을 요구한다.

```bash
python -m pytest tests/meta/test_neural_td.py -q
```

Deterministic mechanism 확인:

```bash
# 온라인 B36 / 2304 TD steps: 실제 로컬 실험에서는 일부 case가 실패했다.
python -m meta_exploration.neural_td_experiment gate \
  --config configs/meta/neural_td_pilot.json \
  --out runs/neural_td_meta_v1/server_gate_online --device cuda:0

# 같은 B36/2304 steps를 B 종료 후 수행: 로컬 6 cases 통과.
python -m meta_exploration.neural_td_experiment gate \
  --config configs/meta/neural_td_gate_batch.json \
  --out runs/neural_td_meta_v1/server_gate_batch --device cuda:0
```

Gate의 update budget은 실험 기본값보다 크다. Gate 통과는 neural TD가 관측 데이터로 combo를 실행할 수 있다는 mechanism 검증이며 B32/default-compute sample efficiency 증명이 아니다. 실패한 gate는 nonzero exit하고 결과를 남긴다.

작은 A2/A3 pilot, 기존 full-study checkpoint를 reference로 사용:

```bash
python -m meta_exploration.neural_td_launch \
  --config configs/meta/neural_td_pilot.json \
  --out runs/neural_td_meta_v1/server_pilot \
  --regimes A2 A3 --seeds 0 1 \
  --existing-root runs/full_study_corrected_v1 --require-gpu

python -m meta_exploration.neural_td_experiment summarize \
  --inputs runs/neural_td_meta_v1/server_pilot \
  --out runs/neural_td_meta_v1/server_pilot_summary
```

Launcher는 arithmetic probe를 통과한 GPU마다 독립 worker를 하나 두고 완료한 worker에 다음 job을 배정한다. GPU reset/타인 process 종료는 하지 않는다. 실패는 `jobs.json`/개별 log에 남긴다. 자동으로 full-study를 시작하지 않는다. CPU에서는 `--require-gpu`를 빼고 `--cpu-workers`를 지정한다.

개별 job:

```bash
python -m meta_exploration.neural_td_experiment run \
  --config configs/meta/neural_td_pilot.json --regime A3 --seed 0 \
  --out runs/neural_td_meta_v1/custom_A3_seed0 \
  --existing-root runs/full_study_corrected_v1 --device cuda:0
```

Reference checkpoint가 없으면 Q-only와 4개 TD methods를 평가하고 unavailable methods를 provenance에 명시한다. `--existing-root`를 주었는데 파일이 없으면 실패하며 random model로 대체하지 않는다. Existing references는 자신의 checkpoint architecture로 load하며 그 training config를 저장한다. Pilot의 TD/Meta training budget과 historical PPO reference의 training budget이 다름을 결과 해석에 반영해야 한다.

`--resume`는 완료된 rollout batch checkpoint에서만 지원한다. 중간에 끊긴 batch/pretraining logs는 보존하고 재실행을 거절하여 같은 interactions를 조용히 재사용하거나 비용에서 누락하지 않는다. Partial evaluation은 별도 `evaluate --source <run> --out <fresh path>`로 실행한다. 새 config로 이전 checkpoint를 이어 학습하지 않는다.

## Main study 준비 — 지금 자동 실행하지 않음

Pilot의 안정성/compute/online-B 실패 원인을 확인한 뒤에만 아래 명령을 사용한다. 이 문서에 명령을 제공하는 것은 결과 검증을 대체하지 않는다.

```bash
python -m meta_exploration.neural_td_launch \
  --config configs/meta/neural_td.json \
  --out runs/neural_td_meta_v1/main \
  --regimes A1 A2 A3 B --seeds 0 1 2 3 4 \
  --existing-root runs/full_study_corrected_v1 --require-gpu
```

`evaluation_timings=["online","batch"]`이면 같은 online-trained meta-explorer를 timing ablation으로 평가한다. Batch-aware meta-training은 `prefix_timing="batch"`인 **별도 train run**을 만들고 비교해야 한다. TD gamma=1, LOO, frozen-K outer objective도 각각 별도 config/run이다.

## 결과

Run별 `checkpoints/{td_initial.pt,neural_td.pt,meta_resume.pt}`, `pretraining.jsonl`, `training.jsonl`, `costs.json`, `provenance.json`, `worlds_diagnostic.json`, `evaluation/metrics.json`을 저장한다. Trajectory logging을 켜면 public JSONL과 TD-update JSONL도 저장한다. Hidden diagnostic 파일은 agent 입력으로 읽지 않는다.

Summary는 timing별 aggregated CSV/JSON, meta-minus-baseline paired hierarchical bootstrap CI, evidence groups, adaptation curves/AUC/reach/RMST, B-budget reward/combo plots, training curves, TD/replay/compute diagnostics를 생성한다. Bootstrap의 바깥 단위는 training seed, 안쪽 단위는 paired held-out world다. 조건부 evidence groups는 선택 편향이 가능한 기술 통계이며 인과적 method 우위로 해석하지 않는다.

[실제 CPU 검증/pilot 보고서](../examples/neural_td_meta_v1/REPORT.md)를 참조한다. Full-study 결과는 아직 없다.
