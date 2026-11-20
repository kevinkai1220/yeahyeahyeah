# History utilization / PPO optimization diagnostics

이 도구는 기존 corrected full-study checkpoint를 **읽기만** 한다. 기존 runs와 checkpoint에 쓰지 않으며 새 output 디렉터리만 허용한다. 로컬 CPU pilot과 서버 full-study 모델의 결론을 구분해야 한다.

## Phase 1: 코드에서 확인한 경계와 상태

- `Controller`는 episode별 `hidden=None`으로 시작한다. `ingest(H)`는 **exploiter 자신의 encoder/GRU**로 `H`의 action-selection 직전 observation vectors를 replay한다. Explorer hidden을 직접 넘기지 않는다.
- `H[-1].reward`는 ingest에 포함되지 않고 실제 post-B observation에서 처음 소비된다. Reset intervention도 이 현재 observation은 유지한다. 즉 reset은 현재 정보까지 지우는 조건이 아니다.
- Actor와 critic은 encoder 및 GRU를 공유하고, linear heads는 독립적이다. Single, explorer, dual exploiter는 각각 별도 model/optimizer다.
- Frozen은 weights 고정, history 업데이트만 수행한다. Online은 실제로 수집한 segment만 PPO에 사용하고 비terminal boundary에서 bootstrap한다. 업데이트 후 전체 context를 새 weights로 재생한다. 모델 복사와 optimizer 초기화로 world 간 adaptation leakage를 막는다.
- R1→R2→R1은 termination이 아니다. Horizon만 terminal이다. 홀수 B에서도 R2를 그대로 넘긴다.
- Training observed-suffix credit은 완료된 suffix return을 explorer boundary bootstrap으로 사용한다. Test에서는 미래 suffix를 사용하지 않는다.
- `Game.snapshot()`은 mode, previous actions, rewards, opponent hidden 및 양쪽 RNG state를 deep-copy한다. Frozen opponent model만 공유한다.

## 서버에서 먼저 실행할 명령

Repository root, 프로젝트의 기존 Python/torch 환경에서 실행한다. 아래 첫 evaluation은 **2 seeds × 6 worlds × A2/A3**의 확인용이다. `cuda:0`은 서버에서 사용 가능한 장치로 바꾼다. 다른 사용자의 작업을 종료하거나 reset하지 않는다.

```bash
nvidia-smi
python -m pytest tests/meta/test_credit_speed.py tests/meta/test_history_optimization.py -q
python -m meta_exploration.history_diagnostics \
  --checkpoint-root runs/full_study_corrected_v1 \
  --out runs/history_optimization_diagnostics_v1/checkpoint_pilot \
  --regimes A2 A3 --seeds 0 1 --worlds 6 \
  --modes frozen online --device cuda:0
```

Checkpoint는 `<root>/A2/seed_0/checkpoints/trained.pt` 및 같은 run의 `config.json` 구조를 사용한다. 없으면 명시적으로 실패하며 random weights로 대체하지 않는다. 기존 GPU 서버의 파일은 GitHub에 올릴 필요가 없다.

검증 후 전체 checkpoint evaluation을 사용자가 명시적으로 실행하려면:

```bash
python -m meta_exploration.history_diagnostics \
  --checkpoint-root runs/full_study_corrected_v1 \
  --out runs/history_optimization_diagnostics_v1/checkpoint_evaluation \
  --regimes A2 A3 --seeds 0 1 2 3 4 --worlds 100 \
  --modes frozen online --device cuda:0
```

이 명령은 재학습이 아니다. Small network의 scalar inference/CPU environment 때문에 GPU 사용률이 낮을 수 있다. 독립 seed를 서로 다른 GPU에서 실행하려면 `--seeds`와 `--device`, `--out`을 모두 다르게 지정한다. 여러 프로세스가 같은 output에 쓰면 안 된다. 분리 실행 결과의 통합 집계 기능은 아래 `summarize` 명령을 사용한다.

```bash
python -m meta_exploration.history_diagnostics_summary \
  --inputs runs/history_optimization_diagnostics_v1/checkpoint_evaluation \
  --out runs/history_optimization_diagnostics_v1/checkpoint_summary
```

## Intervention 정의 및 한계

같은 learned prefix로 얻은 anchor post-B snapshot에서 모든 suffix를 시작한다. 각 조건의 policy/action RNG와 environment/opponent RNG를 동일하게 초기화하며 이후 action divergence는 허용한다.

- `real`: 실제 anchor exploration history.
- `reward_shuffled`: 초반 reward 순서만 교란, 마지막 두 transitions 유지. Token chain은 재구성하지만 reward law와 정합적이라는 보장은 없다.
- `action_shuffled`: 초반 joint action tokens의 순서를 섞고 token chain을 재구성한다. Reward/action의 관계는 깨질 수 있다.
- `reset`: recurrent memory를 비우고 실제 현재 observation만 받는다.
- `coverage`, `random`, `repeat`, `q_learning`: 동일 initial snapshot의 별도 sandbox에서 얻은 **실제 observed history**를 anchor에 붙인다.
- `successful_observed`: Coverage 후 관측상 가장 좋았던 pair를 반복하는 여러 history 중 repeated pair reward의 관측 lower-bound score가 가장 높은 것을 선택한다. Hidden 정답은 선택에 사용하지 않는다. Positive evidence가 없을 수도 있으며 `observed_repeated_positive_evidence`를 반드시 확인한다. 후보 수만큼 추가 interaction 비용을 기록한다.

대체 history는 자기 trajectory에서는 실제 관측이지만 anchor로 옮기는 순간 counterfactual memory intervention이다. 대체 history의 마지막 state와 anchor가 다를 수 있다. 이를 실제 행동으로 그 history를 수집했을 때의 end-to-end strategy 가치나 엄밀한 인과적 정보 효과로 해석하지 않는다. Q는 observed histories/reset만 replay하고 fabricated shuffled transitions는 받지 않는다.

Single도 여기서는 **같은 learned prefix를 받은 exploiter checkpoint**로 검사한다. Main Single의 자기 정책으로 B를 수집하는 성능 실험과 구분한다. Q는 항상 TD 업데이트하며 frozen neural의 직접적인 weight-freeze counterpart가 아니다.

Type 2 switching은 같은 actual handoff observation에서 memory 유무에 따른 distribution 차이와 good action argmax 여부다. Initial preferred action은 initial zero-history observation에서 구한다. Evidence curve는 fixed weights로 실제 historical states의 policy를 재평가한다. 두 비교는 서로 다른 의미다.

Type 3는 첫 action probability, 해당 action을 private snapshot에서 강제 수행한 뒤 두 번째 probability, 두 값의 product, 실제 suffix ordered-pair completion을 별도로 기록한다. Product는 **하나의 동일 opponent/noise RNG response에 조건부**인 값이며 unconditional expected pair probability가 아니다. 홀수 B의 R2 handoff는 첫 action/product를 undefined로 남긴다. Diagnostic forced action은 평가용 분기이며 agent rollout/training에 삽입하지 않는다.

## Probe

`probe_samples.json`은 GRU ingest 직후 state와 **그 state가 실제 읽은 observation tokens**로 계산한 61개 targets를 저장한다. 마지막 reward를 미리 target에 넣지 않는다. Counts, means, variances, unobserved masks는 R1 single actions와 R2 ordered pairs를 구분한다. Opponent counts, 마지막 읽은 mode/action one-hot도 포함한다. Hidden reward/true combo/world type은 feature/target으로 사용하지 않는다.

- `within_checkpoint_probes.json`: checkpoint/seed별 world-disjoint split. 학습 fold의 scaling과 mean baseline, fixed ridge penalty만 사용한다. 주 해석 대상이다.
- `probes.json`: leave-one-checkpoint-seed-out readout. GRU 좌표가 seed마다 달라질 수 있어 **실패가 정보 부재를 뜻하지 않는다**. 별도 보조 진단이다.

Probe decodability가 actor 활용을 입증하지 않으며, 적은 pilot worlds의 probe는 신뢰성 있는 연구 결론에 부족하다.

## Phase 3–4: 작은 새 학습 실험

기존 checkpoint를 이어 학습하지 않는다. 모든 variant의 shared pretraining weights 동일성을 assert한다. 기본은 단일 Type 3, constant opponent, sigma=0, 128 main episodes, B18/K32, hidden32, seed0이다. Full-study hidden64/B32/K128과 다르다.

```bash
python -m meta_exploration.optimization_pilot \
  --config configs/meta/history_optimization_pilot.json \
  --out runs/history_optimization_diagnostics_v1/minimal_gpu \
  --stages deterministic --variants baseline cadence1 value_coef \
  --seeds 0 --device cuda:0
```

성공/실패를 확인한 뒤에만 단계별로 `random_opponent`, `noisy`, `multi_constant`, `multi_mixed`를 선택한다. 필요한 경우 `--seeds 0 1 2`로 확장한다. 자동 full-study 실행은 없다.

- `baseline`: corrected batch8/4epochs.
- `cadence1`: 같은 rollout batch8의 advantage를 한 번 정규화하고 episode minibatch1로 4epochs. 각 episode hidden/context는 독립이며 old log-prob와 rollout version은 고정이다. Padding tokens는 loss에서 제외된다. Epoch 증가 실험이 아니다.
- `value_coef`: coefficient .5→.05만 변경.
- `value_scale`: original-unit value prediction/GAE를 유지하고 critic residual만 /10. MSE에서는 coefficient /100과 수학적으로 동등한 scale ablation이다. PopArt나 learned target normalization으로 부르지 않는다.
- `group_clip`: encoder+GRU, actor head, critic head를 각각 clip. Shared parameters는 한 번만 clip한다. Shared trunk에서 actor/critic gradient 기여를 분리하는 알고리즘은 아니다.

Gradient logs는 actor/critic shared-gradient norm/cosine, 각 parameter group의 clipping 전후 gradient norm, shared/head별 parameter delta, original-unit critic MSE와 target 분포를 포함한다. Adam 때문에 raw gradient 크기를 실제 update 억제 배수로 해석하면 안 된다. Global clipping 기본값과 기존 reward objective는 바뀌지 않는다.

`costs.json`은 minibatch를 반영한 실제 logged optimizer steps를 집계한다. Interaction/token pass/optimizer step/wall time을 분리해 비교한다. `learning.png/pdf`, raw `training.jsonl`, `pilot_evaluation.json`, `summary.json`, oracle diagnostic이 생성된다. Oracle controller는 evaluation diagnostic이며 sigma0 constant 조건에서 DP expected reward와 realized reward가 같음을 test한다.

## 산출물 / provenance

Checkpoint evaluation은 run별 JSONL에 raw suffix, evidence curve, actual-prefix metrics, intervention별 probability 및 compute를 저장한다. `provenance.json`에는 source checkpoint SHA256, original/effective config, git commit/status, source SHA256, Python/torch/device가 들어간다. `paired_summary.json`은 world-paired 차이를 seed→world hierarchical bootstrap한다. Threshold 미도달은 기존 speed module의 right-censoring/RMST 정의를 유지한다. World를 독립 training seed로 세지 않는다.

후속 GRU+TD 또는 observed-statistics policy, actor/critic trunk 분리는 앞선 checkpoint intervention 및 minimal learning 결과를 본 뒤 결정한다. 이번 변경에서는 architecture를 교체하지 않았다.
