# GRU, Q-table, opponent 영향 분석과 개선 계획

작성일: 2026-11-19

이 문서는 업로드된 full-study 결과와 현재 구현을 바탕으로 GRU의 적합성, 명시적인 Q update의 이점, opponent의 영향을 검토하고 후속 실험 순서를 정리한다. **분석과 제안 문서이며, 아래 개선 사항이 구현되거나 재실험까지 완료되었다는 의미는 아니다.**

## 1. 핵심 판단

- GRU는 세 world type에 공통으로 사용할 수 있는 타당한 recurrent backbone이다. 현재 성능만으로 표현 능력이 부족하다고 결론 내릴 수 없다.
- 현재 비교는 엄밀히는 **GRU+PPO 대 tabular TD**다. 메모리 표현과 학습 알고리즘의 효과가 동시에 달라진다.
- 작은 상태 공간에서는 Q-table의 명시적인 상태 표현과 TD backup이 강한 이점이다. 다만 명시적인 업데이트가 통계적으로 정확한 value를 보장하는 것은 아니다.
- 현재 실험에는 동시 학습하는 opponent가 없다. 상대 행동의 불확실성은 학습을 어렵게 만들 수 있지만, co-learning에 따른 비정상성은 이번 결과의 설명이 아니다.
- 우선순위는 **credit assignment 검증·수정 → updater/memory/opponent 분리 실험 → 관측 기반 메모리 개선 → backbone 변경**이다.

## 2. 결과의 범위와 해석 한계

주요 비교 조건은 B=32, K=128, online evaluation이며, 각 single-type regime은 5 seeds × 100 test worlds를 사용한다. 수치는 실제 결과 분석에서 가져왔다. 자세한 전체 결과는 [REPORT.md](REPORT.md)를 참고한다.

| 지표 | Type 2 Dual Neural | Type 2 Dual Q | Type 3 Dual Neural | Type 3 Dual Q |
|---|---:|---:|---:|---:|
| K 구간 step당 평균 reward | 1.651 | 2.715 | 0.502 | 2.005 |
| B에서 좋은 combo를 시도한 world에서 해당 combo의 K 구간 재사용률 | 6.7% | 71.1% | 8.7% | 32.1% |

두 dual method는 동일한 exploration history와 post-exploration 환경 snapshot에서 시작한다. 따라서 결과는 neural exploiter가 해당 history를 행동에 연결하는 과정에 병목이 있음을 시사한다.

단, 재사용률은 **실제로 좋은 combo를 B에서 시도한 world에 조건부인 지표**다. 전체 world의 optimal-combo 사용률과 같지 않다. 또한 이 차이만으로 GRU의 기억 실패, PPO 최적화 실패, pretraining 부족 중 하나를 특정할 수 없다.

업로드된 결과에는 checkpoint와 원시 step trajectory, training/runtime 로그가 포함되지 않아 hidden-state probe 및 학습 중 gradient 진단은 추가 실행이 필요하다.

## 3. GRU는 공통 모델로 적절한가?

### 3.1 사용 가능성과 학습 성공은 다르다

관측·행동·보상을 recurrent network에 전달하고 hidden state로 새로운 task에 적응하는 방식은 meta-RL의 기존 접근이다. RL²는 recurrent state에 빠른 적응 알고리즘을 학습하는 구도를 제시했다. Recurrent model-free RL의 성능이 architecture와 hyperparameter 선택에 크게 좌우된다는 실험적 근거도 있다. [RL²](https://arxiv.org/abs/1611.02779), [Ni et al., 2022](https://proceedings.mlr.press/v162/ni22a.html)

현재 world들은 동일한 action/observation 형식을 사용하므로 하나의 GRU backbone을 공통으로 사용하는 데 구조적인 모순은 없다. 다만 GRU에는 다음 능력이 동시에 요구된다.

1. 어떤 sequence를 시도했는지 기억한다.
2. 반복 관측한 noisy reward를 평균화한다.
3. 상대 행동 효과와 자신의 행동 효과를 구분한다.
4. 추정한 정보를 실제 strategy로 전환한다.

**Combo 길이가 2라고 필요한 기억도 2-step인 것은 아니다.** 수십 턴에 걸친 시도 횟수와 reward evidence를 누적해야 한다. Hidden dimension 64가 부족할 가능성은 있지만, 현재 증거로 이를 단정하거나 Transformer 교체를 우선할 수는 없다.

### 3.2 구분해야 할 실패 유형

| 실패 후보 | 확인할 진단 |
|---|---|
| History 통계를 저장하지 못함 | Hidden state로부터 관측 기반 통계를 예측하는 probe |
| History를 저장하지만 actor가 활용하지 못함 | Reward prediction 성능과 행동 성능을 함께 비교 |
| History에 둔감함 | 실제/빈/순서 변경 history를 입력하는 통제 평가 |
| PPO 학습 신호가 약하거나 잘못됨 | Advantage, actor gradient, KL, clip fraction, critic error 확인 |
| Task 다양성 또는 학습량 부족 | Train-world 성능과 held-out 성능, 학습량별 curve 비교 |

History 교란 실험은 sequence와 mode의 의미를 깨뜨릴 수 있으므로 단독으로 원인을 확정하지 않는다. 어떤 정보를 제거했는지 명시하고, 각 branch의 환경 상태와 관측 정합성을 관리해야 한다.

## 4. Q-table의 explicit update가 유리할 가능성

현재 Q-table 배열은 `(2, 4, 4, 3)`으로 96개 값이다. State는 mode와 이전 양쪽 action을 표현한다. 작은 문제에서는 이 수동 state representation과 아래 TD update가 강한 inductive bias다.

```text
Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') − Q(s, a)]
```

Episode termination에서는 bootstrap을 제거하지만, combo 이후 mode reset은 termination이 아니므로 bootstrap을 유지한다.

| 요소 | GRU+PPO | Tabular Q-learning |
|---|---|---|
| 경험 저장 | 학습된 recurrent state 갱신 | 명시적인 state-action 값 |
| 새 reward 반영 | Recurrent inference 및 PPO update | 해당 entry에 즉시 TD update |
| Delayed reward 전달 | Advantage와 gradient | Transition replay와 TD backup |
| 사전학습 부담 | 경험을 해석하는 규칙까지 학습 | 업데이트 규칙이 미리 지정됨 |

따라서 Q가 현재 환경에서 더 빠르게 적응하는 것은 충분히 가능한 결과다. 반면 Q에도 다음 한계가 있다.

- Noisy reward, 방문 부족, bootstrap 및 max 연산에 따른 오차가 있다.
- 동일 trajectory를 반복 replay해도 독립 관측 수가 늘어나지는 않는다.
- 고정 learning rate와 제한된 interaction budget에서 정확한 최적 Q값을 보장하지 않는다.
- Opponent가 더 긴 history나 관측 불가능한 내부 상태에 의존하면 현재 Q-state가 충분하지 않을 수 있다.
- 새 world마다 Q를 초기화하므로 neural policy처럼 world 간 parameter transfer를 수행하지 않는다.

결론적으로 현재 결과는 **이 환경에서 explicit TD가 효율적이었다**는 근거이지, Q-table이 GRU보다 일반적으로 정확하다는 증명은 아니다.

## 5. Opponent는 학습을 방해하는가?

### 5.1 현재 실험에 포함된 상대

현재 uploaded study는 uniform random과 고정 A→B→C 순환 opponent를 사용한다. 동시에 학습하는 opponent는 포함하지 않았다. 코드의 선택적 PPO opponent 지원과 실제 실험에 사용된 opponent를 혼동하면 안 된다.

Uniform random 상대에서는 현재 상대 action을 사전에 예측할 수 없다. World와 history, agent action을 조건으로 보면 reward 변동에는 Gaussian noise 외에도 opponent action에 따른 reward mean 변동이 포함된다.

```text
Var(R | world, H, a)
  = sigma² + Var_b[mu_world(H, a, b)]
```

이는 독립적인 reward noise와 주어진 조건의 opponent 분포를 가정한 분해다. Agent는 관측한 상대 action으로 사후 추정할 수 있지만, 행동 선택 전에 현재 상대 action을 입력받아서는 안 된다.

### 5.2 실제 subgroup 결과

| Type 3, online B=32 | Random opponent | 고정 순환 opponent |
|---|---:|---:|
| Single Policy | 0.478 | 0.521 |
| Q-learning | 2.187 | 2.309 |
| Dual Neural | 0.499 | 0.506 |
| Dual Q-table | 1.840 | 2.170 |

출처: [by_opponent.csv](by_opponent.csv). 값은 K 구간 step당 평균 reward다.

Dual Neural은 두 조건 모두 낮다. 상대의 무작위성만이 유일한 병목이라는 설명은 부족하다. **그러나 각 subgroup에는 서로 다른 reward world가 포함되므로 이 표는 causal ablation이 아니다.** 순환 상대의 예측 가능성은 oracle reward 자체도 바꿀 수 있다.

정확한 후속 비교는 동일 hidden world에서 opponent만 바꾸고, 다음 순서로 수행한다.

1. 한 action을 유지하는 constant opponent: 진단용 단순 조건.
2. Uniform random opponent.
3. 고정 순환 opponent.
4. Frozen history-dependent opponent: 추가 확장.
5. Co-learning opponent: 별도 연구 조건.

가능한 random streams를 통제하고 raw reward와 opponent별 oracle regret을 함께 보고한다. Frozen history-dependent opponent의 부분관측 문제와, 실제 parameter가 변하는 co-learning의 비정상성은 구분한다.

## 6. 우선 수정할 사항: episode 내부 advantage normalization

현재 [learning.py](../../meta_exploration/learning.py)는 `ppo_update`에 전달된 단일 trajectory/segment 안에서 advantage를 정규화한다.

```python
adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
```

Explorer의 B segment를 따로 업데이트할 때, 높은 K return이 B 전체 advantage를 공통으로 높이거나 같은 비율로 키우는 효과가 이 정규화로 제거될 수 있다. 이는 좋은 downstream 결과를 만든 episode와 나쁜 episode 사이의 차이를 약화시킨다.

실제 target 함수와 같은 normalization 식으로 수행한 반례는 다음과 같다. Prefix reward와 critic 값을 0으로 두고, B=32/K=128에서 suffix reward를 매 step 1 또는 10으로 설정했다.

| GAE lambda | K return | B raw advantage 평균 | 두 조건의 정규화 advantage 최대 차이 |
|---|---|---|---:|
| 1.0 | 128 vs 1280 | 128 vs 1280 | 0 |
| 0.95 | 128 vs 1280 | 9.561 vs 95.612 | 약 7.15e-7 |

GAE lambda=1에서는 두 경우 모두 B normalized advantage가 0이다. Main 값인 0.95에서도 두 normalized vector는 거의 같다. [진단 결과](advantage_audit.json), [재현 코드](advantage_audit.py)

**이 반례는 실제 학습의 모든 downstream 신호가 사라졌다는 증명은 아니다.** 실제 prefix reward와 critic 값은 0이 아니며, entropy/value loss의 gradient도 별도로 존재한다. 그러나 architecture를 평가하기 전에 검증해야 할 구조적인 문제다.

### 권장 수정과 회귀 검증

- 같은 policy version으로 여러 독립 world의 complete B+K rollout을 수집한다.
- 각 policy별로 여러 episode의 해당 segment를 합쳐 normalization한다.
- Explorer와 exploiter는 optimizer와 policy objective를 분리한다.
- Single은 B+K 전체, Dual explorer는 B, Dual exploiter는 K를 학습한다.
- `gamma=1.0`, phase별 reward weighting, 별도 GAE lambda를 유지한다.
- 높은 downstream return이 유리한 exploration 행동의 gradient에 반영되는지 regression test를 작성한다.
- Batch 크기 변경에 따른 PPO transition/update 수와 총 연산 비용을 기록한다.

Test online evaluation에서는 아직 관측하지 않은 미래 K reward를 사용할 수 없다. Truncated segment에는 value bootstrap을 사용하고, rollout별 policy version을 관리한다. Training의 complete-episode credit assignment와 test의 online update를 구분해야 한다.

## 7. 원인을 분리하는 최소 비교 실험

기존 네 main method는 유지하고 아래 항목은 명시적인 diagnostic ablation으로 추가한다.

| 비교 | 주요 질문 | 해석상의 주의점 |
|---|---|---|
| 수정된 GRU+PPO | 기존 학습 신호 문제가 성능을 제한했는가? | 기존 budget과 변경된 update 비용을 기록 |
| GRU+Q-head/TD | Recurrent memory를 유지해도 TD가 더 잘되는가? | On/off-policy와 replay 차이도 함께 기록 |
| 관측 통계+MLP/PPO | 통계를 직접 제공하면 PPO가 학습하는가? | 새로운 정보가 아니라 history에서 계산한 정보만 제공 |
| Tabular TD | 작은 state 공간의 explicit update 기준선 | World 간 transfer가 없는 baseline |

이 비교도 완벽한 단일요인 실험은 아니다. Architecture별 parameter 수, replay 횟수, optimizer step, interaction 수를 함께 기록하고 해석한다.

### 진단 순서와 판단 기준

1. **단순 조건 학습 검증:** 단일 world, constant opponent, sigma=0에서 학습 가능한지 확인한다. 이는 디버깅용이며 multi-world main pretraining을 대체하지 않는다.
2. **Multi-world Type 1 검증:** 관측 history를 사용해 random보다 개선되는지 확인한다.
3. **Combo와 noise 추가:** 같은 reward worlds에서 opponent와 sigma를 바꾸며 성능 저하 지점을 찾는다.
4. **동일 history의 downstream 비교:** Neural/Q branch의 snapshot, mode, opponent 상태, RNG 복원을 유지한다.

결과별 판단:

- GRU+TD 성공, GRU+PPO 실패: objective/update 방식이 주요 후보.
- 관측 통계+PPO 성공, GRU+PPO 실패: 통계 추출·저장 과정이 주요 후보.
- Constant opponent 성공, random 실패: opponent uncertainty 처리와 reward 평균화를 점검.
- Reward prediction은 좋지만 return은 낮음: 정보 추정보다 actor/control 학습을 점검.
- 단순 deterministic world부터 실패: 용량 확대보다 구현·최적화 점검을 우선.

어느 경우도 하나의 관측만으로 원인을 확정하지 않고 반복 seed와 paired comparison으로 확인한다.

## 8. 구조적 개선: 관측 기반 메모리와 task inference

### 8.1 Explicit statistics + recurrent context

GRU에 모든 통계 계산을 맡기는 대신 history에서 계산한 다음 정보를 보조 입력으로 사용할 수 있다.

- Mode와 joint action/sequence별 관측 횟수.
- 관측 reward 평균과 분산.
- 미관측 항목 mask.
- Opponent action 빈도 및 transition 빈도.

모든 값은 실제 history에서만 계산한다. True reward mean, hidden combo identity, world type은 policy 입력에 넣지 않는다. R1/R2 reward를 무분별하게 합치지 않고 mode와 sequence 조건을 구분한다. 전체 reward의 관측 평균을 그대로 true combo bonus라고 해석하지 않는다.

이 방식은 정보 접근 권한을 늘리지 않지만 representation과 inductive bias를 바꾼다. 따라서 별도 ablation으로 보고한다.

### 8.2 Auxiliary prediction

선택적 auxiliary head로 관측 reward 또는 다음 opponent action을 예측하게 할 수 있다.

- Reward target은 실제 관측한 noisy reward이며 hidden mean이 아니다.
- Reward prediction에서 기록된 현재 양쪽 action을 사용할 경우 이는 transition 관측 이후의 학습이다.
- Actor와 다음 opponent 예측기는 행동 선택 시점까지의 history만 사용한다.
- Auxiliary training은 optional flag로 분리하고 PPO-only main 결과와 섞지 않는다.

Task inference와 control을 분리하는 방향은 PEARL 같은 meta-RL 연구에서도 제시된다. 다만 PEARL의 성능이 현재 환경에서 자동으로 재현된다는 의미는 아니다. [PEARL](https://proceedings.mlr.press/v97/rakelly19a.html)

### 8.3 Backbone 변경은 이후에 수행

학습 신호와 단순 조건 검증 후 GRU 64/128, LSTM 등을 비교한다. Transformer는 더 긴 history나 명시적인 retrieval 요구가 확인될 때 추가한다. Architecture를 바꾼 실험에서도 해당 architecture의 Single과 Dual exploiter는 동일한 shared pretrained checkpoint에서 시작해야 한다.

## 9. 유지해야 할 연구 조건과 보고 항목

- Main objective: `J = lambda * exploration_return + exploitation_return`.
- Main neural gamma=1.0; GAE lambda와 Q gamma는 별도로 기록.
- Mode reset은 episode termination이 아님.
- Train/test hidden parameters 분리 및 동일 held-out worlds 평가.
- Main pretraining은 여러 train worlds의 non-privileged history만 사용.
- Single/Dual neural exploitation의 동일 pretrained checkpoint 시작.
- 동일 B+K test budget 및 world 간 model/optimizer/recurrent state leakage 방지.
- Paired dual evaluation은 동일 B history와 post-exploration snapshot 사용; 홀수 B에서도 mode 유지.
- Frozen neural evaluation과 online evaluation을 분리; Q는 계속 online learning한다는 차이 명시.
- Interaction, PPO transition/update, replay, model parameter 수와 compute를 각각 기록.
- Multi-GPU throughput 개선과 학습 알고리즘 개선을 별개로 검증.

추가 로그는 policy별 advantage 분포, actor/value gradient norm, entropy, approximate KL, clip fraction, critic error, policy version을 포함하는 것이 좋다. Model probe를 위해 checkpoint와 필요한 관측 history도 보존한다.

## 10. 권장 실행 순서

1. Downstream credit regression test 및 여러 episode를 사용한 normalization 수정.
2. 단순 환경 smoke/learning test와 짧은 multi-world pilot.
3. 동일 reward worlds의 opponent ablation.
4. GRU+TD 및 관측 통계+PPO 진단 baseline.
5. 필요한 경우 auxiliary inference 및 backbone ablation.
6. 확인된 변경만 반영해 A1/A2/A3/Mixed, 5 seeds 전체 실험 재실행.

각 단계에서 learned exploration은 동일 exploiter를 사용하는 random/repeat/coverage history와 비교한다. 현재 추가 control에서는 learned history의 일관된 우월성이 입증되지 않았다. Q-table이 작은 문제에서 더 잘하는 결과도 유효한 연구 결과이며, neural 승리를 위해 환경을 사후 변경해서는 안 된다.

## 참고 자료

- [전체 실험 분석](REPORT.md)
- [Opponent별 집계](by_opponent.csv)
- [Advantage 진단 결과](advantage_audit.json)
- [Advantage 진단 코드](advantage_audit.py)
- [RL²: Fast Reinforcement Learning via Slow Reinforcement Learning](https://arxiv.org/abs/1611.02779)
- [Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs](https://proceedings.mlr.press/v162/ni22a.html)
- [Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables](https://proceedings.mlr.press/v97/rakelly19a.html)
