# 새로 업로드된 training diagnostics 분석

분석 대상: Git commit `0921018` (`Add corrected training diagnostics and trajectory samples`). 기존 결과를 보존하고 새 로그를 읽어 분석했다. 학습 코드 수정이나 추가 학습은 수행하지 않았다.

## 결론

1. **Batch normalization을 통한 episode 간 downstream return 신호 보존은 실제 학습 로그에서도 확인된다.** 다만 신호가 존재하는 것과 유용한 탐색을 학습하는 것은 다르다.
2. **A2(Type 2)의 Neural 성능 향상은 통계적으로 확인된다. 그러나 world별 좋은 combo를 추론하는 능력보다 seed별 선호 action을 반복하는 경향이 강하다.** 다른 좋은 combo를 탐색 중 관측해도 exploitation에서 거의 사용하지 않는다.
3. **A3(Type 3)는 held-out generalization 이전에 training 자체가 거의 개선되지 않는다.** GRU 용량 부족을 먼저 결론 내리기보다 PPO/critic/shared representation의 학습을 점검해야 한다.
4. **Q-table이 항상 우월한 것은 아니다.** 동일 prefix에서 A2는 Dual Neural이 Dual Q보다 높고, A3는 반대다. 순서가 있는 combo의 발견·활용과 명시적 TD update의 효과를 분리할 필요가 있다.
5. 기존 실험과 수정 실험은 normalization만 다른 비교가 아니다. Batch 수집, boundary target, pretraining, optimizer step cadence 등이 함께 바뀌었다.

## 자료와 검증 범위

`examples/full_study_corrected_diagnostics/`의 A1/A2/A3/B × 5 seeds, 총 20개 run을 분석했다. Main 비교는 online evaluation, B=32, K=128, regime별 500 held-out worlds다.

- 완료 marker 20개, update record 9,760개 및 world metric 8,000개 확인.
- Policy version 연속성, batch episode 구성, finite loss/gradient, optimizer step 합계 확인.
- 전역 정규화 advantage의 평균/표준편차 및 업데이트 후 parameter 변화 확인.
- 업로드된 aggregate와 raw world metric 간 464개 수치 비교 통과.
- Raw trajectory 800개 길이·mode·action chain·return 검사 통과.
- Dual Neural/Q의 raw prefix 200쌍은 method label을 제외한 관측/행동/reward/probability가 동일했다. 전체 world metric에서는 2,000쌍의 prefix 요약값이 일치했다.
- Dual Neural sample의 실제 action을 재생해 **32,000 transitions의 reward, opponent action, termination이 재현됨**을 확인했다. 이는 environment 재생 검증이며 neural checkpoint 추론 재실행은 아니다.

Raw trajectory는 A1에 없고, A2/A3 각 50 worlds, Mixed는 combo type 100 worlds에만 있다. Sample 선정 방식은 확인되지 않았다. 아래 주요 조건부 분석은 sample뿐 아니라 전체 A2 500 worlds의 metric을 사용한다.

Checkpoint, GPU runtime/hardware 원본 자료는 이번 업로드에도 없다. 실제 checkpoint 초기값 동일성, hidden-state intervention, GPU 활용률은 이 자료만으로 재검증하지 못했다. 실행 완료 표시는 성능·구현의 모든 정확성을 보증하지 않는다.

## 1. Credit assignment: 신호는 보존되지만 탐색 학습 성공은 별개

설정은 rollout batch 8, normalization=batch, dual credit=observed_suffix, gamma=1, GAE lambda=.95, exploration reward weight=.1이다. Online adaptation의 normalization은 여전히 `legacy_episode`다.

Explorer의 batch 내 episode별 normalized advantage 평균과 해당 downstream return의 상관은 A1=.99992, A2=.99999, A3=.99947, Mixed=.99986이었다. Episode별 normalized advantage 평균의 변동도 남아 있다. 따라서 과거의 episode별 정규화가 episode 간 downstream 차이를 제거하던 현상을 현재 training에 그대로 적용할 수 없다.

그러나 높은 return이 탐색 행동 때문인지, 쉬운 world/기존 exploiter 선호 때문인지는 이 상관으로 구분되지 않는다. 이 수치는 **신호 전달 검사**이지 **탐색의 인과적 기여 검사**가 아니다.

## 2. A2 개선의 성격: 선호 action과 맞는 world에서 강함

기존 결과와 새 결과를 동일 seed/world로 맞춘 hierarchical paired bootstrap(5,000회; seed를 뽑은 뒤 seed 내 world를 재표집)의 95% CI:

| A2 corrected − original | K-window mean reward 차이 | 95% CI |
|---|---:|---:|
| Single | +1.165 | [+0.884, +1.416] |
| Dual Neural | +0.767 | [+0.211, +1.209] |
| Dual Q | −0.554 | [−0.718, −0.378] |

반면 corrected Dual Neural − Single은 −0.064, CI [−0.151, +0.031]로 명확한 Dual 우위를 보여주지 않는다. CI가 0을 포함하는 것은 동등성을 증명하는 것도 아니다.

Seed별 초기 zero-history Single policy의 argmax action은 A/A/B/A/A였다. 업로드된 Dual Neural trajectory에서도 각 seed의 10개 world 모두 같은 action이 suffix의 최빈 action이었다.

이 선호 action과 실제 최적 same-action combo가 일치하는지에 따라 전체 A2 500 worlds를 나누면:

| Dual Neural 조건 | worlds | 최적 combo 활용률 | K mean reward | absolute oracle 90% 도달률 |
|---|---:|---:|---:|---:|
| 선호 action = 최적 combo | 164 | 93.2% | 5.064 | 100% |
| 선호 action ≠ 최적 combo | 336 | 4.27% | 1.126 | 0% |
| 불일치이지만 prefix에서 좋은 combo 시도 | 232 | 4.12% | 1.085 | 0% |
| 불일치이며 관측 reward의 성공 evidence도 확보 | 151 | 4.14% | 0.968 | 0% |

활용률은 valid second-action 기회 중 최적 combo 사용 비율이다. `시도`와 noisy reward 기반 `supported` 판정은 기존 metric 정의를 사용했다. Oracle 90%는 기존 absolute-ratio metric의 trailing window=16, consecutive hold=8 조건이다.

**불일치 world의 69.0%(232/336)에서 좋은 combo를 탐색 중 시도했는데, 그 world들에서도 활용은 약 4%다.** 따라서 여기서는 discovery 부족만으로 낮은 성능을 설명할 수 없다. 또한 높은 threshold 도달률도 대부분 기존 선호와 world가 맞는 경우에 집중된다. 이를 모든 world에서 빠른 history 기반 adaptation으로 해석하면 안 된다.

같은 prefix를 받은 Dual Q는 선호 일치/불일치 world에서 최적 combo 활용률이 각각 37.4%/25.7%로, Neural보다 이 선호에 덜 종속된다.

### 추가 실행한 고정-action 대조

단순 반복과 Neural의 차이를 확인하기 위해 CPU에서 500개 A2 held-out world를 추가 평가했다. Seed별 **초기 zero-history policy argmax**를 선택하고 전체 160 steps 동안 반복했다. Action 선택에 test reward나 hidden 정답을 사용하지 않았으며 원 실험과 world/opponent/noise seed를 맞췄다. 새로 학습한 baseline이나 checkpoint 추론은 아니다.

- 고정 반복 K mean reward: **2.011**
- Single − 고정 반복: **+0.471**, paired 95% CI [+0.351, +0.587]
- Dual Neural − 고정 반복: **+0.407**, CI [+0.302, +0.519]

즉 Neural을 단순 반복과 완전히 같다고 할 수 없다. 다만 hidden parameter를 diagnostic으로만 사용한 conditional expected reward 분해에서는 다음과 같다.

| A2 | Base component | Combo component | 합계 |
|---|---:|---:|---:|
| 고정 반복 | −0.355 | 2.363 | 2.008 |
| Single | 0.035 | 2.444 | 2.479 |
| Dual Neural | 0.002 | 2.413 | 2.415 |

Dual Neural의 고정 반복 대비 expected reward 개선 약 0.407 중 약 0.357은 base component 차이다. 개선 대부분을 새로운 최적 combo를 찾아 활용한 결과로 해석하기 어렵다. 이 분해는 행동 trajectory에 대한 관측적 분해이며 특정 학습 메커니즘의 인과 증명은 아니다.

## 3. A3: train에서도 개선되지 않음

Training 초기 120 episodes와 마지막 120 episodes의 downstream mean reward를 seed 평균으로 비교했다. 각 구간은 60 train worlds의 두 cycle이다.

| Regime | Single 초기 → 후기 | Dual Neural 초기 → 후기 |
|---|---:|---:|
| A1 | .808 → .844 | .817 → .834 |
| A2 | 1.373 → 2.608 | 1.464 → 2.556 |
| A3 | .475 → .459 | .457 → .429 |
| Mixed | .772 → .790 | .781 → .803 |

A3의 pretraining도 .474 → .475로 거의 정체했다. Held-out에서만 실패하는 문제가 아니다.

동일 prefix의 Dual Neural − Dual Q reward 차이는 A2에서 **+0.257**, CI [+0.034, +0.473], A3에서 **−1.398**, CI [−1.524, −1.282]다. Q-table의 명시적 value update가 Type 3의 활용에 유리할 가능성을 지지하지만, representation과 update algorithm을 동시에 바꾸므로 어느 요인이 원인인지는 분리되지 않는다.

## 4. Optimization diagnostics

아래는 seed별 마지막 20 updates의 median을 구한 뒤 seed 간 median을 취한 값이다. 3-action uniform entropy는 ln(3)=1.099다. Critic gradient는 value loss coefficient가 적용된 값이다.

| Policy | Entropy | Actor grad norm | Critic grad norm | Value explained variance | PPO clip fraction |
|---|---:|---:|---:|---:|---:|
| A2 Single | .413 | .305 | 334.99 | .181 | .009 |
| A2 Dual exploiter | .343 | .396 | 398.42 | .191 | .015 |
| A3 Single | 1.072 | .184 | 43.13 | .00019 | ≈0 |
| A3 Dual exploiter | 1.083 | .196 | 39.74 | ≈0 | ≈0 |
| A2 Explorer | 1.055 | .310 | 1340.23 | .00131 | ≈0 |
| A3 Explorer | .958 | .268 | 161.95 | .00477 | ≈0 |

후기 update에서는 batch 평균 total gradient norm이 global clipping 기준 .5를 지속적으로 넘는다. **Global gradient clipping과 PPO importance-ratio clipping은 다르다.** PPO clip fraction이 거의 0이므로 PPO ratio clip에 자주 걸려 학습이 멈췄다는 해석은 맞지 않는다.

Critic gradient 우세, 거의 0인 value explained variance, A3의 높은 entropy는 critic/shared-trunk optimization 점검을 우선할 근거다. 단, Adam의 좌표별 scaling과 gradient 방향 때문에 norm 비율을 actor parameter update 억제 배수로 해석해서는 안 된다. Head별 delta와 shared-gradient alignment는 현재 로그에 없어 원인으로 확정할 수 없다.

## 5. Budget 및 mixed generalization

Main policy별 optimizer steps는 기존 4,800 → 수정 600이며, pretraining은 1,200 → 152다. 하지만 main policy별 optimized transitions와 PPO epochs=4는 유지된다. **Step 수가 1/8이라는 것이 transition 처리량 또는 총 연산량도 1/8이라는 뜻은 아니다.** 더 큰 batch로 더 적게 optimizer.step을 수행한다.

각 main method는 192,000 training interactions를 사용한다. Actor transitions는 Single 192,000, Dual exploiter 153,600, Explorer 38,400이다. Shared pretraining interactions는 96,000이다. 이 비용과 추가 Explorer model/optimizer 비용을 분리해야 한다.

Mixed와 specialized를 동일 seed/world subset으로 맞춰도 Type 2 Neural gap은 남는다. Mixed − A2는 Single −1.506(CI [−2.107, −.815]), Dual −1.504([−2.091, −.879])다. 그러나 type당 train world 수는 60 대 20, main episodes는 1,200 대 약 400이므로 representation interference만으로 설명할 수 없다.

## 우선순위가 높은 다음 실험

1. **Batch normalization을 유지한 optimizer cadence ablation.** Rollout은 같은 policy version으로 8 episodes 수집하고 전체 batch에서 한 번 advantage normalization을 수행한다. 이후 sequence minibatch 크기를 달리해 update 횟수를 비교한다. Minibatch별 재정규화는 하지 않는다. 단순히 PPO epochs를 8배 늘리는 실험과 구분한다.
2. **Critic/shared-trunk ablation.** Value target scale 또는 value loss coefficient를 통제하고 actor/critic head별 parameter delta, shared-gradient norm/alignment를 기록한다. Reward objective 변경과 value optimization scaling을 혼동하지 않는다.
3. **동일 정보의 활용 검사.** 같은 post-B snapshot과 관측 기반 coverage history에서 Neural/Q, real/shuffled/reset history를 비교한다. 특히 기존 선호 action과 다른 좋은 combo가 관측된 A2 world를 분리한다.
4. **A3 쉬운 학습 조건 통과.** Constant opponent, sigma=0, 단일 world부터 실제 train improvement를 확인한 뒤 opponent/noise/world diversity를 추가한다.
5. 그 다음 observed-statistics input, GRU+TD/Q-head를 비교해 memory와 update algorithm을 분리한다. 지금 자료만으로 GRU 확대나 Transformer 교체를 우선할 근거는 부족하다.

Opponent는 random/fixed cyclic이며 co-learning opponent가 아니다. Opponent가 learning을 방해하는지의 인과적 결론은 같은 hidden world에서 opponent만 바꾼 통제 실험 전에는 내릴 수 없다.

## 재현 및 산출물

Repository root에서 numpy/torch/matplotlib 사용이 가능한 Python으로 실행한다.

```bash
python analysis/corrected_training_diagnostics/analyze.py
python analysis/corrected_training_diagnostics/deep_checks.py
python analysis/corrected_training_diagnostics/behavior_checks.py
```

세 스크립트 모두 실제 실행 완료했다. 분석 assertion 및 CPU environment replay를 실행했으며, 이번 작업에서 PPO unit test/CUDA smoke test/추가 neural training을 실행했다고 주장하지 않는다.

주요 산출물:

- `audit.json`, `replay_audit.json`: 자료 정합성 검증 결과
- `paired_contrasts.json`: seed/world paired bootstrap 비교
- `A2_conditional_behavior.json`: 선호 일치·발견·관측 evidence별 활용 및 threshold 도달
- `fixed_action_comparison.json`, `reward_decomposition.json`: 추가 고정 반복 대조와 보상 분해
- `training_stats.json`, `credit_correlation.json`, `training_endpoints.json`: optimization 및 학습 변화
- `A1_training.png/pdf`, `A2_training.png/pdf`, `A3_training.png/pdf`, `B_training.png/pdf`: 실제 training log 그래프

기존 `analysis/corrected_full_study/` 분석은 보존했다. 이 보고서는 새 로그 덕분에 확인할 수 있게 된 진단과 기존 해석의 보완을 담는다. 5 seeds의 bootstrap CI 역시 적은 seed 수에 따른 불확실성을 가진다.
