# History utilization / PPO optimization: 구현 및 실행 보고서

기준 repository commit: `0921018`. 이번 변경은 아직 commit/push하지 않았다. 기존 Full-study 및 corrected Full-study 결과와 checkpoint는 보존했다.

**서버의 `runs/full_study_corrected_v1` checkpoint는 이 로컬 환경에 없다. 따라서 아래 수치는 full-study checkpoint에 대한 검증 결과가 아니다.** 그 checkpoint를 직접 읽는 서버 실행 도구를 구현했고, 로컬의 이전 소규모 pilot checkpoint로 기능과 진단 절차를 검증했다. 서버 실행 명령 및 실험 정의는 [사용법](../../docs/history_optimization_diagnostics.md)에 있다.

## 변경한 기능

- `meta_exploration/history_diagnostics.py`: checkpoint/config 검증, shared initialization equality audit, 공통 post-B snapshot intervention, preference switching, ordered-pair conditional probability, empirical completion, evidence curves, observed-statistic probes, paired seed/world bootstrap.
- `meta_exploration/history_diagnostics_summary.py`: 독립 seed 실행 결과 통합, paired 비교, probability/reward/probe 요약, PNG/PDF.
- `meta_exploration/learning.py`: global advantage normalization을 유지하는 sequence minibatch 옵션, original-unit objective를 유지하는 critic residual scaling, shared actor/critic gradient alignment, group별 clipping norm 및 parameter delta. 기본값은 기존 corrected PPO와 동일하다.
- `meta_exploration/training.py`: 실제 update log로 optimizer step 비용 계산. 짧은 final batch 및 minibatch cadence를 반영한다.
- `meta_exploration/optimization_pilot.py`: fresh single-world deterministic Type 3부터 순차 확대할 수 있는 작은 pilot, 고정된 shared pretraining, oracle sanity check, Random/Repeat/Q/Single/Dual 평가.
- `configs/meta/history_optimization_pilot.json`, regression tests 및 문서.

History intervention은 learning objective를 바꾸지 않는다. Shuffled history는 counterfactual context로 표시하고 Q의 fabricated TD data로 사용하지 않는다. Coverage/observed-success histories도 자기 sandbox에서는 실제 관측이지만 공통 anchor로 옮기는 비교는 counterfactual memory intervention이다. End-to-end history generator 우열과 구분한다.

## 실제 실행 범위

1. 수정 전 meta regression: **57 passed, 1 CUDA skipped**.
2. 변경 후 전체 test suite 실행. 최종 결과는 아래 검증 기록에 기재한다.
3. 이전 staged pilot의 A2/A3 × seeds 0/1/2 × 6 held-out worlds: frozen/online intervention smoke 실행. 이 모델은 hidden32, B8/K32의 작은 pilot 모델이다.
4. 관측 기반 반복-evidence 생성기를 보완한 최종 frozen pilot: 같은 3 seeds × 6 worlds × A2/A3. Full-study 모델로 대체하지 않았다.
5. Single Type 3 world, constant opponent, sigma0, seed0, shared pretrain24 + main128 episodes, hidden32, B18/K32. Baseline / cadence1 / value coefficient .05 세 variant 실제 실행.
6. CPU torch `2.15.0+cpu`, CUDA unavailable, `nvidia-smi` 없음. CUDA backward smoke는 GPU 서버에서 실행해야 한다.

출력은 `runs/history_optimization_diagnostics_v1/`에 분리했다. 각 checkpoint 평가에는 SHA256/config/git/source provenance를 남긴다. 모든 variant의 shared pretrain 동일성은 실제 assert했다. 추가 final-code smoke에서 이전 checkpoint의 Single/Dual 초기 weights와 shared pretrain equality도 확인했다.

## A2: local pilot의 switching 및 history 효과

Frozen Dual 기준 18 worlds 중 initial preference와 정답이 다른 world는 10개다. 보완된 observed-success history에서 실제 좋은 combo의 반복 evidence까지 확보한 mismatch world 5개에서도 switching은 0/5였다. Real history 이후 handoff argmax switching은 **0/10**이었다. 일치한 8개에서는 optimal action argmax가 유지됐다. 이는 handoff의 분포에 대한 측정이며 suffix 모든 시점의 전환 여부를 뜻하지 않는다.

같은 actual handoff observation에서 reset 대비 probability 차이를 측정하고, real과 intervention의 suffix action RNG를 맞췄다.

- 불일치 group의 real-history good-action probability: **0.1674**.
- 같은 group의 reset probability: **0.2464**.
- Real exploitation의 good-combo utilization: **3.75%**.
- Reward shuffle은 action probability를 조금 바꿨지만 이 유한 sample에서 sampled suffix 행동/reward는 같았다. 이것은 두 policy distribution이 수학적으로 같다는 뜻이 아니다.

전체 18 worlds, 3 checkpoint seeds의 hierarchical paired bootstrap(1,000회), **intervention − real**:

| Intervention | K mean reward 차이 | 95% CI | Good-action probability 차이 |
|---|---:|---:|---:|
| Reward-shuffled | 0.0000 | [0, 0] | −0.00018 |
| Reset | −0.1166 | [−0.2400, −0.0074] | −0.06997 |

Reset의 reward 감소만으로 reward information 활용을 입증할 수 없다. GRU warm-up이나 기존 선호 강화와도 양립한다. Mismatch group에서 real history가 good-action probability를 낮추는 패턴을 함께 봐야 한다. 3 seeds의 작은 smoke 결과를 5-seed Full-study에 일반화하면 안 된다.

## A3: marginal을 넘어서 ordered pair 검사

동일 local pilot Dual, frozen real history, 18 worlds:

| 측정 | 값 |
|---|---:|
| 좋은 ordered pair의 첫 action probability | 0.4161 |
| 첫 action 강제 수행 후 두 번째 action probability | 0.2532 |
| world별 conditional product의 평균 | 0.08181 |
| 실제 suffix ordered-pair completion | 0.09028 |

두 probability 평균의 곱과 world별 product 평균은 다르다. Conditional product는 각 snapshot의 한 opponent/noise response에 대한 diagnostic이며 unconditional completion expectation으로 부르지 않는다.

Reset−real reward 차이는 **−0.00224**, paired CI **[−0.10350, +0.11140]**였다. 이 표본에서는 real history의 명확한 reward 우위를 확인하지 못했다.

최종 observed-success generator는 coverage 후 **관측 reward가 좋은 pair를 반복**해서 evidence를 확보한다. Hidden 정답은 selection에 쓰지 않고, 이후 diagnostic에서 실제 good combo의 evidence 여부를 구분한다. 최종 A3에서 실제 좋은 combo의 반복 evidence가 확인된 11/18 worlds의 utilization은 real history 6.25% → 선택한 evidence history 6.82%였다. 표본 평균 reward는 .421 → .465였다. A2의 해당 9/18 worlds에서는 sampled suffix reward/utilization이 real과 같았다. 상세 성공-evidence 조건부 비교는 `executed_intervention_details.json`에 저장한다. Evidence가 확보되지 않은 sample을 successful history라고 해석하지 않는다.

## Hidden-state probe

GRU가 실제 ingest한 tokens로 counts/reward mean/variance/opponent count/마지막 mode-action 정보를 계산했다. 최종 prefix reward는 아직 GRU가 읽지 않았으므로 target에도 미리 넣지 않는다.

각 checkpoint당 6 worlds를 world-disjoint 4 train / 2 test로 나눈 linear ridge probe는 매우 작은 학습 sample이다. Dual의 count/reward-mean prediction은 pooled error 기준으로 train-mean baseline보다 나빴다:

| Regime | Count skill vs train mean | Reward-mean skill |
|---|---:|---:|
| A2 | −3.30 | −1.07 |
| A3 | −0.77 | −1.30 |

이는 **이 readout과 표본에서 decoding에 성공하지 못했다**는 결과다. GRU에 정보가 없다는 증거로 확정하지 않는다. Variance target이 상수/거의 0인 항목의 skill은 해석하지 않는다. Leave-one-seed-out probe도 제공하지만 checkpoint 간 hidden coordinate가 정렬되지 않았다는 추가 한계가 있어 보조 진단으로만 사용한다.

## A3 deterministic minimal learning: reward는 올랐지만 combo 실행 실패

이 실험은 seed0의 한 train world, 128 episodes, 짧은 K32다. 원 full-study를 재현하는 실험이 아니다. 학습 초/후기는 첫/마지막 16 episodes의 K mean reward다.

| Variant | Single 초→후 | Dual 초→후 | Frozen train Single | Frozen train Dual | Dual optimal combo utilization |
|---|---:|---:|---:|---:|---:|
| Baseline | 1.142→1.883 | 0.980→1.887 | 2.593 | 2.150 | 0% |
| Minibatch1 | 1.166→2.074 | 0.970→2.401 | 2.593 | 2.389 | 0% |
| Value coefficient .05 | 1.142→1.891 | 0.987→1.909 | 2.593 | 2.268 | 0% |

Diagnostic oracle의 이 world 평균은 **5.812**이며, sigma0/constant 조건에서 DP와 실제 oracle rollout return이 일치했다. Neural은 쉬운 조건에서도 이 budget 내에서 높은 ordered-combo 실행 확률을 학습하지 못했다. 단일 world에서 reward가 오르는 현상은 immediate-reward trap으로의 이동과 양립한다.

동일 train world의 Q-only=2.802, Random=−0.037, Repeat=−1.356이었다. 여기서 Neural은 이 Random/Repeat rollout보다 높지만 Q-only보다 낮았다. 한 world/seed의 sampled rollout이므로 우열의 일반적 증거가 아니다. Dual Q는 variant별 explorer history가 달라지므로 variant 간 차이를 동일-history exploitation 차이라고 해석하지 않는다.

## Gradient와 cadence

마지막 Dual update의 실제 로그(각 update 내부 optimizer operations 평균):

| Variant | Shared actor norm | Shared critic norm | Shared cosine | Total pre-clip | Total post-clip |
|---|---:|---:|---:|---:|---:|
| Baseline | .01959 | 7.18959 | +.04108 | 95.322 | .500 |
| Minibatch1 | .000109 | .24557 | −.03802 | 87.622 | .500 |
| Value coefficient .05 | .03185 | .75487 | −.30974 | 9.623 | .500 |

Baseline의 actor-head delta=.02237, critic-head delta=.02657, shared delta=.16649였다. Actor update가 완전히 0인 상황은 아니다. Coefficient를 줄여 critic norm은 줄었지만 combo execution은 해결되지 않았다. Cosine의 부호가 variant/학습 시점에 따라 달라지므로 “항상 critic이 actor를 반대로 미는 것이 원인”이라고 단정할 수 없다.

Main method당 interactions는 모두 **6,400**, PPO epochs는 **4**다. Single token passes=25,600, Dual exploiter=16,384, Explorer=9,216을 유지한다. Optimizer steps만 baseline/value-coef는 각 policy **64**, cadence1은 **512**다. Shared pretraining은 각 run에서 interactions **2,064**, optimizer steps **12**로 동일하다.

이 pilot은 구현 도중 추가된 optimizer-cost 집계 패치 전에 시작했으므로, 새 pilot의 `costs.json`은 실제 `training.jsonl` update 수로 재집계하고 `minimal_cost_accounting_audit.json`에 이전 값과 실제 값을 남겼다. 원 Full-study 파일은 변경하지 않았다. 학습 결과나 gradient를 재계산해서 개선한 것은 아니다.

## 확인된 사실과 남은 가설

확인된 사실:

- Real/shuffled/reset을 같은 actual state에서 실행하는 코드와 odd-B/online isolation regression이 동작한다.
- 로컬 pilot의 mismatch world에서는 좋은 action으로 handoff switching이 없었다.
- Minibatch cadence만 늘리거나 critic coefficient만 줄이는 작은 실험으로 Type 3 ordered-combo 실행을 해결하지 못했다.
- Global clipping이 발생하지만 actor head도 실제로 업데이트된다.

아직 확인되지 않은 것:

- **서버 Full-study checkpoint의** switching, intervention 및 hidden-state probe 결과.
- 충분한 train worlds/episodes/seed에서 GRU에 정보가 없어서 실패하는지, readout/actor optimization이 실패하는지.
- Shared trunk gradient 간섭의 인과 효과. Group clipping/scale 옵션은 구현했지만 이번 main pilot에서 실행하지 않았다.
- Opponent의 원인 기여: deterministic 첫 단계만 실행했다. 이후 random/noisy/multi-world stage는 명령으로 선택 가능하다.
- GRU+TD, observed-statistics policy, architecture 변경은 서버 checkpoint 결과 확인 후 결정해야 하므로 이번에 추가하지 않았다.

## 검증 기록 및 다음 실행

반복-evidence collector의 최초 실행에서 pair schedule 길이 오류를 발견했다. 9-slot schedule로 수정하고 반복 수집 regression을 추가했다. 실패 디렉터리는 덮어쓰지 않았고 최종 실행은 `final_frozen_pilot_v2`에 분리했다.

최종 전체 test suite는 **95 passed, 1 skipped (CUDA), 69.11초**였다. 실행 개수와 검증 기록은 `validation.json`을 참조한다. CUDA 실행은 미검증이다. 서버에서는 [문서의 checkpoint pilot 명령](../../docs/history_optimization_diagnostics.md)을 먼저 실행한다. 대규모 재학습/백그라운드 full-study는 시작하지 않았다.
