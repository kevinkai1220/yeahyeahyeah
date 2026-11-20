# Neural Off-policy TD Meta-Exploration — 실제 구현 및 CPU pilot

기준 commit `3dd5a95` 위에 독립 Neural TD 경로를 추가했다. 기존 PPO / Tabular Q source와 기존 results/checkpoints는 변경하지 않았다. 실행 시 새 코드는 uncommitted 상태였으며 run별 source SHA256/config/git provenance를 저장했다. 최종 배포 commit은 Git history를 참조한다.

**결론: Neural TD가 관측 transitions의 parameter update만으로 ordered combo를 실행할 수 있는 것은 확인했다. 그러나 기본 online update budget의 작은 pilot에서는 Meta-Explorer 우위나 Type 3 combo exploitation 개선을 입증하지 못했다.** Full-study는 실행하지 않았다.

## 구현

- Exploiter: 11 observed features → 2-layer ReLU MLP → 3 Q-values. GRU/history encoder 없음.
- Double DQN + target network + uniform replay + Huber + Adam, gamma=.99. Gamma1/hard-target/online-B/batch-B ablation 설정 가능.
- World마다 theta0 복사, target/optimizer/replay/RNG/statistics reset. B/K 및 mode reset은 nonterminal.
- Explorer: observation/Q/count/observed reward statistics/TD error를 받는 51-feature MLP actor/critic. Intrinsic reward 없음.
- Outer: complete `0.1 * B return + K return`의 MC reward-to-go, 같은 version의 independent rollouts를 batch-normalized PPO로 업데이트. Optional same-world LOO baseline 및 비용 기록.
- Pretrained theta0는 train-world data만 사용하고 meta-training 중 고정. Test adaptation은 world-local Q weights만 변경.
- 9-method evaluation, original seed/world/opponent/noise/metric code 재사용. Existing reference checkpoint가 없으면 unavailable을 명시하며 대체 모델을 만들지 않는다.

코드: `meta_exploration/neural_td.py`, `neural_td_experiment.py`, `neural_td_launch.py`. 설정 및 정확한 명령은 [사용법](../../docs/neural_td_meta.md).

## 1. Deterministic mechanism gate: 실패와 통과 모두 보존

Type 2/3 각각 3 seeds, constant opponent, sigma0. Randomly initialized Q, observed coverage data만 사용했다. True combo는 평가 점수에만 사용했다. K128에서 epsilon0, Q weights frozen. 각 case의 optimal combo utilization ≥90%를 통과 기준으로 했다.

| Gate | Prefix interactions | TD steps / world | 결과 |
|---|---:|---:|---|
| Online during B | 36 | 2,304 | 4/6 통과; Type2 seed0, Type3 seed1 실패 |
| Online, 더 긴 coverage | 180 | 2,880 | 5/6 통과; Type2 seed0 실패 유지 |
| Batch after B | 36 | 2,304 | **6/6 통과** |

Batch-B의 Type 3 utilization은 **100%, 98.4375%, 100%**였다. Type 2는 세 seed 모두 100%였다.

B36의 online/batch 비교는 동일 coverage 행동·환경 데이터·initial weights·총 optimizer steps를 사용한다. 따라서 단순 데이터 수나 update 수 차이가 아니다. 다만 update 시점에 따라 replay에서 볼 수 있는 데이터 분포가 달라진다. 초반 좁은 데이터의 과도한 반복학습과 optimization 경로 의존성이 원인 후보이며, 특정 내부 원인을 확정한 것은 아니다.

Batch32 samples ×2,304 steps = **73,728 sampled transitions/world**다. 이 gate는 높은 replay compute를 사용한 mechanism 검증이며, default-budget 적응 속도나 Q-table 대비 효율성의 증명이 아니다. 실패한 gate는 exit code1과 `passed=false`로 저장했다.

## 2. Small pilot 설계

- A2/A3 × seeds0/1: 총4 training runs.
- Train worlds6, held-out worlds4/seed/regime: regime당8 test worlds, **독립 training seeds는 2개**.
- Hidden32, theta0 pretraining960 interactions, main24 episodes × B32/K128 =3,840 interactions/run.
- TD prefix2 updates/step, K1 update/step. Main default prefix4보다 작은 CPU pilot 설정이다.
- 같은 theta0/optimizer/replay budget으로 Random/Coverage/Repeat/Existing Explorer/Meta 비교.
- Evaluation B16/32 × online/batch B-timing × Frozen-K/Online-K; 총 **1,152 world-method-condition rows**.
- Meta는 online-B/online-K로만 학습했다. Batch-B와 frozen-K는 evaluation ablations이며 해당 조건에 맞춰 재학습한 explorer가 아니다.

Original PPO reference는 로컬 staged pilot checkpoint를 사용했다: hidden32, train worlds12, constant opponent/sigma0, pretrain24, main96 episodes × B8/K32. 새 TD는 random/fixed opponents와 sigma1로 학습했다. Held-out evaluation 환경과 budgets는 맞췄지만 **training distribution과 horizon/pretraining 비용이 다르다**. 이 reference를 원 corrected full-study checkpoint 결과나 엄밀한 알고리즘 우위 증거로 부르지 않는다. 서버에서는 `--existing-root runs/full_study_corrected_v1`로 실제 full-study reference를 사용할 수 있다.

## 3. Primary pilot 결과: B32 / online-B / Online-K

각 숫자는 regime당8 held-out worlds 평균이다.

| Method | A2 reward | A2 optimal combo use | A3 reward | A3 optimal combo use |
|---|---:|---:|---:|---:|
| Neural TD Random | 2.126 | 0.59% | 2.030 | 0.78% |
| Neural TD Coverage | 2.118 | 0.39% | 2.042 | 0.59% |
| Neural TD Repeat | **3.333** | **42.58%** | 1.868 | 1.17% |
| Neural TD Existing Explorer | 2.680 | 20.31% | 2.042 | 0.59% |
| **Neural TD Meta** | **2.110** | **0.39%** | **2.042** | **0.59%** |
| Q-only reference | 2.506 | 12.89% | 2.157 | 4.10% |
| Dual Q reference | 2.887 | 46.68% | 1.818 | 22.66% |
| Single PPO reference | 2.079 | 40.43% | 0.725 | 11.13% |
| Dual PPO reference | 2.033 | 40.04% | 0.514 | 8.59% |

A3 Neural TD reward가 PPO reference보다 높지만 **최적 ordered combo 활용률은 더 낮다**. 주로 높은 immediate reward의 단순 행동으로 얻은 결과다. Combo exploitation 개선이라고 해석할 수 없다.

A2 Meta와 Coverage의 discovery는 모두100%였지만 활용률은0.39%였다. A3 Coverage discovery100%, Meta75%였으며 활용률은 둘 다0.59%였다. 발견을 실제 활용으로 연결하는 병목이 여전히 남는다.

Seed별 Meta reward:

| Regime | Seed0 | Seed1 |
|---|---:|---:|
| A2 | 2.043 | 2.177 |
| A3 | 2.064 | 2.019 |

## 4. Paired confidence intervals

동일 seed/world의 Meta−reference 차이를 먼저 구하고 seed→world hierarchical bootstrap(1,000회)을 수행했다. 아래는 primary K mean reward 차이의95% CI다.

| Contrast | A2 | A3 |
|---|---|---|
| Meta − Random | −0.016 [−0.040, 0.000] | +0.011 [0.000, +0.045] |
| Meta − Coverage | −0.008 [−0.032, 0.000] | 0.000 [0.000, 0.000] |
| Meta − Repeat | −1.223 [−1.988, −0.406] | +0.173 [0.000, +0.417] |
| Meta − Existing Explorer | −0.570 [−1.395, −0.008] | 0.000 [0.000, 0.000] |

일부 zero-width intervals는 이 표본에서 공통 RNG를 사용한 suffix의 reward가 같았기 때문이다. 모집단 정책의 동등성을 증명하는 것은 아니다. Training seeds가2개뿐이므로 interval 자체의 안정성도 제한적이다.

## 5. Timing / frozen ablation

A2 Repeat는 Frozen-K에서4.011, Online-K에서3.333이었다(online-B). K의 추가 TD update가 항상 이득이 아니었다. A2 Existing Explorer의 batch-B/Online-K는3.301로 online-B/Online-K의2.680보다 높았지만, 이는 작은 평가 ablation이며 meta-explorer의 학습 성공을 뜻하지 않는다.

A3 Meta/Coverage는 online-B, batch-B, Frozen-K, Online-K 모두 평균 약2.042로 차이가 없었다. 추가 데이터로 network parameters는 변했지만 이 표본에서 행동의 argmax가 달라질 만큼 유용한 adaptation으로 이어지지 않은 것으로 해석할 수 있다. 정확한 내부 원인은 추가 검사 대상이다.

Observed evidence가 확보된 group과 미확보 group의 결과는 `evidence_groups.json`에 저장한다. Group은 각 method의 history로 정해지므로 조건부 평균을 무작위배정에 의한 인과 효과로 해석하면 안 된다.

## 6. Training / compute

Run별 동일 비용:

- Theta0 pretraining: **960 interactions, 960 TD updates, 30,720 sampled transitions**.
- Meta outer training: **3,840 interactions, inner4,608 TD updates, 147,456 replay samples**.
- Outer PPO: **12 optimizer steps, 768 actor transitions, 3,072 token passes**.
- Parameters: Q **1,539**, target **1,539**(nontrainable), Explorer **2,852**.
- Test B32: prefix **64 TD updates/2,048 samples**. Frozen-K에서 추가0; Online-K에서는 추가 **128 updates/4,096 samples**.
- All TD baselines share exactly that test update budget. Meta alone additionally pays outer training; Existing Explorer has its own prior training cost.

Meta-training K mean reward, first8→last8 episodes:

| Regime / seed | Early → late |
|---|---:|
| A2 /0 | 2.325 → 2.673 |
| A2 /1 | 2.078 → 2.149 |
| A3 /0 | 2.033 → 2.046 |
| A3 /1 | 2.025 → 2.116 |

학습 곡선의 작은 상승을 generalization이나 combo 학습의 증거로 해석하지 않는다.Held-out comparison에서는 Meta 우위를 확인하지 못했다.

CPU wall time과 optimizer seconds는 `training_costs.json` 및 per-run runtime/provenance에 저장했다. CUDA가 없어 GPU time은 실행 검증하지 못했다. GPU 실행 시 TD update의 CUDA event time을 별도로 기록한다. Interaction 수, optimizer step 수, replay sample 수, GPU/CPU 시간은 서로 바꾸어 해석하지 않는다.

## 7. 검증 / 범위

실제로 검증한 내용:

- Double DQN의 online argmax / target evaluation 분리, terminal masking, gamma .99/1.
- Deterministic one-transition TD error 감소, hard/soft target 경로.
- AA/BB/CC 및 ordered AB/BA/CB 통계, mode reset과 odd-B boundary.
- B transitions가 실제 Q weights를 변경, Frozen-K의 weight/replay/optimizer 고정.
- Identical initialization, world별 Adam/replay reset, cross-world insertion 차단.
- Complete downstream return에 따른 explorer advantage/gradient 변화, batch normalization 보존, policy-version mismatch 검출.
- Same-world LOO target 보존, fixed theta0 및 checkpoint resume.
- Existing explorer prefix pairing, A1/A2/A3 metric 처리, failed-GPU exclusion 및 output collision 검사(mocked scheduler).
- 기존 PPO/Tabular regression 전체 포함.

최종 전체 test suite는 **121 passed, 2 CUDA skipped (40.01초)**였다. Neural TD 전용 suite는 **26 passed, 1 CUDA skipped**였다. 실행 기록은 `validation.json`에 저장했다. CUDA tests는 실제 장치 부재로 skip했고 성공으로 간주하지 않았다. 반복-evaluation과 source checksum을 확인했지만 large-scale training 및 A1/Mixed pilot은 실행하지 않았다.

## 연구 질문에 대한 현재 답

1. **Neural TD는 combo exploitation을 학습할 수 있는가?** 높은 replay budget의 batch-B deterministic gate에서는 가능했다. PPO보다 효율적/우월한지는 입증되지 않았다.
2. **Learned explorer가 Random/Coverage보다 유용한가?** 이번 작은 held-out pilot에서는 근거가 없다. A2에서는 Repeat/Existing보다 낮았다.
3. **Type3 발견이 활용으로 연결되는가?** Default pilot에서는 거의 연결되지 않았다. 발견률과 활용률을 별도로 봐야 한다.
4. **새 world 일반화가 되는가?** 2 seeds/8 held-out worlds의 결과로 긍정 결론을 내릴 수 없다. Outer training24 episodes도 매우 짧다.
5. **비용까지 포함해 이득인가?** 현재 Meta의 추가 outer compute에 상응하는 이득을 확인하지 못했다.

다음은 large-scale 재실행보다 **고정된 observed dataset에서 TD update 수·시점·pretrained vs random theta0·state representation을 분리**하는 것이 우선이다. Online-B의 초기 sparse replay에 과도하게 맞추는 문제인지 확인하고, default TD learner가 얻은 정보를 활용할 수 있음을 확인한 뒤 outer meta-training을 확대해야 한다. 새로운 recurrent exploiter나 hidden-label supervision을 넣어 결과를 좋게 만들지 않았다.

## 재현 명령

정확한 서버 명령은 [사용법](../../docs/neural_td_meta.md)에 있다. Raw data는 `runs/neural_td_meta_v1/`에, GitHub용 요약과 대표 PNG/PDF는 이 디렉터리에 저장했다. Full-study를 자동 시작하지 않았다.
