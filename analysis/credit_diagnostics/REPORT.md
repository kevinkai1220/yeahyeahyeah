# 학습 파이프라인 수정 및 CPU pilot 검증 결과

작성일: 2026-11-19. 기존 full-study 및 분석은 보존했다. 이번 결과는 새로운 CPU pilot이며 원본 GPU full-study 결과를 대체하지 않는다. 실험 실행 시점의 commit과 source 상태는 각 run의 provenance에 기록했다.

## 결론

- Policy별 multi-episode batch PPO, actor-only batch advantage normalization, complete observed suffix 기반 explorer target을 구현했다.
- Episode/version/recurrent context 분리, variable-length valid-token loss, CUDA GRU train-mode forward, checkpoint/RNG resume를 검증하는 테스트를 추가했다.
- 작은 deterministic train world에서는 Neural이 random보다 높은 reward를 학습한다. 그러나 이번 pilot에서 corrected 방식의 held-out 성능 우위는 입증되지 않았다.
- Type 3와 mixed 조건에서는 여전히 낮은 neural 성능과 큰 seed variation이 보인다. GRU 자체의 불가능성으로 해석할 근거는 없다.
- Critic gradient dominance와 global clipping은 다음 optimization 진단의 구체적인 후보다.
- Time-to-threshold, censoring/RMST, normalized AUC, paired hierarchical CI와 PNG/PDF graph A–D를 실제 trajectory에서 계산했다.
- CUDA 장치가 없어 실제 CUDA backward 검증은 미실행이다. Full-study 전체 학습과 Phase 3~6 architecture 확장은 실행하지 않았다.

[구현 이론·인터페이스·전체 명령](../../docs/meta_learning_diagnostics.md)

## 실행 범위

- Easy: single train world, constant A opponent, sigma=0, 4 training variants × 3 seeds.
- Staged: multi-world Type 1, Type 2, Type 3, Mixed+noise, corrected variant × 3 seeds.
- 각 job: shared pretraining 24 episodes, main 96 episodes, B=8/K=32, hidden=32, learning rate=.001, PPO epochs=4.
- Test: seed당 6 held-out worlds, frozen/online 분리. Train 진단은 train-world 집합의 앞 6개(단일-world에서는 1개)를 평가했다.
- Shared pretraining은 variant 간 동일 weights임을 assert했다. Historical full-study의 pretraining까지 복제한 비교는 아니다.
- Seed/world bootstrap CI는 작은 pilot에서는 200 replicates, history intervention contrast는 1000 replicates다. 다중 비교 보정 및 equivalence 검정은 수행하지 않았다.

## 수정한 기능

| 파일 | 변경 |
|---|---|
| `learning.py` | PPO batch API, normalization mode, prefix/suffix targets, policy version 검증, gradient/optimization 로그, cuDNN mode 처리 |
| `training.py` | Policy별 queue/flush, batch-boundary checkpoint, RNG/provenance, 실제 optimizer budget |
| `environment.py` | Constant A opponent 및 uniform-random finite-horizon DP reference |
| `evaluation.py` | World별 speed diagnostic, 별도 adaptation compute 기록 |
| `adaptation.py` | Exact reference curve, threshold/censoring, hierarchical/paired 통계, graph A–D, raw JSONL backfill |
| `diagnostics.py` | Staged pilot, shared initialization 검증, train/test 평가, context interventions, 평가 전용 B/K study |
| `plotting.py` | Episode 행과 optimizer update 행 구분 |
| `configs/meta/{corrected,diagnostic_pilot,credit_smoke}.json` | 별도 새 실행 설정 |
| `tests/meta/test_credit_speed.py` | Credit/gradient, boundary, version, hidden isolation, backward, resume, speed, backfill tests |

Legacy default는 명시적으로 유지하며 새 학습은 corrected config를 선택한다. Critic이 서로 다르다는 이유만으로 legacy GAE가 수학적으로 무효인 것은 아니다. Corrected target은 suffix critic 오차 의존을 줄이는 대신 sampled return variance를 받아들이는 선택이다. 그 성능 효과는 별도 ablation으로 검증해야 한다.

## Easy: train 성능과 held-out 일반화 분리

다음 값은 frozen K 구간 step당 평균 reward다.

| Variant | Single train | Dual Neural train | Single test | Dual Neural test |
|---|---:|---:|---:|---:|
| legacy | 2.978 | 2.905 | 0.976 | 0.943 |
| legacy_batch | 2.500 | 2.478 | 1.149 | 1.067 |
| batch_cross_critic | 2.621 | 2.512 | 1.131 | 1.072 |
| corrected | 2.621 | 2.512 | 1.131 | 1.072 |

Uniform-random 기대 reward는 train 1.008, test 0.966이다. Corrected가 train world에서 random보다 높다는 사실은 학습 배선이 전혀 작동하지 않는다는 설명과 맞지 않는다. 단, 단일-world 학습 성공은 meta-adaptation 성공의 증거가 아니다.

동일 batch size=8인 legacy_episode normalization 대비 corrected의 test paired 차이:

- Single: −0.0176, 95% CI [−0.0532, +0.0083].
- Dual Neural: +0.00449, 95% CI [−0.00880, +0.01856].

두 interval 모두 0을 포함한다. Easy에서는 batch_cross_critic과 corrected의 측정 reward가 같았다. 이는 다른 world type에서도 target 변경 효과가 없다는 증명이 아니다.

[전체 seed별 수치](easy_summary.json), [paired comparisons](easy_paired_comparisons.json)

## Staged: corrected의 held-out frozen 성능

| 조건 | Random 기대값 | Single | Dual Neural | Q-only | Dual Q |
|---|---:|---:|---:|---:|---:|
| type1 | 0.966 | 1.047 | 0.885 | 1.932 | 1.651 |
| type2 | 0.431 | 1.656 | 1.940 | 1.973 | 2.100 |
| type3 | 0.446 | 0.472 | 0.336 | 1.757 | 1.784 |
| mixed_noise | 0.502 | 0.450 | 0.720 | 1.214 | 1.133 |

Type 2 Dual Neural의 seed별 평균은 4.380 / 0.953 / 0.487로 변동이 크다. Type 3 Dual Neural은 0.336으로 random 기대값 0.446보다 낮다. 이 pilot의 96 episodes/12 main batch updates만으로 convergence 또는 architecture의 한계를 판정할 수 없다. Staged 조건은 난이도를 순차적으로 추가한 것이며 opponent 단일요인 비교는 아니다.

[Seed별 train/test 및 CI](staged_summary.json)

## History utilization 진단

모든 intervention은 동일 실제 post-B state, mode, opponent/RNG snapshot 및 action-sampling seed를 사용했다. Synthetic history는 환경에서 재실행하지 않았다.

Type 2에서 real history − reset 차이는 +0.1166, exploratory 95% CI [+0.0074, +0.2400]이었다. 하지만 reward-shuffled 및 action-relabelled history와의 reward 차이는 해당 small sample에서 0이었다. 따라서 reset 효과를 reward structure 추론의 성공으로 바로 해석할 수 없다. 일반적인 recurrent warm-up 효과도 가능하다.

Type 3에서 real − reset은 +0.00224, CI [−0.1114, +0.1035]였다. 여러 조건에서 관측 action/reward가 동일했더라도 hidden state나 action probability까지 같다는 뜻은 아니다. Common-random action sampling에서 작은 확률 차이는 행동 차이로 나타나지 않을 수 있다.

[History paired results](history_paired.json)

## Optimization 진단

Easy/corrected, 3 seeds × 12 batches의 batch-mean gradient norm 중앙값:

| Policy | Actor loss gradient | Weighted critic loss gradient |
|---|---:|---:|
| Single | 0.285 | 69.558 |
| Explorer | 0.227 | 112.753 |
| Dual exploiter | 0.333 | 70.627 |

모든 batch의 평균 합산 gradient norm이 clip threshold .5보다 컸다. Actor/critic gradient는 각 loss가 전체 model parameters에 만드는 norm으로 shared GRU도 포함한다. 이것은 critic 항이 shared representation 및 global clipping에 강하게 영향을 줄 가능성의 근거이며 인과적 증명은 아니다. Adam의 per-parameter rescaling과 gradient 방향 때문에 norm 비율만으로 실제 actor parameter update 감소량을 계산할 수 없다.

다음으로는 backbone 교체 전에 동일 데이터/batch budget에서 value coefficient와 clipping을 분리한 ablation, critic prediction error/gradient ratio 추적을 권장한다. 아직 이 optimization ablation은 실행하지 않았다.

[Gradient 집계](gradient_diagnostics.json)

## Budget 및 runtime

각 job의 shared pretraining interaction은 1,344, 각 main method의 논리적 training budget은 3,840 interactions다. Single optimized transitions=3,840; Dual exploiter=3,072; Explorer=768이다. Neural model 하나는 6,852 parameters이며 dual은 두 model을 사용한다.

- Legacy batch=1: policy당 optimizer steps 384.
- Batch=8 세 조건: policy당 optimizer steps 48.
- Shared pretraining: optimizer steps 12.
- Q-table은 world마다 초기화하며 96 entries, inter-world transfer 없음.

따라서 legacy 대비 corrected의 성능 차이는 normalization 외에 update 횟수 차이를 포함한다. Legacy_batch 대조군이 이를 구분한다. Runtime은 logging/diagnostics와 평가를 포함하고 다른 CPU 작업과 동시 실행되었으므로 GPU throughput 또는 엄밀한 속도 benchmark로 해석하지 않는다.

[Easy budgets](easy_budgets.json), [Staged budgets](staged_budgets.json)

Pilot 실행 후 추가한 per-world Q/PPO evaluation timing 필드는 최종 CLI smoke와 평가 전용 K=128 study에서 검증한다. 앞선 pilot metrics에 해당 필드가 없다고 계산하지 않은 시간을 추정해 채우지 않는다.

## Adaptation speed: 실제 K=32 pilot

Absolute ratio 기준, easy/corrected online 평가. 18 held-out worlds 중 oracle mean이 양수가 아닌 1 world는 absolute ratio에서 제외했다. 집계는 valid world를 seed 안에서 평균한 뒤 seed에 동일 가중치를 준다.

| Method | Reach@25% | Reach@90% | RMST@90% |
|---|---:|---:|---:|
| Single | 46.7% | 16.7% | 30.50 |
| Dual Neural | 53.3% | 11.1% | 31.00 |
| Q-only | 94.4% | 17.8% | 31.49 |
| Dual Q | 93.3% | 0.0% | 32.00 |

Dual Q는 이 조건에서 90% threshold에 한 번도 도달하지 않았다. 미도달을 제외하지 않고 RMST=K로 반영했다. W=16, hold=8이라 earliest confirmation=23이고 K=32는 관측 가능한 도달 시간 범위가 매우 좁다. 이 결과를 충분한 장기 adaptation 비교라고 해석하지 않는다.

[전체 threshold/normalization 수치](easy_speed/aggregated_speed.csv), [paired speed](easy_speed/paired_speed.json). A–D graphs는 해당 폴더에 PNG/PDF 32쌍으로 저장했다. K=32 pilot의 B는 8 하나이며, 여러 B에 대한 실제 비교는 별도 K=128 evaluation을 사용한다.

## 기존 full-study raw availability

로컬에 업로드된 `examples/full_study`의 64,000 metric rows에 해당하는 raw JSONL을 확인했으나 없었다. 0개 trajectory를 복원했고 기존 aggregate만으로 crossing time을 만들어내지 않았다. 원격 GPU 서버 접속 정보는 제공되지 않아 서버 filesystem 자체를 검사한 것은 아니다. 서버의 원본 raw JSONL에는 문서의 `--backfill-raw` 명령을 사용할 수 있다.

[Raw availability audit](original_raw_audit.json)

## 테스트와 미실행 항목

- 나눠 실행한 전체 테스트: **82 passed, 1 CUDA skipped**. 마지막 return-logging 변경 후 관련 두 회귀 테스트도 통과했다. [검증 기록](tests_split.txt).
- 중간 monolithic 테스트와 추가 K=128 평가가 SIGTERM(143)로 중단되어 완료로 취급하지 않았다. 테스트는 나눠 재실행했고, 추가 평가는 seed별 작업으로 재실행해 완료했다. 기존 완료 산출물은 보존했다.
- Corrected CPU CLI train+evaluate smoke 완료: [smoke.txt](smoke.txt).
- CUDA: PyTorch 2.15.0+cpu, cuda_available=false, nvidia-smi 없음. CUDA test는 skip이며 실제 GPU backward 통과라고 주장하지 않는다. [hardware.json](hardware.json)
- 기존 full-study checkpoint/results를 수정·삭제하지 않았다.
- 20-job full-study는 실행하지 않았다.
- GRU+TD, statistics+MLP, causal opponent/exposure 비교, LSTM/Transformer 및 auxiliary architecture는 구현 계획/인터페이스만 정리했다.
- Learned exploration이 random/repeat/coverage보다 우월한지 최종 검증하는 Phase 4 runner는 후속 작업이다. 이번 수정만으로 우월성을 주장하지 않는다.

## K=128, B=8/16/32/64 evaluation-only 검증

Easy/corrected checkpoint를 그대로 사용해 3 seeds × 6 held-out worlds × 4 B budgets × 2 evaluation modes × 4 methods = **576개 world/method records**를 수집했다. Training은 B=8/K=32였으므로 이 결과는 evaluation-budget generalization이며, K=128로 재학습한 model의 결과가 아니다. Type 1 / constant opponent / sigma=0 진단에 한정된다.

Absolute ratio, online, 90% threshold, valid world 17개(oracle 비양수 1개 제외), equal-seed 집계:

| Method | B=8 RMST | B=64 RMST | B=8 포함 비용 | B=64 포함 비용 | Reach@90% B=8 → B=64 |
|---|---:|---:|---:|---:|---:|
| Single | 100.88 | 97.19 | 108.88 | 161.19 | 28.9% → 34.4% |
| Dual Neural | 101.06 | 101.06 | 109.06 | 165.06 | 28.9% → 28.9% |
| Q-only | 92.62 | 88.03 | 100.62 | 152.03 | 46.7% → 46.7% |
| Dual Q | 115.33 | 61.02 | 123.33 | 125.02 | 23.3% → 75.6% |

Dual Q는 더 긴 B에서 exploitation 시작 이후의 threshold 도달이 빨라졌지만, 탐색 비용을 포함하면 전체 sample efficiency 향상을 입증하지 못했다. 이것은 combo world에 대한 결론이 아니며 learned explorer의 random/coverage 대비 우위를 입증하는 실험도 아니다.

- Dual Q B64 − B8 rmst_90: -54.311, paired hierarchical 95% CI [-74.892, -32.498].
- Dual Q B64 − B8 total_interaction_rmst_90: +1.689, paired hierarchical 95% CI [-18.892, +23.502].
- Dual Q B64 − B8 reach_90: +0.522, paired hierarchical 95% CI [+0.189, +0.833].

[Budget paired 결과](budget_paired_K128.json), [online PPO/Q update 비용](adaptation_compute_K128.json), [checkpoint 출처와 runtime](K128_sources_and_runtime.json).

전체 기록: `runs/credit_speed_K128_complete_v1/`. 중단된 첫 평가 `runs/credit_speed_K128_v1/`도 삭제하지 않았다. 완료된 seed 0은 그대로 복사했고 seed 1/2는 별도 작업으로 재실행했다.

K=128 graph A–D 및 모든 threshold/normalization CI: [speed_K128](speed_K128/aggregated_speed.csv).

K=128 speed figure 검증: PNG 104개/PDF 104개 생성. Compact world records로 history/budget paired contrasts를 재계산해 저장된 수치와 일치함을 확인했다. [재현 script](reproduce_comparisons.py).
