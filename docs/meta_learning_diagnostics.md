# Meta-exploration 학습 신뢰성 및 적응 속도 진단

## 범위와 보존 원칙

Phase 1과 Phase 2를 구현한다. 기존 네 main method와 GRU backbone을 유지한다. Phase 3~6의 architecture 변경 및 20-job full-study는 자동 실행하지 않는다. `runs/full_study`, `examples/full_study`, 기존 checkpoint/분석은 수정하지 않는다.

기존 default는 재현을 위해 `rollout_batch_size=1`, `advantage_normalization=legacy_episode`, `dual_credit=legacy_cross_critic`다. **새 연구 실행에는 `configs/meta/corrected.json`을 명시한다.** 다른 config로 기존 output directory를 resume하면 거부한다. 새로운 objective를 기존 final model에서 이어 학습하는 경로를 이 비교에 사용하지 않는다.

## 1. Batch PPO

`learning.ppo_batch_update`는 각 policy별 `(history, raw_advantage, critic_target, prefix_context)` 목록을 받는다. `RolloutBatch`는 한 model/optimizer의 trajectory만 보관한다. Training은 같은 policy version으로 여러 episode를 완전히 수집한 뒤 업데이트한다. Pretraining도 같은 구조를 사용한다.

- Single: B+K 전체 actor loss. Optional K-only는 기존 ablation.
- Explorer: B actor loss.
- Exploiter: K actor loss, B history는 recurrent context로만 전달.
- Critic target은 actor advantage normalization으로 변형하지 않는다.
- 서로 다른 policy의 advantage를 합치지 않는다.
- GRU batch row마다 zero hidden state로 시작하고 각자의 context만 replay한다.
- Variable-length row는 오른쪽 padding 후 valid segment만 골라 모든 loss/통계를 계산한다. 뒤쪽 padding은 앞선 valid recurrent output에 영향을 주지 않는다.
- Importance ratio는 수집 시점의 log probability를 사용한다. Model/controller/trajectory version 불일치를 거부한다.
- PPO forward는 `model.train()`에서 수행하고 원래 mode를 복구한다. Eval-mode로 로드한 CUDA GRU에서도 backward가 가능하도록 한다.

### Normalization options

| 값 | 통계 계산 범위 | 용도 |
|---|---|---|
| `batch` | 해당 policy의 batch 전체 valid actor tokens | 수정된 training 기본값 |
| `legacy_episode` | 각 episode의 actor segment | 기존 결과 재현 및 비교 |
| `off` | 없음 | 진단용; 안정적인 정식 대체라고 가정하지 않음 |

Episode 길이가 다르면 token 가중 방식이다. 현재 main은 길이가 같아 episode 가중과 같다. 마지막 불완전 batch도 업데이트하며 checkpoint는 항상 batch 경계에서 저장한다. Checkpoint interval보다 batch의 원자적 저장을 우선한다.

Online test는 서로 다른 world의 adaptation을 합치지 않는다. 한 world의 관측된 rollout interval만 사용하며 `evaluation.online_advantage_normalization`을 별도로 기록한다. 기본값은 기존 `legacy_episode`다. 따라서 **training batch normalization 개선과 online single-world normalization은 서로 다른 설정**이다. Test의 현재 rollout에서 미래 K return을 사용할 수 없다.

## 2. B/K 경계의 이론적 처리

Main objective:

```text
J = E[lambda_explore * sum(t=0..B-1) r_t + sum(t=B..B+K-1) r_t]
gamma = 1
```

Single은 하나의 critic을 사용해 전체 trajectory에서 GAE를 계산한다. Dual은 prefix policy와 suffix policy의 critic이 다르다. 기존 방식은 전체 trajectory의 value 배열을 연결하고 GAE를 적용한다. Lambda=1이면 critic의 중간 항들이 telescoping하여 complete Monte Carlo return에 도달하지만, lambda<1이면 suffix critic의 오차와 서로 다른 critic의 경계 추정에 prefix target이 의존한다.

**서로 다른 critic을 연결한다는 사실만으로 기존 방식이 수학적으로 틀린 것은 아니다.** 두 critic이 같은 composite policy/objective의 phase-conditioned value를 정확히 추정한다면 정상적인 Bellman/GAE 구성이다. 변경 목적은 suffix critic의 근사 오차에 대한 prefix target의 의존을 줄이는 것이다. 대신 sampled suffix return의 분산을 받아들이므로 성능 우월성은 실험으로 확인해야 한다.

수정된 training은 complete suffix를 이미 관측했으므로 다음 값을 사용한다.

```text
G_K = sum(j=0..K-1) gamma^j * r_(B+j)
V_boundary = G_K

delta_(B-1) = lambda_explore * r_(B-1) + gamma * G_K - V_explore(h_(B-1))
A_t = delta_t + gamma * lambda_GAE * A_(t+1), within prefix
```

B 안의 중간 bootstrap은 Explorer 자신의 critic이다. Suffix actor/critic target은 suffix에 대해서만 Exploiter critic으로 계산한다. 이것은 **train-only observed continuation target**이며 phase boundary를 environment termination으로 바꾸지 않는다. K가 끝나지 않은 suffix를 이 target 함수에 전달하면 거부한다.

- Lambda_GAE=1: prefix target은 전체 weighted realized return과 일치한다.
- Lambda_GAE=.95: 앞선 prefix 행동으로 전달되는 sampled continuation 효과에는 GAE attenuation이 남는다. 이 방식이 unbiased Monte Carlo gradient와 항상 같다고 주장하지 않는다.
- Corrected prefix target은 suffix critic 예측값을 바꿔도 변하지 않는다.
- Reward sigma나 mode reset은 `done`을 만들지 않는다. 실제 horizon 종료에서만 bootstrap을 끊는다.
- Discounted ablation gamma=.99는 main undiscounted objective와 구분한다.

### Normalization 반례와 gradient 해석

동일 prefix에서 K return만 높인 두 episode는 raw advantage의 크기/수준이 달라진다. Episode 정규화는 그 차이를 제거할 수 있지만 batch 정규화는 episode 간 상대 차이를 보존한다.

**동일한 action/score gradient를 가진 두 counterfactual prefix를 한 batch에 넣으면 두 contribution의 합이 상쇄될 수 있다.** 이는 정규화 실패가 아니다. 회귀 검증은 높은 return episode와 낮은 return episode의 개별 actor-gradient contribution이 반대 방향임을 검사하며, 동일 행동에 대한 합산 gradient가 반드시 양수라고 잘못 주장하지 않는다.

## 3. 재현성과 로그

`provenance.json`: git commit, dirty status, source SHA256, Python/PyTorch/CUDA/platform, config, seed, rollout 순서.

Batch checkpoint: model, optimizer, policy version, episode index, torch CPU/CUDA RNG, numpy global RNG. Environment/controller RNG는 episode seed에서 재생성한다. Batch 중 interruption은 마지막 완료 batch부터 재실행한다. Append-only 로그의 동일 episode 중복은 checkpoint 이전 재수집에서 생길 수 있으므로 episode 및 policy version으로 구분한다.

`training.jsonl`은 episode return 행과 `*_update` 행을 구분한다. Update 행은 다음을 포함한다.

- Actor/critic loss, entropy, approximate KL, clip fraction.
- Raw advantage mean/std/min/max, normalized mean/std 및 episode별 normalized mean.
- Actor/critic/shared parameter에 대한 각 loss의 gradient norm, clipping 전 합산 norm.
- Explained variance, parameter delta norm, update seconds.
- Rollout policy version, update 이후 version, episode ID 목록, transitions, optimizer steps.

Evaluation의 `adaptation_compute`에는 neural optimizer 시간/transition/update 상세와 Q TD update 수·시간·offline replay 수를 별도로 저장한다.

Gradient norm 진단은 추가 autograd 계산을 사용하므로 throughput 비용이 있다. `costs.json`에 interaction, optimized transition, optimizer step, parameter 수를 별도로 기록한다. Batch=8은 같은 episode 수에서 batch=1보다 optimizer step 수가 적으므로 단독 비교로 원인을 단정하지 않는다.

## 4. 작은 pilot과 history intervention

`meta_exploration.diagnostics`는 새 output directory만 허용한다.

Stages:

1. `easy`: single train world, constant A opponent, sigma=0. 단일-world 진단이며 main pretraining 조건을 대체하지 않는다.
2. `type1`: 여러 Type 1 train worlds, constant opponent, sigma=0.
3. `type2`, `type3`: combo task 추가.
4. `mixed_noise`: 여러 types, random/cyclic opponent, sigma=1.

Variants:

| Variant | Batch | Normalization | Dual boundary |
|---|---:|---|---|
| legacy | 1 | episode | cross critic |
| legacy_batch | 8 | episode | cross critic |
| batch_cross_critic | 8 | batch | cross critic |
| corrected | 8 | batch | observed suffix |

Pilot에서는 모든 variant가 동일한 batch-pretrained model로 시작하는지 weight equality를 assert한다. 이 때문에 `legacy`는 **historical full-study 전체 복제**가 아니라 shared pretraining 이후의 legacy main-training 대조군이다.

Train/test의 frozen reward, analytic uniform-random expectation, seed별 평균, paired hierarchical bootstrap을 저장한다. Tiny pilot은 학습 배선/안정성 진단이며 neural 일반화의 최종 검정이 아니다.

History intervention은 한 번 수집한 B prefix와 post-B snapshot에서 분기한다.

- `real`: 원래 context.
- `reward_shuffled`: 마지막 두 transition을 제외한 reward 순서를 섞고 관측 chain을 재구성.
- `action_relabelled`: 앞부분의 양쪽 action labels를 일관되게 permutation하고 이전 action/관측 chain을 재구성.
- `reset`: hidden/context를 비우되 실제 현재 observation은 동일.

마지막 두 transition을 유지해 최종 action/reward와 실제 handoff observation이 일치하도록 한다. Synthetic history가 true world에서 그럴듯한 trajectory라는 보장은 없다. 이는 representation intervention이며, factual trajectory 성능 비교와 구분한다. 변경 history를 환경에서 재실행하지 않고 fake transition을 Q replay나 PPO actor samples로 사용하지 않는다. Online 모드에서는 synthetic context가 conditioning으로만 작용한다. 작은 B에서는 perturbation 대상이 적거나 없어질 수 있다.

Ridge/linear hidden-state probe는 아직 구현하지 않았다. 후속 probe target은 실제 history의 counts/reward statistics로 제한하고 train/test world를 분리해야 한다.

## 5. Adaptation speed 정의

각 method의 실제 post-B state에서 K-step finite-horizon DP를 계산한다. Random iid, known cyclic phase, constant A opponent를 지원한다. Frozen recurrent PPO opponent는 정확한 oracle을 제공하지 않으며 undefined로 보고한다.

```text
oracle_expected_reward = V_oracle(post_B, K) / K
random_expected_reward = V_uniform_random(post_B, K) / K
absolute(t) = trailing_mean_reward(t) / oracle_expected_reward
baseline_normalized(t) = (trailing_mean_reward(t) - random_expected_reward)
                         / (oracle_expected_reward - random_expected_reward)
```

이는 stationary/long-run oracle이 아니다. Full-horizon 평균 reference를 moving window와 비교하므로 combo의 mode/parity와 window 길이가 crossing에 영향을 줄 수 있다. Default W=16은 짝수다. 두 reference 모두 실제 시작 mode와 이전 양쪽 action을 반영하며 B가 홀수여도 reset하지 않는다. Single/Q-only의 prefix가 다르면 post-B reference도 달라질 수 있다. Dual pair는 동일 reference다.

Oracle forward occupancy를 사용해 optimal finite-horizon policy의 step별 expected reward curve도 계산한다. Uniform random 기준도 exact DP/occupancy를 사용한다. Diagnostic 계산만 hidden parameters에 접근한다.

- W=16 full trailing window만 사용한다. 앞 15 steps는 `null`이다.
- Hold=8: 연속 8개 eligible evaluation steps에서 threshold를 만족해야 한다.
- 연속 moving windows는 서로 겹치므로 hold=8을 독립적인 8번의 통계 검정으로 해석하지 않는다. 이 지표는 관측 reward의 지속 crossing이며 true policy value에 대한 신뢰도 보장은 아니다.
- Time은 첫 crossing 시작점이 아니라 **유지 조건이 확인된 step**이다. 기본 earliest time=23.
- Absolute reference가 0 이하/epsilon 이하이면 undefined.
- Baseline-normalized 분모가 epsilon 이하이면 undefined.
- 범위 밖 음수/1 초과 값은 clipping하지 않는다.
- K<W이면 reference는 정의될 수 있지만 측정 가능한 moving window가 없다. AUC/final 값은 null, 이벤트는 미도달이다.
- Normalized AUC는 eligible steps의 discrete sum이다. `auc_observed_steps`, `normalized_adaptation_auc_mean`도 저장한다. 앞부분을 임의로 0으로 채우지 않는다.
- `final_window_oracle_ratio`는 각 normalization의 마지막 full window 값이다. Baseline-normalized branch에서는 그 정의에 맞는 progress 값이다.
- Cumulative oracle regret은 oracle expected cumulative reward minus 실제 sampled cumulative reward다. 가능할 때 true conditional means로 계산한 expected-regret curve도 별도 저장한다. Moving-window ratio와 혼동하지 않는다.

### Censoring와 통계

미도달은 K에서 administrative right censoring한다. 동일 K 그룹에서는 survival이 empirical nonreach curve이고 RMST[0,K]는 `mean(min(T,K))`다. 미도달 world를 제외한 reached-only 평균을 사용하지 않는다.

각 normalization/threshold별 reach@K, RMST, B+RMST를 보고한다. Undefined denominator world는 nonreacher로 넣지 않고 `n_undefined`로 별도 집계한다. Seed를 바깥 단위, world를 안쪽 단위로 resampling한다. Paired comparison은 동일 seed/world의 difference를 먼저 구성한 뒤 같은 계층 bootstrap을 적용한다. 한 seed의 interval은 seed 간 변동을 검증하지 못한다.

그래프:

- A: normalized curve와 hierarchical CI, 25/50/75/90% lines.
- B: threshold별 cumulative reach curve/CI, survival JSON.
- C: normalized AUC mean, 가장 높은 threshold의 reach@K/RMST. 모든 threshold 수치는 CSV/JSON에 저장.
- D: B별 exploitation RMST와 B를 포함한 total interaction RMST.

서로 다른 training variant를 한 root에서 묶어 동일 world를 중복 집계하면 거부한다. Variant별로 별도 aggregate한다.

## 6. 실행 명령

GPU 서버에서 기존 setup을 완료한 경우:

```bash
bash scripts/setup_gpu.sh
source .venv/bin/activate
python -m pytest tests/meta/test_credit_speed.py -q
```

CUDA가 없으면 CUDA test는 skip하며, 이것을 CUDA 통과라고 보고하지 않는다. GPU 서버에서는 아래 테스트가 실제 실행되는지 확인한다.

```bash
python -m pytest tests/meta/test_credit_speed.py -q -k 'backward'
```

작은 CPU pilot:

```bash
python -m meta_exploration.diagnostics \
  --config configs/meta/diagnostic_pilot.json \
  --out runs/credit_easy_new \
  --stages easy --variants legacy legacy_batch batch_cross_critic corrected \
  --seeds 0 1 2 --device cpu

python -m meta_exploration.diagnostics \
  --config configs/meta/diagnostic_pilot.json \
  --out runs/credit_staged_new \
  --stages type1 type2 type3 mixed_noise --variants corrected \
  --seeds 0 1 2 --device cpu
```

기존 pilot checkpoint로 학습 없이 B=8/16/32/64, K=128 평가만 실행:

```bash
python -m meta_exploration.diagnostics \
  --checkpoint-root runs/credit_easy_new/easy/corrected \
  --out runs/credit_speed_K128_new --device cpu
```

원본 checkpoint는 읽기만 하고 평가용 새 경로에 복사한다. 원래 training config와 checkpoint hash를 별도 기록한다. 학습 B/K와 평가 B/K가 다를 수 있으므로 budget generalization임을 명시한다.

실제 기록으로 speed graph 생성:

```bash
python -m meta_exploration.adaptation \
  --root runs/credit_easy_new/easy/corrected \
  --out runs/credit_easy_new/speed_corrected --replicates 1000
```

서버에 기존 raw trajectory JSONL이 있는 경우 별도 경로로 backfill:

```bash
python -m meta_exploration.adaptation \
  --root runs/full_study --out runs/full_study_speed_backfill \
  --backfill-raw --replicates 1000
```

원본 tree 안에 backfill output을 만들거나 기존 output을 덮어쓰지 못하게 한다. JSONL이 없으면 `backfill_audit.json`에 missing files를 기록하고 speed를 복원하지 않는다. 원본 `aggregated.csv`만으로 crossing 시간을 만들 수 없다.

검증 후 사용자가 전체 재실험을 선택할 때의 명령(이번 작업에서는 실행하지 않음):

```bash
python -m meta_exploration launch \
  --config configs/meta/corrected.json --out runs/full_study_corrected_v1 \
  --regimes A1 A2 A3 B --seeds 0 1 2 3 4 \
  --require-gpu --workers-per-gpu 2 --frozen-batch-size 32

python -m meta_exploration.adaptation \
  --root runs/full_study_corrected_v1 \
  --out runs/full_study_corrected_speed_v1 --replicates 1000
```

GPU 처리량 개선은 위 학습 비교와 별도로 benchmark한다. Batch PPO가 더 빠르거나 느린 사실만으로 학습 신뢰성 개선을 주장하지 않는다.

## 7. 산출물 구조

```text
runs/<new-study>/
  <stage>/<variant>/seed_<seed>/
    config.json, identity.json, provenance.json, costs.json
    checkpoints/{shared_exploitation_pretrain,training_resume,trained}.pt
    training.jsonl
    training_histories/             # observed-only trajectories
    diagnostics/                   # hidden metadata; agent input과 분리
    evaluation/metrics.json        # world별 adaptation_speed 포함
    evaluation/<mode>_B<B>_K<K>/*.jsonl
  pilot_metrics.json
  summary.json                     # seed별 train/test reward 및 CI
  paired_comparisons.json
  train_test_diagnostics.json
  history_interventions.json
  budgets.json
  <stage>_learning.{png,pdf}
```

## 8. Phase 3~6: 계획 및 인터페이스

이 단계들은 현재 구현했다고 주장하지 않는다. Phase 1/2 결과를 먼저 검토한다.

### Phase 3: updater와 memory 분리

- GRU+TD/Q-head: 기존 `Transition`을 replay 단위로 사용. Episode-aware sequence replay, burn-in context, target network, terminal mask, target update interval 필요. Behavior policy version과 off-policy replay를 PPO와 혼용하지 않는다.
- Observed statistics+MLP/PPO: `Observation.vector`를 직접 바꾸기보다 별도 observed-history encoder interface를 둔다. Mode/joint action/ordered pair별 count, mean, variance, unobserved mask를 생성한다. True bonus는 사용하지 않는다.
- Tabular TD: 기존 `QLearner.act/update/replay` 유지.
- 각각 replay 횟수, target updates, parameters, transitions, optimizer steps를 별도로 기록한다. Architecture와 updater가 함께 바뀌는 비교를 단일요인 비교라고 부르지 않는다.

### Phase 4: explorer 전략

`HistoryCollector`의 random/repeat/coverage, `QLearner`, `Controller`는 이미 `act(observation)` 인터페이스를 공유한다. `history_interventions(..., strategy=...)`는 이 generator를 사용할 수 있다. 정식 전략 비교 runner는 generator를 바꾼 B history와 각 post-B snapshot을 저장하고 동일 exploiter weights/learning rule로 평가해야 한다. 서로 다른 prefix 행동 때문에 post-B state가 달라지는 것은 허용하되, 동일 prefix를 비교하는 downstream branches는 같은 snapshot을 사용한다.

Time-to-first-good-combo, base/bonus contribution, conditional utilization 및 strategy 간 paired speed 보고를 후속 runner에 추가한다. 단순 discovery 증대와 exploitation 개선을 구분한다.

### Phase 5: opponent 및 exposure

Constant/random/cyclic 환경과 exact oracle은 구현했다. 정식 causal comparison runner는 같은 `World` 객체를 유지하고 `make_game`의 opponent 설정만 바꾼다. 현재 pilot stages는 난이도를 순차적으로 추가한 것이며 모든 stage 차이가 opponent 단일요인 실험은 아니다.

Mixed exposure 비교는 (a) 같은 총 interaction, (b) 같은 type별 exposure 두 패널을 별도로 구성한다. 후자는 mixed total cost가 증가하므로 비용을 표시한다. 같은 총 episode에서 mixed가 type당 1/3을 본 결과만으로 representation interference를 결론 내리지 않는다.

### Phase 6: architecture

GRU 64/128 → LSTM → explicit statistics+recurrent context → observed reward/next-opponent auxiliary prediction 순으로 검토한다. Architecture별 Single/Dual은 동일 pretrained checkpoint에서 시작한다. 현재 상대 action을 actor에 미리 제공하거나 hidden reward를 auxiliary label로 사용하지 않는다. Transformer는 장기 retrieval 병목이 확인된 후에만 검토한다.
