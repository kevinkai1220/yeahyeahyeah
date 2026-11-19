# 구현 검증

실행 환경: Python 3.11, PyTorch 2.15.0+cpu, NumPy 2.4.6.

- `python -m unittest discover -s tests`: 14개 테스트 통과.
- `runs/smoke`: T=12, K=2/5/6, GRU=128, 모델별 PPO 2회 업데이트. Opponent 사전학습 → single/dual 학습 → checkpoint 로드 → 평가 완료. 평가 rollout 180개, 조건별 요약 45개, curve row 540개.
- `runs/default_shape_check`: T=100, K=10/25/50, GRU=128, 업데이트당 8 tasks × 8 rollouts, PPO epoch=4, minibatch=16. 모델별 1회 업데이트와 평가 완료. 평가 rollout 90개, curve row 4,500개. 실행 중 기록된 안정성 경고 없음.
- Oracle의 Random/Fixed 최적 기대 return은 작은 horizon의 완전탐색과 일치함을 테스트.
- Exploitation loss가 exploration observation 구간까지 역전파됨을 테스트.
- Rollout 수집 중 가중치 고정, rollout 간 hidden 초기화, 단계 전환 시 recurrent replay 일치를 테스트.

기본 1,000회 업데이트 실험과 최소 3개 training seed의 성능 비교는 실행하지 않았습니다. 위 결과는 코드와 실행 경로 검증이며, 학습 수렴이나 strategic exploration의 우위를 입증하지 않습니다. 실행한 config와 모델, 원시 결과는 각 `runs/` 디렉터리에 보관했습니다.

## Lambda / V-trace 비교 추가 검증

- 기존 14개 + 신규 6개, 총 20개 단위 테스트 통과.
- λ=0/0.5/1 return target, 홀수 K 경계, V-trace on-policy MC 일치, 손계산 off-policy target/advantage 검증.
- 배치 rollout과 순차 rollout의 action/reward/log-probability 일치(Random/Fixed/PPO 상대) 검증.
- 세 dual 평가 조건과 짝수/홀수 K에서 기존 평가 지표와 새 배치 평가 지표 일치 검증.
- Replay capacity, target-only V-trace head 업데이트, exploration-only PPO head 업데이트 검증.
- `runs/lambda_offpolicy_smoke`: 6조건 × 3 K = 18개 모델 각 2 updates 완료. 평가 기록 324개, 기록된 안정성 경고 0건.
- Smoke 결과로 비교표 생성 및 task paired bootstrap 집계 경로 검증.
- 본 실험 진행 및 완료 상태는 `runs/lambda_offpolicy_seed0/status.json`을 참조. 최종 성능 보고서는 해당 디렉터리의 `report.md`에 생성된다.
- 추가 회귀 검증: 같은 Adam optimizer에 exploration momentum을 먼저 만든 다음 V-trace를 수행해 exploration actor/critic이 정확히 유지되는지 확인. 이어서 target momentum이 있는 상태에서 exploration-only PPO를 수행해 target actor/critic이 정확히 유지되는지 확인. 수정 후 총 20개 테스트 통과.
- 최종 비교에는 `runs/lambda_offpolicy_seed0_corrected`만 사용한다. 첫 실행의 V-trace 결과는 `OFFPOLICY_SUPERSEDED.md`에 사유를 기록하고 보관했다.
- `runs/lambda_offpolicy_seed0_corrected` 완료: 최종 비교 모델 18개 모두 100 updates, 6조건 × 46,080 = 276,480 평가 기록. 조건별 45개 그룹 × 1,024 기록과 키 중복 없음을 확인. 안정성 경고 0건. 재사용한 PPO checkpoint 및 원시 평가 파일의 SHA-256이 최초 실행과 일치함을 확인. `verification.json`에 검증 결과 저장.

## 발견 시점 / 학습 추적 검증

- 총 23개 단위 테스트 통과. Trace 유무에 따른 action/reward/RNG 동일성, 홀수 K의 발견/재활용 경계, 미발견 검열, 발견 이후 lag 정렬을 확인했다.
- Tracking 활성/비활성의 동일 seed V-trace 학습에서 모든 최종 모델 가중치가 정확히 동일함을 확인했다. update 0/1/2 checkpoint 보존도 확인했다.
- `runs/tracking_smoke`: V-trace λ=0.5, K=50, 2 updates, 진단 task 2개. Checkpoint 3개, learning-curve 행 9개, round trace 5,400개, 최종 평가 18개 기록까지 완료. 확률 벡터 합과 trace 수 검증.
- 본 tracking 장기 학습은 시작하지 않았다. 사용자 tmux 실행용 runner는 `scripts/run_tracked.py`이다.
