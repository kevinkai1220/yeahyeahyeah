# MARL Strategic Exploration v2

원본 `../marl_strategic_exploration`은 수정하지 않았습니다. 기존 실험 설명은 README_V1.md에 보존했습니다. 기존 runs/가상환경/git 이력은 복사하지 않았습니다.

## 변경 사항

- K별 exploit 사전학습 → frozen exploit을 사용하는 exploration 학습. 사전학습에서는 shared encoder/GRU와 target actor/critic만 업데이트합니다. 이후에는 encoder/GRU와 target을 고정하고 exploration actor/critic만 업데이트합니다. 따라서 탐색 표현력은 사전학습된 GRU에 제한됩니다.
- 사전학습 rollout의 75%에서 K 이전 임의의 짝수 시점에 실제 combo 두 행동을 실행합니다. 나머지 prefix 행동은 random입니다. 관측에는 실제 행동·보상만 들어가며 combo metadata는 주입 행동 선택에만 사용됩니다. PPO loss는 K 이후에만 적용하고 전체 history를 replay하여 GRU를 학습합니다. 평가와 본 탐색 학습에는 정답 주입이 없습니다.
- 게임 풀 256 → 8192, 업데이트당 64 games × 16 rollouts. 게임 풀을 shuffle 순회하여 누락 없이 반복 경험합니다. opponent 5000 updates, single 및 exploration 10000 updates, K별 exploit 사전학습 10000 updates. 이는 학습 예산이며 수렴이나 foundation 성능을 보장하는 기준은 아닙니다.
- 기존 generator의 reward scale/interaction/combo 변형을 더 많이 샘플링합니다. transition은 기존 alternating R1/R2로 유지됩니다.
- 환경 보상 indexing, 관측, action sampling, recurrent rollout, MC return 및 PPO minibatch를 지정 device에 유지합니다. 독립 rollout을 병렬 처리하고 시간축은 full-sequence GRU replay합니다. FP32, minibatch 256; per-step NumPy 왕복이 없습니다. 지표와 checkpoint 저장 시에는 CPU 동기화합니다.

## 실행

CUDA 지원 PyTorch가 설치된 Python 환경에서:

```bash
cd /home/kevinlees1220/marl_strategic_exploration_v2
python -m marl.cli run --config configs/default.json --output runs/v2_seed0
```

기본 device는 cuda이며 GPU가 없으면 오류를 냅니다. CPU로 조용히 fallback하지 않습니다. 작은 기능 검증:

```bash
../marl_strategic_exploration/.venv/bin/python -m unittest discover -s tests -v
../marl_strategic_exploration/.venv/bin/python -m marl.cli run --config configs/v2_smoke.json --output runs/new_smoke
```

`dual_kK_exploit_pretrain.jsonl/.pt`에 사전학습 지표/최종 checkpoint를, `dual_kK.jsonl/.pt`에 탐색 학습 결과를 저장합니다. 기존 평가 코드와 checkpoint 형식을 사용합니다. 평가에서는 모델을 CPU에 로드합니다. 사전학습의 reward 지표는 주입 이력 분포에서 측정되므로 최종 held-out 성능과 구분해야 합니다. optimizer resume, 자동 수렴 판정 및 legacy V-trace/tracking의 GPU 지원은 포함하지 않습니다. 기존 scripts는 v1 실험 재현용이며 v2 실행에는 위 CLI를 사용하세요.

## Google Colab

`notebooks/marl_v2_colab.ipynb`를 Colab에 업로드하고 GPU 런타임에서 위에서 아래로 실행합니다. 노트북에 소스를 포함했으므로 별도 코드 업로드가 필요 없습니다. Drive 연결, GPU smoke, pilot/full 설정, 단계별 학습·평가 및 곡선 표시를 제공합니다. 기본은 짧은 pilot입니다. 소스 수정 후 `python scripts/build_colab_notebook.py`로 포함된 소스를 갱신하세요. 중단된 학습의 optimizer resume는 지원하지 않습니다.
