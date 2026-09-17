# 시각–운동 토폴로지 × 학습법 실험

이 실험의 목적은 다음 가설을 통제된 조건에서 검증하는 것이다.

> 실제 MaleCNS 부분 그래프는 동일한 레이어 크기를 가진 차수 보존 재배선 및 무작위 희소 그래프보다 PC-ALM 학습에서 높은 표본 효율과 안정적인 credit propagation을 보이는가?

## 비교 구조

모든 조건은 같은 입력, 정답, train/test split, 출력 차원과 seed를 사용한다. 바뀌는 것은 토폴로지와 학습법뿐이다.

| 축 | 조건 |
|---|---|
| 토폴로지 | `native`, `degree_preserving`, `random` |
| 학습법 | `bp`, `pc`, `pcalm` |
| 기본 반복 | seed 5개 |

`degree_preserving`은 각 뉴런의 in/out degree를 유지한 채 edge endpoint를 교환한다. `random`은 레이어 크기와 총 edge 수만 유지한다. PC와 PC-ALM에서는 settled state를 detach한 뒤 각 투영의 로컬 제약항을 따로 미분한다.

## 가상 실험

MaleCNS 원본 데이터 없이 전체 흐름을 검증할 수 있다.

```bash
uv sync --extra dev
uv run flypcalm benchmark \
  --config configs/benchmarks/virtual_visual_motor.yaml \
  --output-dir results/virtual-visual-motor
```

이 설정은 좌우 시야 위치가 가까운 뉴런끼리 더 자주 연결되는 synthetic retinotopic graph를 만든다. 네 행동은 다음 규칙으로 생성된다.

| 시각 패턴 | 행동 |
|---|---|
| 오른쪽의 강한 움직임/장애물 | `turn_left` |
| 낮고 대칭적인 움직임 | `forward` |
| 왼쪽의 강한 움직임/장애물 | `turn_right` |
| 높은 양안 looming | `stop` |

과제 라벨은 네트워크나 토폴로지로부터 만들지 않는다. 따라서 모든 조건이 정확히 같은 외부 과제를 학습한다. 클래스 수도 균형을 맞춘다.

이 가상 결과는 코드와 실험 설계의 검증일 뿐, MaleCNS 또는 초파리 행동에 관한 증거가 아니다. `report.md`에도 이 경고가 자동으로 들어간다.

통계 비교용 가상 실행은 다음 설정을 사용한다.

```bash
uv run flypcalm benchmark \
  --config configs/benchmarks/virtual_visual_motor_multiseed.yaml \
  --output-dir results/virtual-visual-motor-multiseed
```

## 출력

실행 결과 폴더에는 다음 파일이 생긴다.

| 파일 | 내용 |
|---|---|
| `report.md` | 사람이 읽는 비교표와 해석 경고 |
| `aggregate.csv` | topology×method별 seed 평균과 표준편차 |
| `trials.csv` | 각 seed/조건의 최종 결과, accuracy AUC, 80% 도달 epoch |
| `metrics.csv` | epoch별 loss, accuracy, residual, dual norm |
| `result.json` | 설정, 세부 gradient 지표를 포함한 전체 결과 |

`test_accuracy_auc`는 epoch별 test accuracy의 평균으로, 같은 epoch budget에서 학습 속도를 비교하는 간단한 표본 효율 지표다. `epochs_to_80pct`는 처음으로 80%를 넘은 epoch이며 도달하지 못하면 비어 있다. `gradient_cosine_before/after`는 readout을 제외한 희소 connectome projection에서 해당 학습법의 update와 end-to-end BP gradient의 cosine similarity다. BP 조건은 구현 검산상 1에 가까워야 한다.

## 실제 MaleCNS 실험으로 전환

### 1. 뉴런 배정표

`assignments.csv`에 `body_id`, `layer`, `module`, `sign`을 넣는다. 시각 입력층 뉴런에는 선택 열 `input_position`도 넣는다. 값은 왼쪽 시야 `-1`, 정면 `0`, 오른쪽 시야 `+1` 범위로 정규화한다. 이 열을 사용하면 가상 과제의 좌우 자극이 실제로 지정한 입력 뉴런에 매핑된다.

`input_position`이 없으면 코드가 뉴런 ID 정렬 순서를 이용한 proxy 위치를 사용하고 결과에 `input_layout=index_proxy`를 기록한다. 이는 최종 연구 결과에 사용하면 안 된다.

### 2. 그래프 생성

```bash
uv run flypcalm prepare \
  --edges data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather \
  --assignments data/assignments/visual-motor.csv \
  --min-weight 5 \
  --output data/processed/malecns-visual-motor.pt \
  --report results/projection-report.json
```

`projection-report.json`에서 피드백·건너뛰기·레이어 내부 연결의 손실 비율을 먼저 확인한다. 보존된 edge가 지나치게 적으면 레이어 배정부터 수정한다.

### 3. 5-seed 벤치마크

```bash
uv run flypcalm benchmark \
  --config configs/benchmarks/malecns_visual_motor.yaml \
  --output-dir results/malecns-visual-motor
```

실제 MaleCNS 결과에서 가장 먼저 볼 비교는 `native+pcalm`과 `degree_preserving+pcalm`이다. 단순 `native` 대 `random` 차이만으로 실제 배선의 장점을 주장하면 안 된다.

여기서도 과제 자극과 행동 정답은 synthetic proxy다. 따라서 MaleCNS 토폴로지가 이 과제의 국소 학습에 미친 영향은 비교할 수 있지만, 실제 초파리가 같은 행동을 학습했다거나 생물 행동을 재현했다고 주장할 수는 없다.

## 해석 기준

다음 결과가 여러 seed에서 함께 나타날 때에만 토폴로지 이점을 주장할 근거가 생긴다.

1. `native+pcalm`의 test accuracy 또는 표본 효율이 `degree_preserving+pcalm`보다 높다.
2. native graph에서 PC-ALM의 BP-gradient cosine이 표준 PC보다 높거나 깊이에 따른 감소가 작다.
3. 이 차이가 특정 초기화 한 개가 아니라 seed 반복에서 유지된다.
4. edge 수, 출력 차원, 데이터와 학습 budget이 모두 동일하다.

가상 smoke 설정은 seed가 하나이고 epoch도 적다. 그 결과의 조건별 순위는 해석하지 않는다. 또한 이 정적 과제는 temporal/RL credit assignment나 실제 초파리 행동을 검증하지 않는다.
