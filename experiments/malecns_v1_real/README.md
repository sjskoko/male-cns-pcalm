# MaleCNS v1.0 실제 토폴로지 반복실험

이 폴더는 공식 MaleCNS v1.0의 연결표·뉴런 주석·신경전달물질 예측을 사용한 첫 반복실험을 고정한다. 원본 약 1 GB 파일은 포함하지 않으며, `selection-report.json`에 파일별 SHA-256을 기록했다.

## 질문과 범위

검증 질문은 다음과 같다.

> 실제 MaleCNS 시각계→중앙뇌→하행성 회로가 같은 크기와 edge budget의 재배선 회로보다 PC-ALM으로 합성 시각–행동 규칙을 더 잘 학습하는가?

**실제인 것:** 뉴런 ID, 세포 superclass/type, 방향성 연결, 연결 강도, 신경전달물질 예측.

**합성인 것:** 좌우 시각 자극, `turn_left/forward/turn_right/stop` 정답, 정적 지도학습 목적함수. 따라서 이 결과는 초파리 행동의 재현 결과가 아니다.

## 회로 선별

MaleCNS 주석의 exact superclass를 사용해 다음 실제 경로를 찾았다.

1. `visual_projection`
2. `cb_intrinsic`
3. 별도의 `cb_intrinsic`
4. `descending_neuron`

weight 5 이상인 3-edge 경로를 찾은 뒤 내부 edge를 `log1p(입력 지지) × log1p(edge weight) × log1p(출력 지지)`로 순위화했다. 입력층은 soma side 기준 좌·우 16개씩 골랐고, 연결이 끊긴 노드는 제거했다. 같은 뉴런이 두 레이어에 중복되지 않도록 강제했다.

| 항목 | 값 |
|---|---:|
| 공식 연결 행 | 151,856,684 |
| weight ≥ 5 연결 | 7,622,864 |
| 최종 뉴런 | 132 (`32 → 40 → 36 → 24`) |
| 최종 인접 순방향 edge | 1,001 (`208 → 620 → 173`) |
| 입력 좌/우 | 16 / 16 |
| 알려진 흥분/억제/불명 sign 뉴런 | 87 / 23 / 22 |

`input_position=-0.58/+0.58`은 soma side를 이용한 좌우 proxy다. 실제 receptive field 또는 eyemap 좌표가 아니다. Acetylcholine은 `+1`, GABA와 histamine은 `-1`, glutamate와 기타 불확실한 전달물질은 `0`으로 보수적으로 처리했다.

선택된 뉴런 사이에는 인접 순방향 외 연결도 1,662개 있었다. feed-forward 투영은 전체 2,663개 중 1,001개(37.6%)만 보존하고, intra-layer 1,059개, feedback 392개, skip-forward 211개를 제외한다. 이는 현재 결과의 가장 큰 구조적 한계다.

## 반복실험

- 날짜: 2026-09-17
- 장치: CPU
- 반복: seed 10개
- 조건: 토폴로지 3종 × 학습법 3종 = 총 90회
- 데이터: train 512, test 256, 4개 균형 class
- 학습: 12 epoch, batch 32
- PC/PC-ALM inference budget: 8

### 결과

| 토폴로지 | 학습법 | 최종 정확도 | Accuracy AUC | BP-gradient cosine |
|---|---|---:|---:|---:|
| 실제 MaleCNS | BP | 91.4% ± 14.1% | 0.619 | 1.000 |
| 실제 MaleCNS | PC | 94.1% ± 8.0% | 0.622 | 0.585 |
| 실제 MaleCNS | PC-ALM | **93.6% ± 9.1%** | 0.619 | 0.601 |
| 차수 보존 재배선 | PC-ALM | **94.1% ± 12.1%** | 0.641 | 0.612 |
| 무작위 희소 | PC-ALM | **96.8% ± 6.7%** | 0.661 | 0.649 |

PC-ALM의 시드별 짝지은 비교에서 실제 배선은 차수 보존 재배선보다 평균 0.55%p 낮았고 실제 배선 승률은 20%였다. 무작위 희소 회로보다 평균 3.24%p 낮았으며 승률도 20%였다. 실제 배선에서 PC-ALM은 PC보다 gradient cosine이 약 0.016 높았지만, 최종 정확도는 약 0.51%p 낮았다.

따라서 **이 설정에서는 실제 MaleCNS 토폴로지의 학습상 이점을 관찰하지 못했다.** 결과는 PC-ALM 코드가 실제 배선에서 정상 실행됨을 보여주지만, 실제 배선이 이 합성 과제에 특별히 맞는다는 가설은 지지하지 않는다. seed가 10개뿐이고 분산이 크므로 통계적 유의성을 주장하지 않는다.

가능한 이유는 (1) soma side가 실제 시각 receptive field가 아니고, (2) 행동 라벨이 생물 데이터가 아니며, (3) 순환·피드백 연결의 62.4%를 feed-forward 투영에서 버렸기 때문이다. 다음 실험은 실제 eyemap 좌표와 행동/신경활동 표적, recurrent PC-ALM을 사용해야 한다.

## 재현

저장소 루트에서 다음 한 줄로 다운로드, 체크섬 검증, 회로 선별, 그래프 투영, 10-seed 실행을 반복할 수 있다.

```bash
bash scripts/run_malecns_v1_real.sh
```

세부 파일:

- `assignments.csv`: 선택된 뉴런과 레이어, sign, 입력 proxy
- `selection-report.json`: 선택 규칙, 후보 수, 원본 체크섬
- `projection-report.json`: 보존·제외된 edge 집계
- `results/report.md`: 조건별 요약
- `results/paired_effects.csv`: 같은 seed의 실제 배선−대조군 차이
- `results/trials.csv`: 90개 trial 결과
- `results/metrics.csv`: epoch별 지표
- `results/result.json`: 전체 설정·결과·gradient 진단

## 데이터 출처와 라이선스

원자료는 [MaleCNS v1.0 공식 다운로드](https://male-cns.janelia.org/download/)에서 받으며 CC BY로 제공된다. 데이터셋의 연구 배경은 [Male CNS Connectome 프로젝트](https://male-cns.janelia.org/)와 MaleCNS 논문을 인용해야 한다. 이 폴더의 `assignments.csv`와 JSON 보고서는 MaleCNS v1.0의 소규모 파생물이다.
