# 실제 MaleCNS 순환 배선에서 시간축 PC-ALM 학습

## 연구 질문과 데이터

**실제 MaleCNS 부분 회로의 순환·층내·건너뛰기 연결을 복원한 시간축 신경망에서, PC-ALM이 움직임을 학습하고 동일 구조의 PC보다 개선되는가?**

실제 MaleCNS v1.0에서 추출한 132개 뉴런, 2,663개 연결을 사용했다. 대조 구조는 같은 뉴런의 인접 순방향 연결 1,001개다. 실제 손글씨 이미지(load_digits)를 움직여 6프레임 입력을 만들었다. **배선과 원본 이미지는 실제 자료지만, 움직임·정답은 합성**이며 자연 영상이나 초파리 신경활동 검증은 아니다. 픽셀-뉴런 대응과 시간상수도 공학적 가정이다.

## 실행 결과

10개 시드 × 2개 연결구조 × 3개 학습법 = **60회 학습**을 완료했다. 각 20에포크, PC/ALM 추론 8회. 수치 발산 0회. 수학·실제 연결 수·이미지 분리 테스트 3개 통과.

| 구조 | 방법 | 평균 시험 MSE ↓ | 방향 정확도 ↑ |
|---|---|---:|---:|
| 순방향 1,001개 연결 | BP | 0.51209 | 24.61% |
| 순방향 1,001개 연결 | PC | 0.51593 | 23.59% |
| 순방향 1,001개 연결 | PC-ALM | 0.51838 | 23.59% |
| 전체 2,663개 연결 | BP | 0.51518 | 25.63% |
| 전체 2,663개 연결 | PC | 0.52043 | 26.95% |
| 전체 2,663개 연결 | PC-ALM | 0.52224 | 27.58% |

**제로 출력 기준 MSE는 0.5, 무작위 방향 기준 정확도는 25%다. 이번 설정은 BP까지 포함해 유용한 움직임 일반화를 입증하지 못했다.** 전체 회로 ALM의 정확도가 조금 높아 보여도 이를 성공이나 우위로 해석해서는 안 된다.

동일 회로 내 주지표 MSE에서는 ALM이 PC보다 나빴다. 두 비교 Holm p=0.00390625. 하지만 전체 모델이 기준선 근처이므로 이 작은 차이보다 과제/입력/모델 적합성 문제가 먼저다. [전체 표와 신뢰구간](results/RESULTS.md).

초기 결과 확인 후 추가한 고정 영상 매칭 sanity control은 시험 정확도 100%, MSE 0이었다(학습·튜닝 없음). 이 대조는 알려진 이동 생성 규칙을 이용하므로 일반 영상 성능을 뜻하지 않는다. 입력에 방향 정보가 존재한다는 점만 확인한다. 신경망 설정을 이 결과에 맞춰 바꾸지는 않았다. [기준선 결과](results/motion_baseline.csv).

## 해석과 다음 단계

현재 결과로 초파리 배선의 무용성이나 PC-ALM의 일반적 실패를 주장할 수 없다. 생물학적 위치와 무관한 입력 어댑터, 6프레임에 제한된 경로 전달, 고정 시간상수, 작은 학습 표본, 튜닝하지 않은 설정을 먼저 검증해야 한다.

다음 실험은 별도 개발 시드에서 다음 순서로 진행해야 한다: (1) 일반 시간축 모델이 이 과제를 학습하는지 확인, (2) 입력/출력 대응과 관측 길이 개선, (3) 동일 튜닝 기회를 준 BP·PC·ALM 비교, (4) 실제 영상/흐름 데이터와 추가 부분 회로로 확장. 연결 수가 달라 구조 간 비교는 용량 차이에 혼동되므로 matched-edge 대조가 필요하다.

## 실행

저장소 루트에서 Python 3.11+:

```bash
python -m pip install -r experiments/temporal_connectome/requirements.txt
OPENBLAS_NUM_THREADS=1 python -m pytest -q experiments/temporal_connectome/test_temporal.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/temporal_connectome/experiment.py --output results/temporal-reproduction
```

출력 폴더가 있으면 중단한다. 매 실행 후 기록을 저장하지만 자동 재개는 지원하지 않는다. sanity control은 `python experiments/temporal_connectome/motion_baseline.py`이며, 고정된 `results/motion_baseline.csv`를 다시 생성한다.

설계·수식·통계·자료 한계: [PROTOCOL.md](PROTOCOL.md). 실행 코드와 자료 체크섬: [manifest](results/manifest.json). 개별 결과: [trials](results/trials.csv). 학습 과정: [curves](results/curves.csv). 원자료 출처는 이전 credit_rebuild manifest 및 공식 MaleCNS 다운로드 문서를 참조한다. 원자료/파생 연결의 이용 조건은 코드 MIT와 구분한다.
