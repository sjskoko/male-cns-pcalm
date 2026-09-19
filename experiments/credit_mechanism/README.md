# 방향이 맞아도 학습이 좋아지지 않는 이유

## 연구 아이디어

가제: **Direction Is Not Descent: Optimizer and Constraint Effects in Connectome-Constrained Predictive Coding**.

실제 MaleCNS 배선에서 PC-ALM이 잘 학습하려면 무엇이 필요한가? 기울기의 방향만 평가하던 관점에서 벗어나 **방향 → 크기 → 옵티마이저 → 부호 제약 후 실제 이동**을 분리한다. 방향과 크기를 서로 교환하는 진단용 개입으로 원인을 좁힌 뒤, 국소적으로 구현 가능한 dual 감쇠를 개선 후보로 검증한다.

이는 SCI/SCIE 투고를 목표로 발전시킬 연구 설계이며, 신규성 검증이나 게재 수준의 증거가 완성됐다는 뜻은 아니다. 논문 수준으로 추가할 항목과 반증 기준은 [PROTOCOL.md](PROTOCOL.md)에 명시했다.

## 실행된 예비 결과

2개 과제 × Adam/SGD × 2개 시드 × 6개 방법, 48회 실행. 10에포크, 추론 8회. 수학·개입·제약 테스트 3개 통과. [전체 결과](pilot_verified/RESULTS.md), [개별 실행](pilot_verified/trials.csv), [학습 중 진단](pilot_verified/diagnostics.csv).

- 비선형 과제+SGD에서 ALM의 방향을 유지하고 BP의 층별 크기를 빌리면 평균 MSE가 0.39819에서 0.26784로 낮아졌다. 반대로 BP 방향에 ALM 크기를 쓰면 0.38775였다.
- 같은 개입이 Adam에서는 0.07499에서 0.08081로 오히려 악화됐다. 따라서 단순히 크기를 보정하면 해결된다는 주장은 지지되지 않는다.
- dual 감쇠는 Adam에서 작은 수치적 개선을 보였지만 SGD에서는 악화됐다. 개선 방법으로 채택할 근거는 아직 부족하다.

**각 조건 시드 2개인 탐색 결과**다. 통계적 유의성이나 일반적 우위를 주장하지 않는다. 크기 조정에는 BP 정보가 들어가므로 그 개선을 국소 학습 알고리즘의 성과로 소개하면 안 된다. 이전 60에포크 연구와 설정이 달라 직접 순위를 비교하지 않는다.

## 재현

저장소 루트, Python 3.11+:

```bash
python -m pip install -r experiments/credit_rebuild/requirements.txt
python -m pytest -q experiments/credit_mechanism/test_mechanism.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/credit_mechanism/run.py --profile pilot --output results/mechanism-pilot
```

실행 가능한 확대 탐색 프로필(1,440회, 이번에 미실행):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/credit_mechanism/run.py --profile full --output results/mechanism-full
```

출력 폴더가 이미 있으면 중단하며 덮어쓰지 않는다. 각 실행마다 CSV를 저장하지만 자동 재개 기능은 없다. 결과·진단·환경 및 코드/원회로 체크섬을 기록한다. full도 고정 학습률 탐색이며, 공정한 튜닝을 거친 확증 실험을 대체하지 않는다.

## 제출 전 필요한 추가 작업

독립 개발 시드에서 동일 튜닝 기회를 제공하고, 새 평가 시드로 검증해야 한다. 여러 실제 부분 회로, 부호 제약 제거 대조, 외부 과제, 진단 오버헤드를 제외한 계산량 비교, 다중검정 보정이 필요하다. 이 확장들은 현재 구현 범위 밖이다. 연구의 기여 후보는 새로운 초파리 기반 모델 자체가 아니라 **유한 추론에서 학습 신호가 실제 성능으로 이어지는 조건의 규명**이다.
