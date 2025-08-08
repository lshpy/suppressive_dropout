# Suppressive Dropout (CNN, one-spot)

- CIFAR-10에서 **중간 stage 한 지점**에만 Suppressive Dropout 적용.
- 점수식: S_j ∝ (Σ_{k≠j} x_k) * x_j^2 / (1 + b Σ_i x_i^2)^{c+1}

## Run
```bash
pip install -r requirements.txt
python main.py --use_sdrop --drop_ratio 0.2 --b 1.0 --c 1.0 --epochs 30
# 베이스라인
python main.py --epochs 30


---

원하면 **채널 대신 공간 위치(H×W) 단위 억제**나, **여러 지점 삽입/어블레이션 스위치**도 바로 확장해줄게.  
일단 이 버전으로 돌려보고 로그 올라오면, `drop_ratio / b / c`랑 “적용 위치” 튜닝 들어가자.
