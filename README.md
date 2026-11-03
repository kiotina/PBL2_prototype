# Real-only Diffusion 딥보이스 탐지 (PBL2 prototype)

진짜 음성(bona fide)만으로 DDPM을 학습하고, 노이즈 예측 오차를 이상 점수로 사용해
ASVspoof 2019 LA의 가짜 음성(spoof)을 탐지하는 최소 모델입니다.

## 폴더 구조

```
PBL2_prototype/
├─ src/
│   ├─ config.py        경로와 모든 하이퍼파라미터
│   ├─ preprocess.py    음성 → log-Mel, 정규화 통계, 저장
│   ├─ train.py         DDPM(U-Net) 학습
│   └─ evaluate.py      이상 점수, 임계값, EER·AUROC
├─ outputs/             실행 결과 (Git 제외)
├─ requirements.txt
└─ README.md
```

## 설치 (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

## 실행
venv 환경에서 실행

```powershell
python src/preprocess.py
python src/train.py
python src/evaluate.py
```

