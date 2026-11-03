# ===============================
# config.py
# 모든 경로와 하이퍼파라미터를 한곳에 모아 둔 파일
# preprocess.py, train.py, evaluate.py가 이 파일의 값을 가져다 쓴다.
# 실험할 때는 이 파일의 값만 바꾼다.
# ===============================

import os  # 폴더 경로를 합치고 폴더를 만들 때 사용한다.

import torch  # GPU 사용 가능 여부를 확인하고 난수 시작값을 고정할 때 사용한다.


# ===============================
# 1. 경로
# ===============================

DATA_ROOT = r"H:\Workspace\Dataset\LA"  # ASVspoof 2019 LA 데이터셋 폴더
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src 폴더의 한 칸 위 = 프로젝트 폴더
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")  # 어디서 실행하든 항상 프로젝트 폴더의 outputs에 저장한다.

PROTOCOL_DIR = os.path.join(DATA_ROOT, "ASVspoof2019_LA_cm_protocols")  # 라벨(프로토콜) txt 파일이 있는 폴더

PROTOCOL_FILES = {  # 분할 이름 → 프로토콜 파일 이름
    "train": "ASVspoof2019.LA.cm.train.trn.txt",  # 학습용 라벨
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",  # 검증용 라벨
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",  # 평가용 라벨
}

PREPROCESS_DIR = os.path.join(OUTPUT_DIR, "preprocess")  # 전처리 결과를 저장할 폴더
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")  # 학습한 모델을 저장할 폴더
EVAL_DIR = os.path.join(OUTPUT_DIR, "evaluation")  # 평가 결과를 저장할 폴더


# ===============================
# 2. 음성 설정
# ===============================

SAMPLE_RATE = 16000  # 모든 음성을 16kHz로 사용한다. (ASVspoof는 원래 16kHz)
SEGMENT_SEC = 2.0  # 모델에 넣는 조각 하나의 길이(초)
MIN_SEGMENT_SEC = 1.0  # 이보다 짧게 남은 마지막 조각은 버린다. (검증·평가용 조각)


# ===============================
# 3. log-Mel 설정
# ===============================

N_FFT = 512  # 한 번에 FFT를 계산할 샘플 수
WIN_LENGTH = 400  # 실제로 보는 창의 길이 (400샘플 = 25ms)
HOP_LENGTH = 256  # 창을 옮기는 간격 (256샘플 = 16ms)
N_MELS = 80  # Mel 주파수 칸 수 (그림의 세로 칸 수)
LOG_EPS = 1e-6  # log(0)을 막기 위해 더하는 아주 작은 값

SEGMENT_FRAMES = 1 + int(SEGMENT_SEC * SAMPLE_RATE) // HOP_LENGTH  # 2초 조각의 프레임 수 = 126
MIN_SEGMENT_FRAMES = 1 + int(MIN_SEGMENT_SEC * SAMPLE_RATE) // HOP_LENGTH  # 1초 조각의 프레임 수 = 63


# ===============================
# 4. 무음 mask 설정 (기능만 만들어 두고 기본값은 끈다)
# ===============================

MASK_SILENCE = False  # True면 무음 프레임도 mask=0으로 만들어 loss와 점수에서 뺀다.
SILENCE_DB_THRESHOLD = -40.0  # 파일에서 가장 큰 프레임보다 이만큼(dB) 작으면 무음으로 본다.


# ===============================
# 5. Diffusion 설정
# ===============================

TIMESTEPS = 1000  # 노이즈를 섞는 전체 단계 수
BETA_START = 0.0001  # 첫 단계에서 추가할 노이즈의 크기
BETA_END = 0.02  # 마지막 단계에서 추가할 노이즈의 크기 (선형 스케줄)


# ===============================
# 6. 모델 설정
# ===============================

BASE_CHANNELS = 32  # U-Net 첫 단계의 채널 수 (32 → 64 → 128)
TIME_EMBEDDING_DIM = 128  # timestep 숫자 하나를 128개의 특징값으로 바꾼다.
GROUP_NORM_GROUPS = 8  # GroupNorm에서 채널을 몇 묶음으로 나눠 정규화할지 (32, 64, 128 모두 나눠떨어진다)
UNET_FRAMES = 128  # U-Net 안에서 126프레임을 128로 채운다. (2번 반으로 줄여도 128 → 64 → 32로 딱 나눠진다)


# ===============================
# 7. 학습 설정
# ===============================

BATCH_SIZE = 32  # 한 번에 학습할 조각 수 (RTX 3070 Ti 8GB 기준으로 시작)
LEARNING_RATE = 0.0001  # 가중치를 한 번에 수정하는 정도
MAX_EPOCHS = 100  # 최대 학습 횟수
PATIENCE = 10  # 이 횟수만큼 val loss가 좋아지지 않으면 학습을 멈춘다. (early stopping)
GRAD_CLIP = 1.0  # gradient가 갑자기 커지는 것을 막는 상한
SAVE_EVERY = 10  # 이 epoch마다 체크포인트를 따로 저장한다. (나중에 분석용)
NUM_WORKERS = 0  # 데이터를 불러오는 별도 프로세스 수 (Windows에서는 0이 안전하다)

SEED = 42  # 실험을 반복해도 같은 결과가 나오도록 난수 시작값을 고정한다.
VAL_SEED = 1234  # 검증용 t와 노이즈를 뽑는 전용 난수 시작값
VAL_T_BIN_SIZE = 200  # val loss를 t 구간(0–200, 200–400, …, 800–1000)별로 나눠 기록할 때 구간 폭
LOSS_EPS = 1e-8  # masked MSE에서 mask=1 칸이 0개일 때 0으로 나누지 않도록 막는 값


# ===============================
# 8. 점수 계산과 판정 설정
# ===============================

SCORE_TIMESTEPS = [100, 300, 500]  # 이상 점수에 사용할 t
SCORE_NOISE_SEEDS = [1729, 2718]  # t마다 사용할 고정 노이즈의 시작값
INFER_HOP_SEC = 1.0  # 평가할 때 2초 창을 1초씩 밀면서 자른다.
INFER_HOP_FRAMES = int(INFER_HOP_SEC * SAMPLE_RATE) // HOP_LENGTH  # 1초 = 62프레임

THRESHOLD_PERCENTILE = 95  # dev bona fide 점수의 95백분위를 임계값으로 쓴다.


# ===============================
# 9. 개발용 설정 (작게 돌려볼 때만 사용)
# ===============================

FILE_LIMIT = None  # 숫자를 넣으면 분할마다 앞에서부터 그 개수의 파일만 사용한다. (예: 100)
EVAL_LIMIT = None  # 숫자를 넣으면 eval에서 무작위로 그 개수만 평가한다. (예: 2000)


# ===============================
# 10. 실행 장치와 난수
# ===============================

if torch.cuda.is_available():  # NVIDIA GPU를 사용할 수 있는지 확인
    DEVICE = torch.device("cuda")  # GPU를 사용할 수 있으면 GPU로 계산한다.
else:  # 사용할 수 있는 GPU가 없다면
    DEVICE = torch.device("cpu")  # CPU로 계산한다.

torch.manual_seed(SEED)  # PyTorch 난수 시작값을 고정한다.
