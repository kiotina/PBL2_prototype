# ===============================
# train.py
# 진짜 음성(bona fide)의 log-Mel만으로 DDPM(U-Net)을 학습한다.
# U-Net은 "노이즈가 섞인 Mel"을 보고 "섞인 노이즈 ε"를 맞히도록 배운다.
#
# 읽는 파일 (outputs/preprocess 폴더, preprocess.py가 만든 것)
#   train_full_mels.pt   : train real 파일마다 전체 길이의 정규화된 log-Mel
#   val_mels.pt          : dev real 고정 2초 조각
#
# 만드는 파일 (outputs/checkpoints 폴더)
#   best.pt              : val loss가 가장 낮았던 모델
#   last.pt              : 마지막 epoch의 모델
#   epoch_XXX.pt         : SAVE_EVERY epoch마다 따로 저장한 모델 (나중 분석용)
#   history.csv          : epoch마다 train/val loss와 t 구간별 val loss
#   loss_curve.png       : loss 곡선 그림
# ===============================

import csv  # history.csv를 쓴다.
import math  # timestep embedding에서 log를 계산한다.
import os  # 파일 경로를 다룬다.
import time  # epoch마다 걸린 시간을 잰다.

import matplotlib  # 그림을 저장한다.
matplotlib.use("Agg")  # 창을 띄우지 않고 파일로만 저장한다.
import matplotlib.pyplot as plt  # loss 곡선을 그린다.
import torch  # 텐서 계산과 학습
import torch.nn as nn  # 신경망 층(Conv, GroupNorm 등)
import torch.nn.functional as F  # 패딩 같은 함수
from torch.utils.data import DataLoader, Dataset, TensorDataset  # 데이터를 배치로 묶어 준다.

from config import *  # config.py의 모든 설정값을 가져온다.


# ===============================
# 1. 노이즈 스케줄 (미리 한 번만 계산해 둔다)
# ===============================

betas = torch.linspace(BETA_START, BETA_END, TIMESTEPS, device=DEVICE)  # 단계마다 추가할 노이즈 크기 β_t (선형으로 증가)
alphas = 1.0 - betas  # α_t = 1 − β_t : 단계마다 원본이 남는 비율
alpha_bars = torch.cumprod(alphas, dim=0)  # ᾱ_t = α_1 × α_2 × … × α_t : t단계까지 원본이 남는 비율
sqrt_alpha_bars = torch.sqrt(alpha_bars)  # √ᾱ_t : x_t에서 원본 x₀에 곱하는 값
sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)  # √(1−ᾱ_t) : x_t에서 노이즈 ε에 곱하는 값


# ===============================
# 2. Forward diffusion (원본에 노이즈 섞기)
# ===============================

def select_random_timesteps(batch_size, generator=None):  # 조각마다 t를 0~T-1 중 하나씩 무작위로 뽑는다.

    t = torch.randint(0, TIMESTEPS, (batch_size,), generator=generator)  # generator를 주면 그 난수로 뽑는다. (검증용)

    return t.to(DEVICE)  # (B,) 정수


def forward_diffusion(x0, t, noise=None):  # x_t = √ᾱ_t · x₀ + √(1−ᾱ_t) · ε

    if noise is None:  # 노이즈를 밖에서 주지 않았으면 (학습할 때)
        noise = torch.randn_like(x0)  # 표준정규분포 노이즈를 새로 만든다.
    # 검증·평가 때는 고정 노이즈를 밖에서 넣어 매번 같은 문제를 풀게 한다.

    sqrt_ab = sqrt_alpha_bars[t].view(-1, 1, 1, 1)  # (B,) → (B, 1, 1, 1): 조각마다 다른 t의 값을 Mel 전체에 곱하도록
    sqrt_1m_ab = sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)  # (B, 1, 1, 1)

    x_t = sqrt_ab * x0 + sqrt_1m_ab * noise  # 노이즈가 섞인 Mel

    return x_t, noise  # U-Net 입력과 정답 노이즈


# ===============================
# 3. Timestep embedding (숫자 t → 벡터)
# ===============================

def get_timestep_embedding(t, dim):  # t를 sin·cos 값 dim개로 바꾼다. (Transformer의 위치 인코딩과 같은 방식)

    half = dim // 2  # 절반은 sin, 절반은 cos
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)  # 빠른 진동부터 느린 진동까지 half개의 주파수
    args = t.float()[:, None] * freqs[None, :]  # (B, half): t마다 주파수를 곱한다.
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, dim)

    return emb  # 가까운 t는 비슷한 벡터, 먼 t는 다른 벡터가 된다.


# ===============================
# 4. U-Net
# ===============================

class ResBlock(nn.Module):  # conv → GroupNorm → SiLU → (t 더함) → conv, 그리고 입력을 더하는 skip

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)  # 크기는 그대로, 채널만 in_ch → out_ch
        self.norm = nn.GroupNorm(GROUP_NORM_GROUPS, out_ch)  # 값의 크기를 일정하게 맞춰 학습을 안정시킨다.
        self.act = nn.SiLU()  # 활성화 함수
        self.time_proj = nn.Linear(TIME_EMBEDDING_DIM, out_ch)  # t 벡터를 이 블록의 채널 수에 맞게 바꾼다.
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)  # 한 번 더 conv

        if in_ch != out_ch:  # 채널 수가 바뀌면 입력을 그대로 더할 수 없으므로
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)  # 1×1 conv로 채널 수만 맞춘다.
        else:
            self.skip = nn.Identity()  # 채널 수가 같으면 그대로 더한다.

    def forward(self, x, t_emb):  # x: (B, in_ch, H, W), t_emb: (B, TIME_EMBEDDING_DIM)

        h = self.conv1(x)  # (B, out_ch, H, W)
        h = self.act(self.norm(h))  # 정규화 → 활성화
        h = h + self.time_proj(t_emb)[:, :, None, None]  # t 정보를 채널마다 더한다. (B, out_ch) → (B, out_ch, 1, 1)
        h = self.conv2(h)  # (B, out_ch, H, W)

        return h + self.skip(x)  # skip: 입력을 더해 정보가 잘 흐르게 한다.


class UNet(nn.Module):  # 입력 (B, 1, 80, 126) → 예측 노이즈 (B, 1, 80, 126)

    def __init__(self):
        super().__init__()
        ch1, ch2, ch3 = BASE_CHANNELS, BASE_CHANNELS * 2, BASE_CHANNELS * 4  # 32, 64, 128

        self.time_mlp = nn.Sequential(  # sin·cos 벡터를 학습 가능한 층에 한 번 통과시킨다.
            nn.Linear(TIME_EMBEDDING_DIM, TIME_EMBEDDING_DIM),
            nn.SiLU(),
            nn.Linear(TIME_EMBEDDING_DIM, TIME_EMBEDDING_DIM),
        )

        self.conv_in = nn.Conv2d(1, ch1, kernel_size=3, padding=1)  # Mel 1채널 → 32채널

        # Encoder: 크기를 반으로 줄이면서 채널을 늘린다.
        self.enc1 = ResBlock(ch1, ch1)  # 80×128, 32채널
        self.down1 = nn.Conv2d(ch1, ch1, kernel_size=3, stride=2, padding=1)  # 80×128 → 40×64
        self.enc2 = ResBlock(ch1, ch2)  # 40×64, 64채널
        self.down2 = nn.Conv2d(ch2, ch2, kernel_size=3, stride=2, padding=1)  # 40×64 → 20×32

        # Bottleneck: 가장 작은 크기에서 전체 모양을 본다.
        self.mid = ResBlock(ch2, ch3)  # 20×32, 128채널

        # Decoder: 크기를 다시 키우고, Encoder 출력을 이어 붙인다(skip connection).
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")  # 20×32 → 40×64
        self.dec2 = ResBlock(ch3 + ch2, ch2)  # (128 + enc2의 64)채널 → 64채널
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")  # 40×64 → 80×128
        self.dec1 = ResBlock(ch2 + ch1, ch1)  # (64 + enc1의 32)채널 → 32채널

        self.conv_out = nn.Conv2d(ch1, 1, kernel_size=3, padding=1)  # 32채널 → 노이즈 1채널

    def forward(self, x, t):  # x: (B, 1, 80, 126), t: (B,)

        num_frames = x.shape[-1]  # 원래 프레임 수(126)를 기억해 둔다.
        x = F.pad(x, (0, UNET_FRAMES - num_frames))  # 오른쪽만 0으로 채워 (B, 1, 80, 128)

        t_emb = self.time_mlp(get_timestep_embedding(t, TIME_EMBEDDING_DIM))  # (B, 128)

        h = self.conv_in(x)  # (B, 32, 80, 128)
        h1 = self.enc1(h, t_emb)  # (B, 32, 80, 128) ← skip으로 쓸 출력
        h2 = self.enc2(self.down1(h1), t_emb)  # (B, 64, 40, 64) ← skip으로 쓸 출력
        h = self.mid(self.down2(h2), t_emb)  # (B, 128, 20, 32)

        h = self.dec2(torch.cat([self.up2(h), h2], dim=1), t_emb)  # (B, 128+64, 40, 64) → (B, 64, 40, 64)
        h = self.dec1(torch.cat([self.up1(h), h1], dim=1), t_emb)  # (B, 64+32, 80, 128) → (B, 32, 80, 128)
        out = self.conv_out(h)  # (B, 1, 80, 128)

        return out[..., :num_frames]  # 채웠던 오른쪽을 잘라 다시 (B, 1, 80, 126)


# ===============================
# 5. 학습 데이터: random crop Dataset
# ===============================

class MelCropDataset(Dataset):  # 파일 전체 Mel에서 매번 다른 위치의 2초(126프레임)를 잘라 준다.

    def __init__(self, mels, masks):
        self.mels = mels  # 파일마다 (80, 프레임 수)인 텐서 목록
        self.masks = masks  # 파일마다 (프레임 수,)인 텐서 목록

    def __len__(self):  # 1 epoch = 파일마다 한 번씩
        return len(self.mels)

    def __getitem__(self, i):  # i번째 파일에서 2초 조각 하나를 잘라 준다.

        mel = self.mels[i]  # (80, 프레임 수)
        mask = self.masks[i]  # (프레임 수,)
        num_frames = mel.shape[1]  # 이 파일의 프레임 수

        if num_frames >= SEGMENT_FRAMES:  # 2초 이상이면
            start = torch.randint(0, num_frames - SEGMENT_FRAMES + 1, (1,)).item()  # 시작 위치를 무작위로 정한다.
            mel = mel[:, start:start + SEGMENT_FRAMES]  # (80, 126)
            mask = mask[start:start + SEGMENT_FRAMES]  # (126,)
        else:  # 2초보다 짧으면
            missing = SEGMENT_FRAMES - num_frames  # 부족한 프레임 수
            mel = F.pad(mel, (0, missing))  # 오른쪽을 0(정규화된 평균값)으로 채운다.
            mask = F.pad(mask, (0, missing))  # 채운 칸은 mask=0 → loss에서 빠진다.

        return mel[None], mask[None, None]  # (1, 80, 126), (1, 1, 126)


# ===============================
# 6. Loss: masked MSE
# ===============================

def masked_mse(pred, target, mask):  # 조각마다 mask=1인 칸에서만 (예측 − 정답)²의 평균을 낸다.

    mask = mask.expand_as(pred)  # (B, 1, 1, 126) → (B, 1, 80, 126): 같은 프레임의 80칸에 같은 mask를 쓴다.
    squared_error = (pred - target) ** 2 * mask  # mask=0인 칸(패딩, 무음)의 오차는 0으로 지운다.

    error_sum = squared_error.sum(dim=(1, 2, 3))  # (B,): 조각마다 오차의 합
    count = mask.sum(dim=(1, 2, 3))  # (B,): 조각마다 mask=1인 칸 수

    return error_sum / count.clamp(min=LOSS_EPS)  # (B,): 조각마다 평균 오차 (칸 수가 0이면 아주 작은 값으로 나눈다)


# ===============================
# 7. 학습 한 epoch
# ===============================

def train_one_epoch(model, loader, optimizer):

    model.train()  # 학습 모드
    total_loss, total_count = 0.0, 0  # epoch 평균 loss를 구하기 위한 합과 개수

    for mel, mask in loader:  # 배치마다
        mel, mask = mel.to(DEVICE), mask.to(DEVICE)  # (B, 1, 80, 126), (B, 1, 1, 126)

        t = select_random_timesteps(mel.shape[0])  # 조각마다 t를 무작위로 하나씩
        x_t, noise = forward_diffusion(mel, t)  # 노이즈를 섞는다.
        pred_noise = model(x_t, t)  # U-Net이 섞인 노이즈를 예측한다.
        loss = masked_mse(pred_noise, noise, mask).mean()  # 배치 평균 loss

        optimizer.zero_grad()  # 이전 배치의 gradient를 지운다.
        loss.backward()  # gradient 계산
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)  # gradient가 너무 커지지 않게 자른다.
        optimizer.step()  # 가중치 수정

        total_loss += loss.item() * mel.shape[0]  # 배치 크기만큼 가중해서 더한다. (마지막 배치는 작을 수 있어서)
        total_count += mel.shape[0]

    return total_loss / total_count  # epoch 평균 train loss


# ===============================
# 8. 검증 (dev real 고정 조각, 고정 t·노이즈)
# ===============================

@torch.no_grad()  # 검증에서는 gradient를 계산하지 않는다. (메모리, 속도)
def validate(model, loader):

    model.eval()  # 평가 모드
    generator = torch.Generator().manual_seed(VAL_SEED)  # 검증 전용 난수. 매번 같은 시작값 → 매 epoch 같은 t와 노이즈
    all_losses, all_t = [], []  # 조각마다 loss와 t를 모은다.

    for mel, mask in loader:  # 배치마다 (순서가 고정되어 있다)
        mel, mask = mel.to(DEVICE), mask.to(DEVICE)

        t = select_random_timesteps(mel.shape[0], generator=generator)  # 검증 전용 난수로 t를 뽑는다.
        noise = torch.randn(mel.shape, generator=generator).to(DEVICE)  # 검증 전용 난수로 노이즈를 만든다.
        x_t, _ = forward_diffusion(mel, t, noise)  # 밖에서 만든 고정 노이즈를 넣는다.
        pred_noise = model(x_t, t)

        all_losses.append(masked_mse(pred_noise, noise, mask).cpu())  # (B,)
        all_t.append(t.cpu())  # (B,)

    all_losses = torch.cat(all_losses)  # (조각 수,)
    all_t = torch.cat(all_t)  # (조각 수,)

    bin_losses = {}  # t 구간별 val loss
    for start in range(0, TIMESTEPS, VAL_T_BIN_SIZE):  # 0, 200, 400, 600, 800
        end = start + VAL_T_BIN_SIZE
        in_bin = (all_t >= start) & (all_t < end)  # 이 구간에 속한 조각
        bin_losses[f"val_t{start:04d}_{end:04d}"] = all_losses[in_bin].mean().item()  # 구간 평균 loss

    return all_losses.mean().item(), bin_losses  # 전체 val loss, 구간별 val loss


# ===============================
# 9. 저장 (체크포인트, history, 그림)
# ===============================

def save_checkpoint(path, model, epoch, val_loss):  # 모델 가중치와 그때의 기록을 저장한다.

    torch.save({
        "epoch": epoch,  # 몇 번째 epoch의 모델인지
        "val_loss": val_loss,  # 그때의 val loss
        "model": model.state_dict(),  # 모델 가중치 (평가에는 이것만 필요하다)
    }, path)


def save_loss_curve(history, out_path):  # train/val loss 곡선을 그린다.

    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, [row["train_loss"] for row in history], label="train")
    ax.plot(epochs, [row["val_loss"] for row in history], label="val (dev real)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("masked MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ===============================
# 10. 실행
# ===============================

if __name__ == "__main__":  # python train.py로 실행했을 때만 아래가 동작한다.

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)  # 결과 폴더를 만든다.
    print(f"장치: {DEVICE}")

    # ----- 데이터 불러오기 -----
    train_data = torch.load(os.path.join(PREPROCESS_DIR, "train_full_mels.pt"))  # 파일마다 전체 Mel
    val_data = torch.load(os.path.join(PREPROCESS_DIR, "val_mels.pt"))  # dev 고정 조각

    train_loader = DataLoader(
        MelCropDataset(train_data["mels"], train_data["masks"]),
        batch_size=BATCH_SIZE,
        shuffle=True,  # epoch마다 파일 순서를 섞는다.
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        TensorDataset(val_data["mels"], val_data["masks"]),
        batch_size=BATCH_SIZE,
        shuffle=False,  # 검증은 순서를 고정해야 매번 같은 t·노이즈가 같은 조각에 간다.
    )
    print(f"train 파일 {len(train_data['mels'])}개, val 조각 {val_data['mels'].shape[0]}개")

    # ----- 크기 확인 -----
    model = UNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    num_params = sum(p.numel() for p in model.parameters())  # 학습할 가중치 개수
    print(f"U-Net 가중치 수: {num_params:,}")

    mel, mask = next(iter(train_loader))  # 배치 하나를 꺼내 본다.
    print(f"train 배치: Mel {tuple(mel.shape)}, mask {tuple(mask.shape)}, 값 범위 {mel.min():.2f} ~ {mel.max():.2f}")
    with torch.no_grad():
        t = select_random_timesteps(mel.shape[0])
        out = model(mel.to(DEVICE), t)
    print(f"U-Net 출력: {tuple(out.shape)} (입력과 같아야 한다)")

    # ----- 점수용 t의 √ᾱ 확인 -----
    for t_score in SCORE_TIMESTEPS:
        print(f"t={t_score}: √ᾱ = {sqrt_alpha_bars[t_score].item():.3f} (원본이 남는 비율)")

    # ----- 학습 전 loss 확인 -----
    init_val_loss, _ = validate(model, val_loader)
    print(f"학습 전 val loss: {init_val_loss:.4f} (1.0 근처여야 한다)")

    # ----- history.csv 준비 -----
    bin_names = [f"val_t{s:04d}_{s + VAL_T_BIN_SIZE:04d}" for s in range(0, TIMESTEPS, VAL_T_BIN_SIZE)]
    columns = ["epoch", "train_loss", "val_loss"] + bin_names + ["seconds"]
    history_path = os.path.join(CHECKPOINT_DIR, "history.csv")
    with open(history_path, "w", newline="", encoding="utf-8") as f:  # 새로 시작하므로 헤더부터 쓴다.
        csv.writer(f).writerow(columns)

    # ----- 학습 -----
    history = []  # 그림을 그리기 위해 epoch마다 기록을 모은다.
    best_val_loss = float("inf")  # 지금까지 가장 낮은 val loss
    epochs_without_improve = 0  # val loss가 좋아지지 않은 epoch 수 (early stopping)

    for epoch in range(1, MAX_EPOCHS + 1):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss, bin_losses = validate(model, val_loader)
        seconds = time.time() - start_time

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **bin_losses, "seconds": seconds}
        history.append(row)
        with open(history_path, "a", newline="", encoding="utf-8") as f:  # epoch마다 바로 적는다. (중간에 멈춰도 남도록)
            csv.writer(f).writerow([row[c] for c in columns])

        save_checkpoint(os.path.join(CHECKPOINT_DIR, "last.pt"), model, epoch, val_loss)  # 매 epoch 덮어쓴다.
        if epoch % SAVE_EVERY == 0:  # SAVE_EVERY epoch마다 따로 남긴다.
            save_checkpoint(os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:03d}.pt"), model, epoch, val_loss)

        if val_loss < best_val_loss:  # val loss가 좋아졌으면
            best_val_loss = val_loss
            epochs_without_improve = 0
            save_checkpoint(os.path.join(CHECKPOINT_DIR, "best.pt"), model, epoch, val_loss)
            mark = " ← best"
        else:  # 좋아지지 않았으면
            epochs_without_improve += 1
            mark = ""

        print(f"epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} | {seconds:.0f}초{mark}")

        if epochs_without_improve >= PATIENCE:  # PATIENCE번 연속 좋아지지 않으면 멈춘다.
            print(f"{PATIENCE} epoch 동안 val loss가 좋아지지 않아 학습을 멈춘다.")
            break

    save_loss_curve(history, os.path.join(CHECKPOINT_DIR, "loss_curve.png"))
    print(f"학습 끝. best val loss {best_val_loss:.4f}, 결과 폴더: {CHECKPOINT_DIR}")
