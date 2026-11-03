# ===============================
# preprocess.py
# 음성 파일을 U-Net에 넣을 수 있는 log-Mel로 바꿔서 저장한다.
#
# 만드는 파일 (outputs/preprocess 폴더)
#   mel_stats.pt         : train real로 계산한 Mel 칸별 평균·표준편차
#   train_full_mels.pt   : train real 파일마다 전체 길이의 log-Mel (학습 때 무작위로 2초씩 자른다)
#   val_mels.pt          : dev real을 고정 2초 조각으로 자른 log-Mel (검증·임계값용)
#   mel_example.png      : 확인용 Mel 그림
#
# 이 파일의 함수(load_audio, wave_to_logmel 등)는 evaluate.py에서도 가져다 쓴다.
# ===============================

import os  # 파일 경로를 다룬다.

import librosa  # 음성 파일을 읽고 Mel-spectrogram을 계산한다.
import matplotlib  # 그림을 저장한다.
matplotlib.use("Agg")  # 창을 띄우지 않고 파일로만 저장한다.
import matplotlib.pyplot as plt  # 그림을 그린다.
import numpy as np  # 숫자 배열을 계산한다.
import torch  # 결과를 PyTorch 텐서로 저장한다.

from config import *  # config.py의 모든 설정값을 가져온다.


# ===============================
# 1. 라벨(프로토콜) 읽기
# ===============================

def read_protocol(split):  # split: "train", "dev", "eval" 중 하나

    protocol_path = os.path.join(PROTOCOL_DIR, PROTOCOL_FILES[split])  # 읽을 프로토콜 파일 경로
    audio_dir = os.path.join(DATA_ROOT, f"ASVspoof2019_LA_{split}", "flac")  # 음성 파일이 있는 폴더

    items = []  # 파일 정보를 하나씩 담을 목록

    with open(protocol_path, "r", encoding="utf-8") as f:  # 프로토콜 파일을 연다.
        for line in f:  # 한 줄씩 읽는다.
            parts = line.strip().split()  # 예: "LA_0079 LA_T_1138215 - - bonafide" → 5조각
            if len(parts) != 5:  # 형식이 다른 줄(빈 줄 등)은 건너뛴다.
                continue

            speaker_id, file_id, _, attack_id, label = parts  # 세 번째 칸은 사용하지 않는다.

            items.append({  # 파일 하나의 정보를 저장한다.
                "path": os.path.join(audio_dir, file_id + ".flac"),  # 음성 파일 경로
                "speaker_id": speaker_id,  # 화자 ID
                "file_id": file_id,  # 파일 ID
                "attack_id": attack_id,  # 공격 종류 (real은 "-")
                "label": label,  # "bonafide"(real) 또는 "spoof"(fake)
            })

    return items  # 파일 정보 목록을 돌려준다.


# ===============================
# 2. 음성 → log-Mel
# ===============================

def load_audio(path):  # 음성 파일 하나를 읽는다.

    wave, _ = librosa.load(  # 파일을 읽어 숫자 배열(waveform)로 바꾼다.
        path,
        sr=SAMPLE_RATE,  # 16kHz로 맞춘다. (이미 16kHz면 그대로)
        mono=True,  # 모노로 맞춘다.
    )
    # 진폭 정규화는 하지 않는다. 합성 흔적(아티팩트)을 지울 수 있어서.

    return wave.astype(np.float32)  # 32비트 실수로 돌려준다.


def wave_to_logmel(wave):  # waveform → log-Mel (80, 프레임 수)

    mel = librosa.feature.melspectrogram(  # STFT → 파워 → Mel 필터를 한 번에 계산한다.
        y=wave,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,  # 크기의 제곱(파워)을 사용한다.
    )

    logmel = np.log(mel + LOG_EPS)  # 값의 차이가 너무 크지 않도록 log로 압축한다.

    return logmel.astype(np.float32)  # (80, 프레임 수)


def make_silence_mask(wave, num_frames):  # 무음 프레임은 0, 소리 있는 프레임은 1

    rms = librosa.feature.rms(  # 프레임마다 소리 크기(RMS 에너지)를 계산한다.
        y=wave,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,  # Mel과 같은 간격으로 계산해야 프레임이 맞는다.
    )[0]

    rms_db = librosa.amplitude_to_db(rms, ref=np.max)  # 가장 큰 프레임을 0dB로 놓고 dB로 바꾼다.
    silence_mask = (rms_db > SILENCE_DB_THRESHOLD).astype(np.float32)  # 기준보다 크면 1(소리 있음)

    return silence_mask[:num_frames]  # Mel 프레임 수에 맞춰 자른다.


def make_frame_mask(wave, num_frames):  # 프레임마다 학습·점수에 쓸지(1) 뺄지(0)

    mask = np.ones(num_frames, dtype=np.float32)  # 처음에는 모든 프레임을 사용한다.

    if MASK_SILENCE:  # 무음 mask를 켠 경우에만
        mask = mask * make_silence_mask(wave, num_frames)  # 무음 프레임을 0으로 바꾼다.

    return mask  # (프레임 수,)


def normalize_logmel(logmel, stats):  # train 통계로 정규화한다. (평균 0, 표준편차 1 근처)

    mean = stats["mean"][:, None]  # (80, 1): Mel 칸마다 평균
    std = stats["std"][:, None]  # (80, 1): Mel 칸마다 표준편차

    return (logmel - mean) / std  # 모든 파일에 같은 통계를 쓴다. (파일별 정규화는 하지 않는다)


def file_to_logmel(path, stats=None):  # 파일 하나 → (정규화된) log-Mel과 mask

    wave = load_audio(path)  # 음성을 읽는다.
    logmel = wave_to_logmel(wave)  # log-Mel로 바꾼다.
    mask = make_frame_mask(wave, logmel.shape[1])  # 프레임별 mask를 만든다.

    if stats is not None:  # 통계가 주어지면
        logmel = normalize_logmel(logmel, stats)  # 정규화한다.

    return logmel, mask  # (80, 프레임 수), (프레임 수,)


# ===============================
# 3. 2초 조각으로 자르기
# ===============================

def pad_to_segment(logmel_piece, mask_piece):  # SEGMENT_FRAMES보다 짧은 조각의 오른쪽을 채운다.

    missing = SEGMENT_FRAMES - logmel_piece.shape[1]  # 부족한 프레임 수

    if missing > 0:  # 부족하면
        logmel_piece = np.pad(logmel_piece, ((0, 0), (0, missing)), constant_values=0.0)  # 정규화된 값 0(평균)으로 채운다.
        mask_piece = np.pad(mask_piece, (0, missing), constant_values=0.0)  # 채운 부분은 mask=0

    return logmel_piece, mask_piece  # (80, 126), (126,)


def cut_segments(logmel, mask, hop_frames):  # 정규화된 log-Mel을 2초 조각들로 자른다.

    segments = []  # (조각 Mel, 조각 mask, 시작 프레임) 목록
    total_frames = logmel.shape[1]  # 전체 프레임 수

    for start in range(0, total_frames, hop_frames):  # hop_frames 간격으로 시작 위치를 옮긴다.
        end = start + SEGMENT_FRAMES  # 조각의 끝 위치
        piece_len = min(end, total_frames) - start  # 실제로 남은 길이

        if piece_len < MIN_SEGMENT_FRAMES:  # 1초보다 짧게 남았으면
            break  # 버리고 끝낸다.

        piece_mel, piece_mask = pad_to_segment(logmel[:, start:end], mask[start:end])  # 부족하면 채운다.
        segments.append((piece_mel, piece_mask, start))  # 목록에 추가한다.

        if end >= total_frames:  # 파일 끝까지 다 잘랐으면
            break  # 끝낸다.

    return segments


# ===============================
# 4. 정규화 통계 계산 (train real만 사용)
# ===============================

def compute_stats(raw_logmels, masks):  # Mel 칸마다 평균과 표준편차를 구한다.

    total_sum = np.zeros(N_MELS, dtype=np.float64)  # 칸별 합
    total_sq_sum = np.zeros(N_MELS, dtype=np.float64)  # 칸별 제곱합
    total_count = 0  # 사용한 프레임 수

    for logmel, mask in zip(raw_logmels, masks):  # 파일마다
        used = logmel[:, mask > 0]  # mask=1인 프레임만 사용한다.
        total_sum += used.sum(axis=1)
        total_sq_sum += (used.astype(np.float64) ** 2).sum(axis=1)
        total_count += used.shape[1]

    mean = total_sum / total_count  # 평균
    std = np.sqrt(total_sq_sum / total_count - mean ** 2)  # 표준편차 = √(제곱의 평균 − 평균의 제곱)
    std = np.maximum(std, 1e-5)  # 0으로 나누지 않도록 아주 작은 값은 막는다.

    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


# ===============================
# 5. 분할별 처리
# ===============================

def select_bonafide(split):  # 해당 분할에서 real(bonafide) 파일만 고른다.

    items = [item for item in read_protocol(split) if item["label"] == "bonafide"]  # real만

    if FILE_LIMIT is not None:  # 개발용으로 개수를 줄인 경우
        items = items[:FILE_LIMIT]  # 앞에서부터 FILE_LIMIT개만 사용한다.

    return items


def build_train(stats_out_path, train_out_path):  # train real → 통계 + 파일별 전체 log-Mel

    items = select_bonafide("train")  # train real 파일 목록
    print(f"[train] real 파일 {len(items)}개 처리 시작")

    raw_logmels, masks, file_ids = [], [], []  # 정규화 전 Mel, mask, 파일 ID

    for i, item in enumerate(items):  # 파일마다
        logmel, mask = file_to_logmel(item["path"])  # 정규화 전 log-Mel과 mask
        raw_logmels.append(logmel)
        masks.append(mask)
        file_ids.append(item["file_id"])

        if (i + 1) % 500 == 0:  # 500개마다 진행 상황을 출력한다.
            print(f"  {i + 1}/{len(items)}")

    stats = compute_stats(raw_logmels, masks)  # train real로만 통계를 계산한다.
    torch.save({"mean": torch.from_numpy(stats["mean"]), "std": torch.from_numpy(stats["std"])}, stats_out_path)

    mels = [torch.from_numpy(normalize_logmel(m, stats)) for m in raw_logmels]  # 정규화한 뒤 텐서로 바꾼다.
    torch.save({  # 파일마다 길이가 달라서 목록(list)으로 저장한다.
        "mels": mels,  # 각각 (80, 프레임 수)
        "masks": [torch.from_numpy(m) for m in masks],  # 각각 (프레임 수,)
        "file_ids": file_ids,
    }, train_out_path)

    total_frames = sum(m.shape[1] for m in mels)  # 전체 프레임 수
    print(f"[train] 저장 완료: 파일 {len(mels)}개, 전체 {total_frames}프레임 (약 {total_frames * HOP_LENGTH / SAMPLE_RATE / 60:.1f}분)")

    return stats, mels[0], torch.from_numpy(masks[0])  # 확인용 그림에 쓸 첫 파일도 돌려준다.


def build_val(stats, val_out_path):  # dev real → 고정 2초 조각

    items = select_bonafide("dev")  # dev real 파일 목록
    print(f"[dev] real 파일 {len(items)}개 처리 시작")

    seg_mels, seg_masks, seg_file_ids = [], [], []  # 조각 Mel, 조각 mask, 조각의 원본 파일 ID
    skipped = 0  # 1초보다 짧아서 조각이 하나도 안 나온 파일 수

    for i, item in enumerate(items):  # 파일마다
        logmel, mask = file_to_logmel(item["path"], stats)  # 정규화된 log-Mel과 mask
        segments = cut_segments(logmel, mask, hop_frames=SEGMENT_FRAMES)  # 겹치지 않게 2초씩 자른다.

        if len(segments) == 0:  # 조각이 하나도 없으면
            skipped += 1
            continue

        for piece_mel, piece_mask, _ in segments:  # 조각마다
            seg_mels.append(piece_mel)
            seg_masks.append(piece_mask)
            seg_file_ids.append(item["file_id"])

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(items)}")

    mels = torch.from_numpy(np.stack(seg_mels))[:, None, :, :]  # (조각 수, 1, 80, 126)
    masks = torch.from_numpy(np.stack(seg_masks))[:, None, None, :]  # (조각 수, 1, 1, 126)

    torch.save({"mels": mels, "masks": masks, "file_ids": seg_file_ids}, val_out_path)
    print(f"[dev] 저장 완료: 조각 {mels.shape[0]}개, 건너뛴 파일 {skipped}개")
    print(f"  Mel 크기 {tuple(mels.shape)}, mask 크기 {tuple(masks.shape)}")
    print(f"  값 범위 {mels.min().item():.2f} ~ {mels.max().item():.2f}, 평균 {mels.mean().item():.3f}, 표준편차 {mels.std().item():.3f}")


def save_example(logmel, mask, out_path):  # 확인용 Mel 그림을 저장한다.

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
    axes[0].imshow(logmel.numpy(), origin="lower", aspect="auto", cmap="magma")  # 아래쪽이 낮은 주파수
    axes[0].set_ylabel("Mel bin")
    axes[0].set_title("normalized log-Mel (train example)")
    axes[1].plot(mask.numpy())  # mask(1=사용, 0=제외)
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].set_ylabel("mask")
    axes[1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ===============================
# 6. 실행
# ===============================

if __name__ == "__main__":  # python preprocess.py로 실행했을 때만 아래가 동작한다.

    os.makedirs(PREPROCESS_DIR, exist_ok=True)  # 결과 폴더를 만든다.

    print("설정 확인")
    print(f"  데이터 폴더: {DATA_ROOT}")
    print(f"  2초 조각 = {SEGMENT_FRAMES}프레임, 1초 = {MIN_SEGMENT_FRAMES}프레임")
    print(f"  무음 mask: {MASK_SILENCE}, 파일 개수 제한: {FILE_LIMIT}")

    stats, example_mel, example_mask = build_train(
        stats_out_path=os.path.join(PREPROCESS_DIR, "mel_stats.pt"),
        train_out_path=os.path.join(PREPROCESS_DIR, "train_full_mels.pt"),
    )

    build_val(stats, val_out_path=os.path.join(PREPROCESS_DIR, "val_mels.pt"))

    save_example(example_mel, example_mask, os.path.join(PREPROCESS_DIR, "mel_example.png"))
    print(f"확인용 그림 저장: {os.path.join(PREPROCESS_DIR, 'mel_example.png')}")
    print("전처리 끝")
