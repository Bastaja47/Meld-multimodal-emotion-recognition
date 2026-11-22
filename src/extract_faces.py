"""
Korak 3: izvlacenje lica iz MELD video klipova.

  1) Iz svakog klipa se uzorkuje N_FRAMES ravnomerno rasporedjenih frejmova.
  2) Na svakom frejmu MTCNN (facenet-pytorch, pretreniran detektor) nalazi i
     isece lice. Koristi se samo za lokalizaciju, ne za obelezja.
  3) Isecena lica (FACE_SIZE x FACE_SIZE) cuvaju se kao .npy niz oblika
     (N_FRAMES, 3, FACE_SIZE, FACE_SIZE), tip uint8.

Pravi embedding vektori se racunaju u sledecem koraku
(extract_video_features.py, InceptionResnetV1).

Ako lice nije nadjeno na nekom frejmu, koristi se prvo uspesno detektovano
lice iz istog klipa. Ako nije nadjeno nijedno, koristi se centralni kvadratni
isecak celog kadra.

Potrebni su ffmpeg i ffprobe u PATH-u.

Rezultat: data/faces/{split}/dia{D}_utt{U}.npy
Skripta moze da se ponovo pokrene - vec napravljeni fajlovi se preskacu.
"""

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm

from config import MANIFEST, FACES_DIR

N_FRAMES = 8
FACE_SIZE = 160

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Koristim device: {device}")

mtcnn = MTCNN(
    image_size=FACE_SIZE,
    margin=20,
    post_process=False,   # zadrzi originalne piksele (0-255); normalizacija je u narednom koraku
    select_largest=True,  # ako ima vise lica u kadru, uzmi najvece (glavni govornik)
    keep_all=False,
    device=device,
)


def sample_frame_indices(total_frames, n):
    if total_frames <= 0:
        return []
    if total_frames <= n:
        return list(range(total_frames))
    return np.linspace(0, total_frames - 1, n).astype(int).tolist()


def probe_size(video_path):
    """Sirina/visina video toka preko ffprobe (brzo, ne broji frejmove)."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        str(video_path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    w_str, h_str = out.split("x")
    return int(w_str), int(h_str)


def read_all_frames(video_path):
    """
    Dekodira ceo video preko ffmpeg-a kao sirove RGB frejmove (pipe).

    Namerno se ne koristi cv2.VideoCapture: njegova ugradjena ffmpeg biblioteka
    se u ovom projektu na ~820 MELD klipova nije mogla da dekodira, dok
    standalone ffmpeg radi za sve klipove.
    """
    try:
        width, height = probe_size(video_path)
    except subprocess.CalledProcessError:
        return []

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-fps_mode", "passthrough",   # ne menjaj/ne dupliraj frejmove zbog promenljivog fps-a
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-loglevel", "error",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    raw = proc.stdout
    frame_size = width * height * 3
    if frame_size == 0 or len(raw) < frame_size:
        return []

    n_frames = len(raw) // frame_size
    arr = np.frombuffer(raw, dtype=np.uint8, count=n_frames * frame_size)
    arr = arr.reshape(n_frames, height, width, 3)
    return [arr[i] for i in range(n_frames)]


def read_sampled_frames(video_path, n=N_FRAMES):
    all_frames = read_all_frames(video_path)
    if len(all_frames) == 0:
        return []
    indices = sample_frame_indices(len(all_frames), n)
    return [all_frames[i] for i in indices]


def center_crop_resize(frame, size=FACE_SIZE):
    """Rezervno resenje: centralni kvadratni isecak kadra, skaliran na size x size."""
    h, w, _ = frame.shape
    m = min(h, w)
    y0, x0 = (h - m) // 2, (w - m) // 2
    cropped = frame[y0:y0 + m, x0:x0 + m]
    resized = cv2.resize(cropped, (size, size))
    return torch.from_numpy(resized).permute(2, 0, 1).float()


def process_clip(video_path):
    frames = read_sampled_frames(video_path)
    if len(frames) == 0:
        return None

    pil_frames = [Image.fromarray(f) for f in frames]
    detected = mtcnn(pil_frames)  # lista: tensor (3, H, W) ili None po frejmu

    good = [d for d in detected if d is not None]
    if len(good) == 0:
        # nijedno lice nije nadjeno u celom klipu -> centralni kadar
        faces = [center_crop_resize(f) for f in frames]
    else:
        fallback = good[0]
        faces = [d if d is not None else fallback for d in detected]

    while len(faces) < N_FRAMES:      # kratki klipovi: ponovi poslednji frejm
        faces.append(faces[-1])
    faces = faces[:N_FRAMES]

    return torch.stack(faces).clamp(0, 255).byte().numpy()  # (N, 3, H, W) uint8


def main():
    manifest = pd.read_csv(MANIFEST)
    for split in manifest["split"].unique():
        (FACES_DIR / split).mkdir(parents=True, exist_ok=True)

    failed = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest)):
        split = row["split"]
        dia, utt = row["dialogue_id"], row["utterance_id"]
        out_path = FACES_DIR / split / f"dia{dia}_utt{utt}.npy"
        if out_path.exists():
            continue
        try:
            arr = process_clip(Path(row["video_path"]))
            if arr is None:
                failed.append(str(row["video_path"]))
                continue
            np.save(out_path, arr)
        except Exception as e:
            failed.append(f"{row['video_path']} | {e}")

    print(f"\nGotovo. Neuspesnih: {len(failed)}")
    if failed:
        fail_path = FACES_DIR / "failed.csv"
        pd.Series(failed).to_csv(fail_path, index=False)
        print(f"Detalji o greskama: {fail_path}")


if __name__ == "__main__":
    main()
