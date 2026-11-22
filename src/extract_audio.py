"""
Korak 2: izvlacenje i pretprocesiranje audio signala iz MELD videa.

Za svaki video iz manifest.csv:
  1) ffmpeg izvuce audio kao mono, 16kHz wav (16kHz ocekuje wav2vec2)
  2) amplituda se normalizuje u opseg [-1, 1]
  3) signal se obrezuje ili dopunjava nulama na tacno 6 sekundi

Potreban je ffmpeg u PATH-u.

Rezultat: data/audio/{split}/dia{D}_utt{U}.wav
Skripta moze da se ponovo pokrene - vec napravljeni fajlovi se preskacu.
"""

import subprocess

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from config import MANIFEST, AUDIO_DIR

SAMPLE_RATE = 16000                          # Hz
CLIP_SECONDS = 6.0
TARGET_LEN = int(SAMPLE_RATE * CLIP_SECONDS)


def extract_wav_ffmpeg(video_path, tmp_wav_path):
    """Iz video fajla izvlaci audio (mono, 16kHz) u privremeni .wav."""
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),   # -y: prepisi izlaz ako postoji
        "-vn",                                   # bez video toka
        "-ac", "1",                              # mono
        "-ar", str(SAMPLE_RATE),                 # sample rate
        "-loglevel", "error",
        str(tmp_wav_path),
    ]
    subprocess.run(cmd, check=True)


def normalize_and_fix_length(wav):
    """Normalizacija u [-1, 1], pa obrezivanje/dopunjavanje na TARGET_LEN uzoraka."""
    max_val = np.max(np.abs(wav)) if len(wav) > 0 else 0.0
    if max_val > 1e-8:
        wav = wav / max_val

    if len(wav) > TARGET_LEN:
        wav = wav[:TARGET_LEN]
    elif len(wav) < TARGET_LEN:
        wav = np.pad(wav, (0, TARGET_LEN - len(wav)))   # dopuna nulama
    return wav.astype(np.float32)


def main():
    manifest = pd.read_csv(MANIFEST)
    for split in manifest["split"].unique():
        (AUDIO_DIR / split).mkdir(parents=True, exist_ok=True)

    failed = []
    tmp_path = AUDIO_DIR / "_tmp_audio.wav"

    for _, row in tqdm(manifest.iterrows(), total=len(manifest)):
        split = row["split"]
        dia, utt = row["dialogue_id"], row["utterance_id"]
        out_path = AUDIO_DIR / split / f"dia{dia}_utt{utt}.wav"
        if out_path.exists():
            continue
        try:
            extract_wav_ffmpeg(row["video_path"], tmp_path)
            wav, sr = sf.read(tmp_path, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)           # za svaki slucaj: stereo -> mono
            wav = normalize_and_fix_length(wav)
            sf.write(out_path, wav, SAMPLE_RATE, subtype="FLOAT")
        except Exception as e:
            failed.append((str(row["video_path"]), str(e)))

    if tmp_path.exists():
        tmp_path.unlink()

    print(f"\nGotovo. Neuspesnih: {len(failed)}")
    if failed:
        fail_path = AUDIO_DIR / "failed.csv"
        pd.DataFrame(failed, columns=["video_path", "error"]).to_csv(fail_path, index=False)
        print(f"Detalji o greskama: {fail_path}")


if __name__ == "__main__":
    main()
