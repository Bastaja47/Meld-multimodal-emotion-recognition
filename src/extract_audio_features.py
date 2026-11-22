"""
Korak 4: ekstrakcija audio obelezja pomocu wav2vec2.

Koristi se facebook/wav2vec2-base (pretreniran, BEZ fine-tuning-a) kao
fiksni ekstraktor obelezja: iz normalizovanog 6s signala encoder daje
sekvencu latentnih reprezentacija dimenzije 768. Cuva se sirova vremenska
sekvenca (T, 768) po klipu. Svi klipovi su iste duzine (6s @ 16kHz), pa je
i T isto za sve.

Rezultat: data/audio_features/{split}/dia{D}_utt{U}.npy  oblika (T, 768)
Klipovi bez audio fajla se preskacu; vec obradjeni se ne racunaju ponovo.
"""

import numpy as np
import pandas as pd
import torch
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from tqdm import tqdm

from config import MANIFEST, AUDIO_DIR, AUDIO_FEAT_DIR

SAMPLE_RATE = 16000
MODEL_NAME = "facebook/wav2vec2-base"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Koristim device: {device}")
print(f"Ucitavam {MODEL_NAME} (prvi put skida tezine, ~360MB)...")

# processor normalizuje ulazni signal na nultu sredinu i jedinicnu varijansu
processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = Wav2Vec2Model.from_pretrained(MODEL_NAME).eval().to(device)
for p in model.parameters():
    p.requires_grad_(False)


@torch.no_grad()
def extract_clip_embedding(wav_path):
    wav, sr = sf.read(wav_path, dtype="float32")
    assert sr == SAMPLE_RATE, f"Ocekivano {SAMPLE_RATE}Hz, dobijeno {sr}Hz"

    inputs = processor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_values = inputs.input_values.to(device)

    out = model(input_values)
    hidden = out.last_hidden_state.squeeze(0)  # (T, 768)
    return hidden.cpu().numpy()


def main():
    manifest = pd.read_csv(MANIFEST)
    for split in manifest["split"].unique():
        (AUDIO_FEAT_DIR / split).mkdir(parents=True, exist_ok=True)

    skipped_missing = 0
    failed = []

    for _, row in tqdm(manifest.iterrows(), total=len(manifest)):
        split = row["split"]
        dia, utt = row["dialogue_id"], row["utterance_id"]
        audio_path = AUDIO_DIR / split / f"dia{dia}_utt{utt}.wav"
        out_path = AUDIO_FEAT_DIR / split / f"dia{dia}_utt{utt}.npy"

        if not audio_path.exists():
            skipped_missing += 1
            continue
        if out_path.exists():
            continue

        try:
            emb = extract_clip_embedding(audio_path)
            np.save(out_path, emb.astype(np.float32))
        except Exception as e:
            failed.append(f"{audio_path} | {e}")

    print(f"\nPreskoceno (nema audio fajl): {skipped_missing}")
    print(f"Neuspesnih: {len(failed)}")
    if failed:
        fail_path = AUDIO_FEAT_DIR / "failed.csv"
        pd.Series(failed).to_csv(fail_path, index=False)
        print(f"Detalji o greskama: {fail_path}")


if __name__ == "__main__":
    main()
