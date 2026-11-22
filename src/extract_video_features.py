"""
Korak 5: ekstrakcija video obelezja pomocu InceptionResnetV1
(pretreniran na VGGFace2, iz facenet-pytorch), bez fine-tuning-a.

Za svaki klip postoji N_FRAMES=8 isecenih lica velicine 160x160, sto je
ulazna velicina ovog modela. Svako lice daje 512-dimenzionalni embedding,
pa se po klipu cuva sekvenca (8, 512).

Napomena: VGGFace2 model je treniran za prepoznavanje IDENTITETA lica, a ne
izraza lica, pa njegovi embeddinzi ne moraju najbolje da hvataju emocije.

Rezultat: data/video_features/{split}/dia{D}_utt{U}.npy  oblika (8, 512)
"""

import numpy as np
import pandas as pd
import torch
from facenet_pytorch import InceptionResnetV1
from tqdm import tqdm

from config import MANIFEST, FACES_DIR, VIDEO_FEAT_DIR

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Koristim device: {device}")
print("Ucitavam InceptionResnetV1 (vggface2) - prvi put skida tezine, ~107MB...")

model = InceptionResnetV1(pretrained="vggface2", classify=False).eval().to(device)
for p in model.parameters():
    p.requires_grad_(False)


def fixed_image_standardization(x):
    # standardna facenet normalizacija: piksele [0, 255] -> priblizno [-1, 1]
    return (x - 127.5) / 128.0


@torch.no_grad()
def extract_clip_embedding(face_path):
    arr = np.load(face_path)  # (N_FRAMES, 3, H, W) uint8
    faces = torch.from_numpy(arr).float().to(device)
    faces = fixed_image_standardization(faces)

    emb = model(faces)  # (N_FRAMES, 512)
    return emb.cpu().numpy()


def main():
    manifest = pd.read_csv(MANIFEST)
    for split in manifest["split"].unique():
        (VIDEO_FEAT_DIR / split).mkdir(parents=True, exist_ok=True)

    skipped_missing = 0
    failed = []

    for _, row in tqdm(manifest.iterrows(), total=len(manifest)):
        split = row["split"]
        dia, utt = row["dialogue_id"], row["utterance_id"]
        face_path = FACES_DIR / split / f"dia{dia}_utt{utt}.npy"
        out_path = VIDEO_FEAT_DIR / split / f"dia{dia}_utt{utt}.npy"

        if not face_path.exists():
            skipped_missing += 1
            continue
        if out_path.exists():
            continue

        try:
            emb = extract_clip_embedding(face_path)
            np.save(out_path, emb.astype(np.float32))
        except Exception as e:
            failed.append(f"{face_path} | {e}")

    print(f"\nPreskoceno (nema fajl lica): {skipped_missing}")
    print(f"Neuspesnih: {len(failed)}")
    if failed:
        fail_path = VIDEO_FEAT_DIR / "failed.csv"
        pd.Series(failed).to_csv(fail_path, index=False)
        print(f"Detalji o greskama: {fail_path}")


if __name__ == "__main__":
    main()
