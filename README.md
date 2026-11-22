# Multimodalno prepoznavanje emocija na MELD skupu (govor + video)

Prepoznavanje emocije u kratkom isečku dijaloga iz serije *Friends* na osnovu **zvuka** (wav2vec 2.0) i **lica govornika** (MTCNN + InceptionResnetV1), uz poređenje unimodalnih modela sa **ranom** i **kasnom fuzijom**.

Skup: [MELD](https://affective-meld.github.io/) (Multimodal EmotionLines Dataset), 7 emocija: neutral, joy, surprise, anger, sadness, fear, disgust.

## Rezultati

Svi rezultati su na **test skupu** (2.444 klipa sa oba modaliteta). Model je biran na dev skupu (rano zaustavljanje po weighted F1), a test skup je korišćen samo jednom, na kraju.

| Model | Varijanta | Accuracy | Weighted F1 |
|---|---|:-:|:-:|
| *Referentni: uvek „neutral"* | – | *0.4742* | *0.3051* |
| Audio | base | 0.4664 | 0.3571 |
| Video | base | 0.4570 | 0.3252 |
| Rana fuzija | base | 0.4497 | **0.3691** |
| Kasna fuzija | base | 0.4669 | 0.3205 |
| Audio | tp_cw | 0.2537 | 0.2616 |
| Video | tp_cw | 0.2361 | 0.2615 |
| Rana fuzija | tp_cw | 0.2480 | 0.2598 |
| Kasna fuzija | tp_cw | 0.2864 | 0.2970 |

**base** = bez temporal pooling-a, bez pondera klasa. **tp_cw** = temporal pooling (mean + std) *i* ponderisanje klasa. Sirovi brojevi su u [`results/results.csv`](results/results.csv), a matrice konfuzije u [`results/confusion_matrices/`](results/confusion_matrices/).

### Kako čitati ove brojeve

Rezultati su skromni i treba ih posmatrati uz referentni model. MELD je jako neuravnotežen: klasa *neutral* čini skoro polovinu test skupa (1.159 od 2.444), pa model koji uvek kaže „neutral" ima tačnost 0.474. Zato:

- **Tačnost nijednog modela ne prelazi referentnu.** Sama tačnost ovde nije dobra mera.
- **Weighted F1 najboljih modela je iznad referentnog** (rana fuzija 0.369, audio 0.357, naspram 0.305), ali razlika je mala. Modeli nauče nešto iz signala, ali ne mnogo.
- **Varijanta tp_cw je lošija po obe metrike.** Ponderisanje klasa namerno smanjuje tačnost (model češće pogađa retke klase), ali ni weighted F1 ne raste. Pošto varijanta menja dve stvari odjednom (pooling i pondere), ne može se reći šta je od to dvoje uzrok.
- **Fuzija ne pomaže jednoznačno.** Rana fuzija je najbolja po F1, a kasna (α = 0.5) je lošija od samog audio modela. Rezultati su dobijeni jednim treniranjem po konfiguraciji, bez ponavljanja sa više seed-ova, pa se razlike od nekoliko poena ne smeju prevrednovati.

Moguća objašnjenja (nisu proverena eksperimentima): oba ekstraktora su zamrznuta i nisu prilagođena emocijama; VGGFace2 embeddinzi su učeni za prepoznavanje *identiteta* lica, a ne izraza; kratke replike iz serije često nemaju jasan emotivni signal; a klase su vrlo neuravnotežene.

## Pristup

1. **Priprema** – iz svakog videa se izdvaja audio (mono, 16 kHz, tačno 6 s) i 8 ravnomerno raspoređenih frejmova. Na frejmovima MTCNN seče lice (160×160); ako lice nije nađeno, koristi se prvo uspešno nađeno lice iz klipa, a u krajnjem slučaju centralni kadar.
2. **Obeležja (zamrznuti modeli, bez fine-tuning-a)** – `facebook/wav2vec2-base` daje sekvencu (T, 768) po klipu; InceptionResnetV1 (VGGFace2) daje (8, 512).
3. **Klasifikatori** – mali MLP-ovi nad obeležjima: audio i video grana (2 sloja), rana fuzija (konkatenacija obeležja → 3 sloja), kasna fuzija (ponderisani prosek softmax izlaza audio i video modela, α = 0.5).
4. **Trening** – Adam (lr 1e-4), batch 32, do 30 epoha, ReduceLROnPlateau, rano zaustavljanje po weighted F1 na dev skupu.

## Struktura projekta

```
├── src/
│   ├── config.py                  # sve putanje (MELD_ROOT preko promenljive okruženja)
│   ├── build_manifest.py          # 1. manifest.csv iz MELD CSV-ova i video fajlova
│   ├── extract_audio.py           # 2. audio (.wav, 16 kHz, mono, 6 s) iz videa
│   ├── extract_faces.py           # 3. lica (MTCNN), 8 frejmova po klipu
│   ├── extract_audio_features.py  # 4. wav2vec2 obeležja (768)
│   ├── extract_video_features.py  # 5. InceptionResnetV1 obeležja (512 po frejmu)
│   ├── build_final_manifest.py    # 6. samo klipovi koji imaju oba modaliteta
│   ├── train.py                   # 7. trening jednog modela
│   ├── late_fusion_eval.py        # 8. kasna fuzija iz istreniranih modela
│   ├── majority_baseline.py       # referentni model („uvek neutral")
│   ├── run_all.py                 # svi treninzi + kasna fuzija jednom komandom
│   └── common.py                  # Dataset, arhitekture modela, pomoćne funkcije
├── results/
│   ├── results.csv
│   └── confusion_matrices/
├── requirements.txt
└── (checkpoints/)                 # generiše se, nije u Git-u
```

Sirovi MELD podaci, izvedena obeležja i checkpoint-i **nisu u repozitorijumu** (preveliki su). Generišu se lokalno koracima ispod.

## Pokretanje

Testirano sa Python 3.11. Potrebni su **ffmpeg + ffprobe** u PATH-u.

```powershell
git clone https://github.com/Bastaja47/meld-multimodal-emotion-recognition.git
cd meld-multimodal-emotion-recognition

python -m venv venv
venv\Scripts\activate
# PyTorch instaliraj prema svojoj CUDA verziji: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

Preuzmi MELD (raspakovan folder sa `train_sent_emo.csv`, `dev_sent_emo.csv`, `test_sent_emo.csv` i folderima sa videima) i postavi putanju:

```powershell
$env:MELD_ROOT = "D:\podaci\MELD.Raw"        # Windows PowerShell
```
```bash
export MELD_ROOT=/putanja/do/MELD.Raw         # Linux/Mac
```

Zatim iz korena projekta, redom:

```bash
python src/build_manifest.py
python src/extract_audio.py
python src/extract_faces.py
python src/extract_audio_features.py
python src/extract_video_features.py
python src/build_final_manifest.py

python src/run_all.py            # trenira sve modele i računa kasnu fuziju
python src/majority_baseline.py  # referentni model
```

Svaki korak se može ponovo pokrenuti: već napravljeni fajlovi i modeli sa postojećim checkpoint-om se preskaču. Izvedeni podaci (audio, lica, obeležja, manifesti) se čuvaju u `MELD_ROOT`, checkpoint-i u `checkpoints/`, a rezultati u `results/`.

## Ograničenja

- Jedno treniranje po konfiguraciji; fiksni seed (42) je dodat naknadno, pa ponovno treniranje daje blago drugačije brojeve od onih u tabeli.
- Pooling i ponderisanje klasa su ispitani zajedno (`tp_cw`), ne odvojeno.
- Test skup ima manje klipova od originalnog MELD test skupa, jer su izbačeni klipovi za koje obeležja nisu mogla da se izvuku (videti ispis `build_final_manifest.py`).
- Ekstraktori obeležja nisu dotrenirani; nije ispitan ni kontekst dijaloga (samo pojedinačne replike).

## Autor

Stojan Bastaja
