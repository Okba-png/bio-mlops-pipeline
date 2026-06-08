# Protein-Sekundärstrukturvorhersage mit Transformer

Dieses Repository enthält eine vollständige MLOps-Pipeline zur Vorhersage der Sekundärstruktur von Proteinen (Q3-Klassifikation: Helix, Faltblatt, Loop) basierend auf deren Aminosäuresequenz. Das Projekt implementiert ein PyTorch-basiertes **Transformer-Encoder-Baseline-Modell** und enthält fertige Skripte für Datenvorbereitung, Training, Testung und Inferenz, sowie ein direkt nutzbares Notebook für **Google Colab**.

---

## Projektstruktur

```text
├── src/
│   ├── __init__.py
│   ├── data_processing.py  # Bereinigung, Filterung und Datensplits
│   ├── dataset.py          # PyTorch Dataset und Dataloader mit dynamischem Padding
│   ├── model.py            # Transformer-Architektur und Positionskodierung
│   ├── train.py            # Trainings- und Evaluierungsschleife
│   └── predict.py          # Inferenzskript für neue Sequenzen
├── tests/
│   ├── test_processing.py  # Unittests für Datenverarbeitung
│   └── test_model.py       # Unittests für Dataset und Modell-Forward-Pass
├── notebooks/
│   └── protein_structure_colab.ipynb  # Interaktives Google Colab Notebook
├── requirements.txt        # Projekt-Abhängigkeiten
├── Makfile                 # Automatisierte Targets (install, lint, test)
└── README.md               # Dokumentation
```

---

## Datensatz

Das Projekt verwendet das Kaggle-Dataset [Protein Secondary Structure](https://www.kaggle.com/datasets/aladdinpersson/protein-secondary-structure) von Aladdin Persson. Es liefert Primärsequenzen (Aminosäuren) sowie die entsprechenden Sekundärstrukturen (Q3 und Q8), extrahiert aus der RCSB PDB mittels DSSP.

---

## Installation & Einrichtung

### Lokales Setup
Stellen Sie sicher, dass Python 3.8+ installiert ist, und führen Sie folgenden Befehl im Hauptverzeichnis aus:

```bash
# Mit Makefile
make install

# Oder manuell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Verwendung

### 1. Daten herunterladen und vorbereiten
Laden Sie die Datei `protein_secondary_structure_data.csv` von Kaggle herunter und platzieren Sie sie im Ordner `data/`.

Bereiten Sie die Daten vor (Filtern nach Länge $\le 512$ Aminosäuren und Erstellung der Splits):
```bash
python -c "from src.data_processing import prepare_data; prepare_data('data/protein_secondary_structure_data.csv', 'data/processed', max_len=512)"
```

### 2. Modell trainieren
Starten Sie das Training des Transformers. Der Skript wählt automatisch CUDA (GPU) falls verfügbar, andernfalls wird auf der CPU trainiert.

```bash
python -m src.train \
    --train_csv data/processed/train.csv \
    --val_csv data/processed/val.csv \
    --test_csv data/processed/test.csv \
    --epochs 10 \
    --batch_size 64 \
    --save_dir checkpoints
```

*Tipp: Verwenden Sie `--subset_fraction 0.1`, wenn Sie das Training schnell mit 10 % der Daten testen möchten.*

### 3. Inferenz (Vorhersage für neue Proteine)
Nutzen Sie das trainierte Modell, um die Sekundärstruktur einer benutzerdefinierten Aminosäuresequenz vorherzusagen:

```bash
python -m src.predict \
    --model_path checkpoints/best_model.pt \
    --sequence "MKTLLIL"
```

---

## Google Colab Pipeline

Wenn Sie kein lokales GPU-Setup besitzen, können Sie das komplette Projekt direkt in Google Colab ausführen. 
Öffnen Sie dazu das Notebook:
👉 `notebooks/protein_structure_colab.ipynb`

Das Notebook klont dieses Repo, führt den Kaggle-Download durch und trainiert das Modell interaktiv auf einer kostenlosen T4 GPU.

---

## Tests & Qualitätssicherung

Das Projekt enthält automatisierte Tests zur Validierung der Modell- und Datenflüsse.

```bash
# Tests ausführen
make test

# Code-Linter ausführen
make lint
```
