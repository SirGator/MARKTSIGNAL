# MARKTSIGNAL – Modelltraining

Dieses Repository enthält ausschließlich das Offline-Training des
wirtschaftlichen Impact-Modells. Das Runtime-System liegt in einem separaten
Projekt und importiert keinen Trainingscode. Es gibt bewusst keinen gemeinsam
importierten Kern.

```text
Frozen Dataset / synthetische Szenarien
                  ↓
     training-lokaler Serializer + Tokenizer + Modell
                  ↓
        versionierter Checkpoint (.pt)
                  ↓
       separates MARKTSIGNAL Runtime-System
```

Die einzige technische Übergabe ist der Checkpoint. Er enthält das Modell,
Vokabular, BPE-Merges, Versions- und Eingabeverträge. Periodische Snapshots
enthalten zusätzlich den Trainingszustand für ein exaktes Resume.

## Inhalt

```text
dataset/                    # versionierte Erzeugung, Schemas und IID/OOD-Splits
training/
├── modeling/               # lokale Modell-, Tokenizer- und Serializer-Kopie
├── pipeline.py             # MLM-Pretraining, Score-Training, Resume
├── input_contract.py       # Frozen-Record-Projektion und Behavioral-Fingerprint
├── runtime_views.py        # sichere, artikelähnliche Frozen-Trainingsansichten
├── paraphrases.py          # Legacy-Paraphrasen und milde Textstörungen
├── metrics.py              # MAE, Richtungstreffer und Strong-MAE
├── bridge.py               # Scenario/Record → lokaler ContextBundle
└── cli.py                  # python -m training
tests/                      # Dataset-, Vertrags- und Trainingstests
```

Das frühere `src/` gehört zum Runtime-System und ist hier nicht Bestandteil des
Pakets. `pyproject.toml` veröffentlicht nur `training*` und `dataset*`.

## Installation

Python 3.11 oder neuer:

```bash
python3 -m pip install -e ".[ml,dev]"
```

Ohne `ml` funktionieren Dataset-, Vertrags- und CLI-Tests; echte Modellläufe
benötigen PyTorch.

## Dataset erzeugen

```bash
python3 -m dataset.generate \
  --output data/dataset_v1 \
  --dataset-version 1.0 \
  --seed 42 \
  --companies 10000 \
  --bases 20000
```

Der vollständige Record-, Mechanismus-, Split- und Versionsvertrag steht in
[`DATASET_SPEC.md`](DATASET_SPEC.md).

## Training

```bash
python3 -m training train \
  --dataset data/dataset_v1 \
  --epochs 10 \
  --batch-size 32 \
  --seed 42 \
  --output .model_checkpoints/economy_encoder_v1.pt
```

Wichtige Standardeinstellungen:

- Checkpoint nach jeder Epoche: `--checkpoint-every 1`
- drei periodische Snapshots je Phase behalten: `--keep-checkpoints 3`
- Early Stopping nach drei nicht verbesserten Score-Epochen:
  `--early-stopping-patience 3`
- Mindestverbesserung der Validation-MAE: `--early-stopping-min-delta 0.0001`
- 20 % zusätzliche, labeltreue Runtime-Ansichten: `--runtime-noise-ratio 0.2`

`--no-summary` ersetzt ausschließlich die Summary durch `[NO_SUMMARY]`; alle
strukturierten Felder, Samples, Seeds und der Tokenizer-Vertrag bleiben gleich.

## Checkpoints und Resume

Für `--output .model_checkpoints/model.pt` entstehen beispielsweise:

```text
.model_checkpoints/model.pt                         # finales bestes Modell
.model_checkpoints/model.checkpoints/best.pt        # beste Validation-MAE
.model_checkpoints/model.checkpoints/pretrain_epoch_003.pt
.model_checkpoints/model.checkpoints/score_epoch_007.pt
```

Periodische Dateien sind weiterhin vom Runtime-Loader als Inferenzmodell
lesbar. Zusätzlich speichern sie Optimizer, Scheduler, DataLoader-Generator,
Python-/Torch-/CUDA-RNG und beim Pretraining den MLM-Head.

Fortsetzen:

```bash
python3 -m training train \
  --dataset data/dataset_v1 \
  --epochs 10 \
  --batch-size 32 \
  --seed 42 \
  --resume .model_checkpoints/model.checkpoints/score_epoch_007.pt \
  --output .model_checkpoints/model.pt
```

Resume ist absichtlich strikt: Dataset-Reihenfolge, Tokenizer, Architektur,
Batchgröße, Lernrate, Epochenziel, Summary-Modus, Noise-Profil und Gerätetyp
müssen zum Snapshot passen. Ein finaler Inferenz- oder Best-Checkpoint enthält
keinen Optimizerzustand und wird deshalb für `--resume` klar abgelehnt.

## Evaluation

Validation, IID und OOD werden vollständig und gebatcht ausgewertet. Berichtet
werden:

- MAE
- Richtungstreffer mit neutralem Band ±0,05
- Strong-MAE für Ziele mit `abs(score) >= 0,5`
- Slices nach Mechanismus, Richtung, Horizont und Impact-Stärke
- bei Frozen-Daten zusätzlich Quelle und Counterfactual-Status

Kleine Slices mit weniger als 20 Beispielen werden sichtbar markiert.

## Eingabevertrag

Frozen Magnitudes und Kontextfaktoren bleiben normierte Ratios in `[0, 1]`.
Es findet keine stille Umrechnung in Prozent statt. Mehrdeutige Units werden
abgelehnt. Frozen-Checkpoints speichern die Projektionsversion, beide Skalen
und einen Behavioral-Fingerprint, der eine Probe-Suite über alle Mechanismen,
Kontextfaktoren, Units und Summary-Modes durch den echten Serializer hashst.

Der Serializer (`context-serializer-v3`) sortiert Facts und historische
Events kanonisch; die Tokenfolge hängt nicht von der Lieferreihenfolge des
Retrievals ab. Das Modell verlangt exakt ein `[CLS]` pro Sequenz — doppelte
oder fehlende `[CLS]` werden abgelehnt, nicht still korrigiert.

Artikelähnliches Rauschen wird erst nach der Split-Zuweisung und nur im
Trainingsanteil erzeugt. Validation und OOD bleiben unverändert. Fehlerhafte
Richtung, Entity oder Eventart werden nicht mit unverändertem Label trainiert.

## Trainingsmechanik

- MLM-Pretraining nutzt BERT-Strategie 80/10/10 (80 % `[MASK]`, 10 %
  Zufalls-Token, 10 % unverändert); Struktur-Tokens wie `[EVENT]`, `[CASE]`,
  `[CONTEXT]`, `[HORIZON]` werden nie maskiert. Ein Batch ohne maskierbare
  Tokens wird abgelehnt statt mit NaN-Loss weiterzulaufen.
- Tokenisierung wird pro Dataset einmal gecacht; Epochen und Validierung
  wiederverwenden die Token-IDs statt BPE wiederholt auszuführen.
- `TrainingConfig` validiert alle Felder hart (Typen, Grenzen,
  `d_model % num_heads`, FIN-Bedingungen); ungültige direkte Python-Nutzung
  scheitert sofort.
- Die Attention nutzt `scaled_dot_product_attention` (fused/flash-Kernel
  auf der GPU); Per-Batch-Input-Prüfungen sind über
  `strict_validation=False` abschaltbar.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

Ohne PyTorch werden ML-Integrationstests übersprungen. Die dependency-freien
Vertrags-, Dataset- und CLI-Tests bleiben ausführbar.

## Grenze

Rohartikel-Extraction gehört zum separaten Runtime-/Extraction-System. Dieses
Training lernt den wirtschaftlichen Score aus kanonischen Events und
abgerufenem Kontext; es ist weder Renditeprognose noch Trading-System.
