# EconomyEncoder V1 — Modell-Spezifikation, Input-Vertrag V2

Die Transformer-Architektur bleibt EconomyEncoder V1. Der hier definierte,
versionierte Serializer-/Checkpoint-Vertrag ist V2.

## Ziel

Das Modell lernt genau eine Funktion:

```text
f(Information, Case, Kontext) -> Score in [-1, +1]
```

Es lernt **keine** Kursprognose, keine Wahrscheinlichkeit, keine Empfehlung.
Der Score beschreibt die erwartete wirtschaftliche Wirkung eines Ereignisses
auf einen konkreten Case über einen festen Horizont.

## Architektur

| Eigenschaft            | V1 Wert          |
|------------------------|------------------|
| Transformer-Typ        | bidirektionaler Encoder |
| Layer                  | 8                |
| Hidden Size            | 384              |
| Attention Heads        | 6                |
| Dimension pro Head     | 64               |
| FFN                    | 1024, SwiGLU     |
| Position              | RoPE             |
| Normalisierung         | Pre-RMSNorm       |
| Vokabular              | 24.000 Tokens (BPE) |
| Kontextlaenge          | 512 Tokens       |
| Output                 | 1 Wert           |
| Output-Aktivierung     | tanh             |
| Loss (Score)           | SmoothL1Loss     |
| Parameter              | ca. 20-25 Mio.   |

## Input-Format und Summary-Vertrag

Ein Eingabebeispiel ist eine einzige, deterministisch serialisierte
Token-Sequenz. Wirtschaftlich relevante Merkmale stehen dabei explizit in
strukturierten Feldern; die freie Summary ist nur ein zusaetzlicher Kanal:

```text
[CLS]
[EVENT] type=input_price_change; summary=Oil prices rose 30%;
        subjects=commodity:oil; direction=increase; magnitude=30;
        unit=percent; confidence=0.9
[CASE] type=airline; name=[CASE_ENTITY]
[HORIZON] 30d
[CONTEXT] subject=[CASE_ENTITY]; predicate=role; value=consumer; ...
[CONTEXT] subject=[CASE_ENTITY]; predicate=input_price_change_exposure;
          value=35; unit=percent; ...
[SEP]
```

Alles geht gemeinsam durch denselben Transformer. Dadurch kann Attention
direkt lernen:

```text
"Oil prices increased"
          |
       "Fuel"
          |
"35% operating costs"
          |
       "Airline"
          |
  -> wirtschaftlich negativ
```

Es gibt keine drei getrennten Transformer.

Jeder Trainingslauf besitzt genau einen Checkpoint-Vertrag
`summary_mode in {full, none}`:

| Modus | Serialisierung | Zweck |
| --- | --- | --- |
| `full` | echte Summary; einzelne `[NO_SUMMARY]`-Samples sind als robuste Augmentation erlaubt | Sprache plus strukturierte Konzepte |
| `none` | jede aktuelle und historische Summary wird exakt zu `[NO_SUMMARY]` | Structured-only-Ablation |

`none` entfernt ausschliesslich die freie Summary. Eventtyp, Subjekte,
Richtung, Magnitude, Einheit, Case, Horizont und Kontextfakten bleiben
unveraendert. Das Quellobjekt wird dabei nicht mutiert.

Der Modus gilt ohne Ausnahme fuer den run-spezifischen Pretraining-Korpus,
Score-Training, IID-Validation, alle OOD-Sets, Smoke-Tests und Produktion.
Training und Produktion verwenden dafuer denselben `ContextSerializer`. Nur
das BPE-Vokabular wird fuer einen fairen Vergleich armsuebergreifend einmal aus
dem gemeinsamen Full-Korpus gelernt; es ist kein Modell-Trainingsbeispiel.

Der Checkpoint speichert mindestens:

```text
config.summary_mode
config.model_version
config.feature_schema_version
config.max_seq_len
config.tokenizer_hash
checkpoint_format_version=2
serializer_contract_version=context-serializer-v2
vocab + merges + model_state
```

Der Produktions-Loader behandelt den gespeicherten Modus als autoritativ und
konfiguriert den `ContextTensorEncoder` automatisch. Ein unbekannter Modus darf
nicht stillschweigend ersetzt werden. Bei alten Checkpoints ohne dieses Feld
darf nur ein nachweisliches Full-Artefakt explizit als Legacy-Full geladen
werden. Alte No-Summary-Artefakte muessen neu trainiert werden, weil ihr
escaped Marker nicht dem V2-Tokenvertrag entspricht.
Die automatisch vergebenen Modell- und Feature-Schema-Versionen tragen den
Summary-Modus im Namen, damit exportierte Scores aus `full` und `none` nicht
dieselbe sichtbare Vertragsversion vortaeuschen.

## Output

Ein einzelner Skalar ueber den `[CLS]`-Vektor:

```text
[CLS]-Vektor (384)
    |
Linear 384 -> 128
    |
   SiLU
    |
  Dropout
    |
Linear 128 -> 1
    |
   tanh
    |
 Score in [-1, +1]
```

## Training in zwei Phasen

### Phase 1 — Pretraining (Sprache + Wirtschaftssprache)

Ziel: Das Modell lernt Wirtschaftssprache und Zusammenhaenge, noch keine Scores.

Daten: Wirtschaftstexte, News, Firmenberichte, Regulierungsmitteilungen.
Methode: Masked-Language-Model (MLM).

Beispiel:

```text
"Higher energy prices increase [MASK] costs."
-> Modell soll lernen: "production" / "operating"
```

Danach kennt das Modell Begriffe wie revenue, supplier, interest rate,
production, margin, export restriction, demand.

### Phase 2 — Score-Training

Ziel: Das Modell lernt die konkrete Funktion f(Event, Case, Kontext) -> Score.

Daten:

1. Regelbasierte Szenarien mit festem Label
2. KI-generierte Sprachvarianten derselben Szenarien (Label bleibt fix)
3. Gegenbeispiele: dasselbe Event, unterschiedliche Cases -> unterschiedliche Scores
4. Handgepruefte Beispiele

Beispiel-Gruppe:

```text
Event:  Oil price +30%
Case A: Airline, Fuel = 35% of costs  -> -0.80
Case B: Oil producer, Produces crude  -> +0.75
Case C: Software company, no exposure ->  0.00
```

Die Gegenbeispiele zwingen das Modell, Case und Kontext zu benutzen.
Sonst koennte es falsch lernen: "price increases" = negativ.

Loss: `torch.nn.SmoothL1Loss()`

### Faire Summary-Ablation

Ein Vergleich `full` gegen `none` verwendet denselben Split, Seed, dieselben
Basisszenarien, dieselbe Paraphrasen-/Augmentationsanzahl, Hyperparameter und
Trainingsschritte. Erst die gemeinsame Serialisierung blendet im `none`-Lauf
die Summary aus. Andernfalls wuerde der Test gleichzeitig Sprache,
Datensatzgroesse und Optimierung veraendern.

Beide Arme verwenden ausserdem dasselbe, aus dem `full`-Korpus gelernte und per
SHA-256 identifizierte BPE-Vokabular. `[NO_SUMMARY]` ist darin immer ein
reserviertes Spezialtoken, auch wenn die Full-Augmentation zufaellig kein
neutralisiertes Sample erzeugt. Dadurch bleiben Vokabulargroesse,
Embedding-Matrix, Parameterzahl und Initialisierung vergleichbar.

`OOD_LANGUAGE` misst nur im `full`-Modus Sprachrobustheit. Im `none`-Modus ist
der Text absichtlich unsichtbar; derselbe Split misst dann den Transfer der
strukturierten Felder und wird entsprechend als Structured-OOD ausgewiesen.

## Synthetische Datengenerierung

Schritt 1: Strukturierte Szenarien mit wirtschaftlicher Regel als Label.

```python
scenario = {
    "event": "input_price_increase",
    "target_role": "consumer",
    "dependency": "high",
    "magnitude": "high",
    "expected_score": -0.75,
}
```

Schritt 2: Eine KI schreibt natuerliche Sprachvarianten. Die KI darf das
Label **nicht** aendern, nur die Formulierung variieren.

Aus "Inputpreis steigt stark, Firma haengt stark davon ab" entsteht:

```text
"Copper prices surged 35% after production disruptions in South America."
"The evaluated manufacturer sources a substantial share of its raw
materials from the copper market."
```

Label bleibt: -0.75

## RAG beim Einsatz

RAG wird **nicht** mittrainiert. Beim Einsatz liefert RAG dem Modell
besseren Kontext:

```text
Artikel/Post
     |
    RAG
     |
[EVENT] + [CASE] + [CONTEXT]
     |
EconomyEncoder V1
     |
  Score
```

## Fertiges Modell

```text
Internetinformation + RAG-Kontext + Case
    |
EconomyEncoder V1
    |
Score in [-1, +1]
```

Der Score ist ein normaler numerischer Datenpunkt fuer ein externes
Analysesystem. Was danach damit passiert, ist nicht Teil dieses Modells.
