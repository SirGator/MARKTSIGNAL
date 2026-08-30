# Dataset-Vertrag V1

MARKTSIGNAL erzeugt Trainingslabels nicht mehr ausschließlich spontan im
Trainingsprozess. Der neue Datenpfad ist eigenständig und versioniert:

~~~text
strukturelle Firmenprofile
        ↓
abstrakter Eventmechanismus + konkretes Subject
        ↓
Event + EconomicContext + Horizon
        ↓
StateDelta
        ↓
EconomicImpact
        ↓
Verifier → feste Splits → JSONL + Manifest
~~~

`training/scenarios.py` bleibt vorerst als Legacy-Kompatibilität erhalten. Ein
Training auf dem neuen Vertrag liest dagegen ausschließlich ein bereits
eingefrorenes Dataset-Verzeichnis.

## Kanonischer Record

`dataset/schema.py` definiert unveränderliche, streng validierte Dataclasses.
Alle Eventmagnituden und strukturellen Faktoren liegen in `[0, 1]`; die sechs
State-Deltas und `economic_impact` liegen in `[-1, 1]`. `None` im Kontext
bedeutet unbekannt, `0.0` bedeutet eine bekannte Abwesenheit der Exposition.

~~~json
{
  "id": "case_000001",
  "event": {
    "mechanism": "input_cost_change",
    "subject_class": "energy",
    "subject": "oil",
    "direction": "increase",
    "magnitude": 0.3
  },
  "target": {"entity_id": "synthetic_00001"},
  "context": {
    "input_dependency": 0.42,
    "hedging": 0.15,
    "pricing_power": 0.25,
    "substitutability": 0.1,
    "inventory_buffer": 0.2
  },
  "horizon_days": 7,
  "state_delta": {
    "revenue_expectation": 0.0,
    "demand": 0.0,
    "operating_costs": 0.078,
    "margin_expectation": -0.059,
    "supply_risk": 0.008,
    "competitive_position": -0.015
  },
  "economic_impact": -0.126,
  "confidence": 0.9,
  "metadata": {
    "source": "synthetic",
    "generator_version": "generate-v1",
    "seed": 42
  }
}
~~~

Die Zahlen sind nur ein gekürztes Formatbeispiel. Der Generator berechnet und
verifiziert die gespeicherten Werte selbst.

## Mechanismen

Die Registry in `dataset/mechanisms/` enthält genau diese 15
subject-unabhängigen Übertragungsmechanismen:

1. `input_cost_change`
2. `output_price_change`
3. `demand_change`
4. `supply_change`
5. `capacity_change`
6. `supplier_disruption`
7. `customer_dependency_change`
8. `financing_cost_change`
9. `currency_change`
10. `regulatory_cost_change`
11. `tax_change`
12. `subsidy_change`
13. `competitive_pressure_change`
14. `contract_change`
15. `labor_cost_change`

Jede Definition nennt gültige Richtungen und Subject-Klassen, notwendige
Kontextfelder und erlaubte State-Delta-Ausgänge. Die Labeler dispatchen nur
über den Mechanismus. Konkrete Namen wie `oil`, `wheat` oder `cloud_compute`
kommen in keiner Scoreformel vor.

## Firmenwelt und Counterfactuals

`dataset/companies/` zieht persistente Firmenzustände aus nicht-uniformen,
teilweise korrelierten Verteilungen. Die Firma besitzt keine Industrieklasse,
die als Label-Shortcut dienen könnte. Derselbe `synthetic_NNNNN`-Zustand wird
über mehrere Events wiederverwendet.

`dataset/counterfactuals.py` variiert pro Familie exakt einen kausalen Input.
Event, Target, Horizon und alle anderen Kontextwerte bleiben gleich; danach
werden StateDelta und Impact neu berechnet. Familien werden beim Splitten nie
auseinandergerissen.

## Feste Splits

Ein Export enthält immer:

~~~text
train.jsonl
validation.jsonl
test_iid.jsonl
test_entity_ood.jsonl
test_parameter_ood.jsonl
test_combination_ood.jsonl
test_concept_ood.jsonl
companies.jsonl
manifest.json
~~~

- Entity-OOD hält vollständige Firmen-IDs aus allen anderen Splits heraus.
- Parameter-OOD reserviert extreme Parameterräume.
- Combination-OOD hält seltene strukturelle Kombinationen zusammen zurück.
- Concept-OOD reserviert `wheat`, `electricity`, `freight`, `cloud_compute`
  und `semiconductor_wafers` als konkrete, im Training unsichtbare Subjects.
- Das Manifest speichert Versionen, Seed, Counts und SHA-256 für jedes
  Artefakt. Ein bestehendes Dataset-Verzeichnis wird niemals überschrieben.

## Erzeugen und trainieren

Ein reproduzierbarer Lauf:

~~~bash
python3 -m dataset.generate \
  --output data/dataset_v1 \
  --dataset-version 1.0 \
  --seed 42 \
  --companies 10000 \
  --bases 20000
~~~

Danach liest der Trainingspfad die eingefrorenen Records:

~~~bash
python3 -m training train \
  --dataset data/dataset_v1 \
  --seed 42 \
  --output .model_checkpoints/economy_encoder_dataset_v1.pt
~~~

Der Dataset-Record wird ohne Label-Leakage in ein produktives
`ContextBundle` übersetzt und vom gleichen `ContextSerializer` wie bei der
Inference serialisiert. Dataset-, Schema-, Generatorversion und Dataset-Seed
werden im Checkpoint festgehalten.

## Geltungsgrenze

Die V1-Formeln sind transparente, deterministische Startlabels. Sie sind keine
empirisch kalibrierte ökonomische Wahrheit. Reale SEC-/Firmen-/Makrodaten
sollen als Nächstes die Parameterverteilungen kalibrieren; Teacher-Labels
sollen erst nach dem regelbasierten Verifier hinzukommen. Für synthetische
Records prüft der Verifier exakte Formelparität. Teacher-/kuratierte Records
dürfen eigenständige State-Deltas tragen, müssen aber Registry, erlaubte
State-Felder, Vorzeichen, Grenzen und die gemeinsame Impact-Aggregation
erfüllen. Jede Änderung der Formeln oder Verteilungen erfordert eine neue
Generator- und Dataset-Version.
