# MARKTSIGNAL

MARKTSIGNAL übersetzt unstrukturierte Internetinformationen in einen nachvollziehbaren wirtschaftlichen Datenpunkt. Es baut **keine** eigene Aktienkurs- oder Renditeprognose.

~~~text
News / Reddit / Social / Firmenmeldung / Politik
                        ↓
             Event Extraction + Deduplication
                        ↓
    RAG-Kontext: Firma, Produkt, Abhängigkeiten, Fakten
                        ↓
               EconomyImpactModel
                        ↓
              Score [-1.0, +1.0]
                        ↓
          JSON-Export für ein externes System
~~~

Ein externes System kann den Score später neben RSI, Momentum, Volumen, Volatilität oder Fundamentaldaten verwenden. Diese Weiterverarbeitung gehört nicht zu MARKTSIGNAL.

## Score-Semantik

ImpactAssessment.score beschreibt den erwarteten **wirtschaftlichen Effekt** eines Ereignisses auf einen konkreten Case über einen festen Horizont:

| Wert | Bedeutung |
| ---: | --- |
| -1 | maximal stark negativer wirtschaftlicher Effekt |
| 0 | kein materieller Nettoeffekt |
| +1 | maximal stark positiver wirtschaftlicher Effekt |

Der Score ist keine Rendite, Kursrichtung, Wahrscheinlichkeit, Kaufempfehlung oder Verkaufsempfehlung.

Beispiel: Ein Ölpreisanstieg kann für eine Airline wegen hoher Kerosinkosten zu -0.78 führen und für einen Ölproduzenten zu +0.72. Die Firmenrolle kommt aus dem RAG-Kontext, nicht aus einem Ticker in den Modellgewichten.

## Ordnerstruktur

~~~text
src/
├── sources/             # davor: externe, unstrukturierte Quellen
│   └── social/          # Reddit-Posts/-Kommentare → SourceDocument
├── extraction/          # SourceDocument → CanonicalEvent
├── memory/              # Event Memory und konservative Deduplication
├── graph/               # World Graph mit belastbaren Firmenfakten
├── retrieval/           # point-in-time RAG-Kontext für Event + Case
├── economy/             # Modellvertrag: ContextBundle → Score
├── models/              # konkretes EconomyImpactModel und Adapter
├── output/              # auditierbarer Score-Export für externe Systeme
├── domain.py            # gemeinsame, unveränderliche Verträge
└── pipeline.py          # Extraction → Memory → Retrieval → Score
~~~

sources liegt bewusst **vor** dem Kern. output liegt bewusst **nach** dem Kern und enthält keine technische Analyse oder Marktprognose.

## Implementiert

- Event Memory mit cutoff-sicherer, konservativer Deduplication
- World Graph und Context Retrieval für Firmen-, Produkt- und Abhängigkeitswissen
- Economy-Pipeline, die bei unzureichendem Fallkontext explizit abbricht
- optionale PyTorch-Referenzimplementierung eines Economy Impact Models
- durchgaengiger `summary_mode`-Vertrag (`full` oder Structured-only `none`) fuer Training, Evaluation, Checkpoint und Produktion
- Reddit-JSON-Collector mit Entity Resolution, Thread-Kommentaren, Pagination, Rate-Limit-Fehlern und append-only Revisionen/Engagement-Snapshots
- Social-Adapter: sichtbare Posts und Kommentare werden zu gewöhnlichen SourceDocument-Werten und durchlaufen denselben Extraction-/RAG-Pfad wie News
- stabiles JSON-Exportformat economy-score-output-v1

Nicht enthalten sind absichtlich ein Prediction-Modell, Markt-/Makro-Feature-Fusion, eine Renditeprognose und ein Trading-System.

## Social als Quelle nutzen

Der Collector macht beim Import keine Netzwerkaufrufe. Ein Post wird erst nach dem tatsächlichen Store-Append verfügbar und anschließend wie eine Nachricht in die Event Extraction übergeben:

~~~python
from src.sources.social import (
    InMemorySocialObservationStore,
    KeywordEntityResolver,
    LexiconSentimentScorer,
    RedditJsonCollector,
    RedditJsonParser,
    SocialIngestionOrchestrator,
    social_observation_to_document,
)

resolver = KeywordEntityResolver({
    "company:airline-x": ("Airline X", "AIRX", "$AIRX"),
})
parser = RedditJsonParser(
    entity_resolver=resolver,
    sentiment_scorer=LexiconSentimentScorer(),
)
collector = RedditJsonCollector(
    subreddit="wallstreetbetsGER",
    parser=parser,
    user_agent="marktsignal-research/0.1 contact@example.invalid",
    include_comments=True,
)
store = InMemorySocialObservationStore()
ingestion = SocialIngestionOrchestrator(collector=collector, store=store)
result = ingestion.collect_and_persist(max_pages=1)

document = social_observation_to_document(result.batch.observations[0])
# economy_pipeline.ingest_document(document)
~~~

Der produktive Reddit-Zugriff benötigt einen angemessenen User-Agent, eine Rate-Limit-/Retry-Strategie und die Einhaltung der jeweils geltenden Plattformregeln. Ein einzelner Social-Post ist dabei Information, nicht eine eigenständige Marktprognose.

## Score an ein externes System übergeben

~~~python
from src.output import assessment_to_json

assessment = economy_pipeline.assess(
    event_id="event:oil-price-rise",
    case=airline_case,
    cutoff=decision_time,
    horizon="30d",
)

payload = assessment_to_json(assessment)
~~~

Der JSON-Export enthält Score, Case, Event, Kontext-Cutoff, Modell-/Schema-Versionen und Evidenz-IDs. Er trägt explizit "is_market_prediction": false.

## Start und Tests

Voraussetzung: Python 3.11+.

~~~bash
python3 -m src info
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
~~~

Tests fuer Modell, Adapter und Training werden ohne die optionale
PyTorch-Abhaengigkeit sauber uebersprungen. Fuer die vollstaendige Suite:

~~~bash
python3 -m pip install -e '.[ml,dev]'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
~~~

Neue V2-Checkpoints speichern Summary-Modus, Serializer-Vertrag und
Tokenizer-Fingerprint. Der Produktions-Loader
konfiguriert den Serializer daraus automatisch; ein No-Summary-Modell erhaelt
damit in Produktion weiterhin exakt `[NO_SUMMARY]`, waehrend alle
strukturierten Event- und Kontextfelder erhalten bleiben. Alte Checkpoints ohne
Modus werden standardmaessig abgelehnt. Ein nachweislich altes Full-Modell kann
explizit mit `legacy_summary_mode="full"` geladen werden. Das vorhandene alte
No-Summary-Modell verwendete dagegen `(NO_SUMMARY)` und muss fuer den exakten
V2-Vertrag neu trainiert werden. Im Legacy-Full-Pfad ist eine echte Summary
Pflicht; ein `[NO_SUMMARY]`-Event wird abgelehnt. Exportierte Modell- und
Schema-Versionen kennzeichnen diesen Legacy-Lauf sichtbar.

Fuer eine faire neue Ablation muessen beide Befehle bis auf Modus und
Ausgabedatei dieselben Argumente verwenden:

~~~bash
python3 -m training train --seed 42 \
  --output .model_checkpoints/model_with_summary_v2.pt
python3 -m training train --seed 42 --no-summary \
  --output .model_checkpoints/model_no_summary_v2.pt
~~~

Beide Laeufe verwenden dann denselben expandierten Split, denselben
Initialisierungs-/Loader-Seed und denselben BPE-Fingerprint. Nur die fuer das
Modell sichtbare Summary unterscheidet sich.

## Aktuelle Grenze

Die Verträge, Retrieval-Grenzen und der Score-Export sind implementiert. Für eine reale Nutzung fehlen noch produktive News-/Firmen-/Politik-Adapter, ein trainierter und evaluierter Economy-Checkpoint, persistente Speicheradapter sowie fachlich kuratierte World-Graph-Fakten. MARKTSIGNAL führt keine Trades aus und ist keine Anlageberatung.

Die saubere, AGPL-sichere Zuordnung der als Inspiration geprüften WSBG-Bausteine steht in [WSBG_INTEGRATION.md](WSBG_INTEGRATION.md). Die vollständigen Zeit-/Score-Verträge stehen in [ARCHITECTURE.md](ARCHITECTURE.md).
