# Architektur: Information → Kontext → Economy Score

## Systemgrenze

MARKTSIGNAL beantwortet nur diese Frage:

~~~text
Was bedeutet eine bekannte Information wirtschaftlich für diesen Case?
~~~

Die Antwort ist ein reproduzierbares ImpactAssessment:

~~~text
(event_id, case_id, cutoff, horizon) → score ∈ [-1, +1]
~~~

Der Score ist kein erwarteter Aktienkurs, keine Rendite, keine Trading-Richtung und keine Handelsempfehlung. Ein nachgelagertes Markt- oder Handelssystem gehört dem jeweiligen Nutzer und ist kein Teil dieses Projekts.

## Verantwortungsgrenzen

~~~mermaid
flowchart LR
    subgraph Before[Vor dem Kern: Quellen]
        N[News]
        S[Social / Reddit]
        F[Firmenmeldungen]
        P[Politik]
    end

    N --> X[SourceDocument]
    S --> X
    F --> X
    P --> X
    X --> E[Event Extraction]
    E --> M[(Event Memory / Dedup)]
    M --> R[Context Retrieval]
    G[(World Graph)] --> R
    C[Case] --> R
    R --> B[ContextBundle]
    B --> I[EconomyImpactModel]
    I --> A[ImpactAssessment score -1 to +1]
    A --> O[Score JSON Export]
    O -. caller-owned .-> D[externes Analyse- oder Trading-System]
~~~

| Schicht | Verantwortlichkeit | Nicht ihre Aufgabe |
| --- | --- | --- |
| sources | externe Inhalte sammeln und zeitlich korrekt speichern | wirtschaftlich bewerten |
| extraction | untrusted Text zu kanonischen Event-Kandidaten normalisieren | Kontext erfinden |
| memory | Events und Provenienz dedupliziert, cutoff-sicher halten | Marktprognosen ableiten |
| graph / retrieval | belastbare, fallrelevante Fakten für RAG liefern | den Score bestimmen |
| economy / models | Event + Case + Kontext wirtschaftlich bewerten | Kursbewegungen prognostizieren |
| output | Score und Audit-Metadaten extern bereitstellen | externe Indikatoren fusionieren |

## Social Media

Social Media hat im Kern dieselbe Rolle wie eine Nachricht. Nach erfolgreicher, append-only Persistenz konvertiert social_observation_to_document() einen Post oder Kommentar zu SourceDocument. Danach ist der Pfad identisch:

~~~text
Reddit-Post → SourceDocument → Event Extraction → RAG → Economy Score
~~~

Likes, Upvotes und Kommentare bleiben zeitlich versionierte Quellmetadaten. Sie werden in V1 weder zu einer Crowd-Prognose aggregiert noch mit technischen Marktdaten fusioniert. Falls ein externes System solche Zähler benötigt, kann es sie separat abrufen; sie verändern nicht die Semantik des Economy Scores.

## Point-in-time und Provenienz

Historische und Live-Auswertung folgen derselben Regel: ein Inhalt ist nur dann verwendbar, wenn das System ihn am cutoff tatsächlich kannte.

| Zeitfeld | Bedeutung |
| --- | --- |
| occurred_at | wann ein wirtschaftliches Ereignis stattfand |
| published_at / authored_at | wann eine Quelle den Inhalt veröffentlichte |
| observed_at / known_at | wann die Quelle für das System sichtbar war |
| ingested_at | wann MARKTSIGNAL den Inhalt tatsächlich persistierte |
| cutoff | Wissensgrenze eines konkreten Assessments |

Event Memory und Context Retrieval schließen Material nach dem Cutoff aus. Social-Persistenz stellt sicher, dass ein neuer Poll niemals künstlich in die Vergangenheit datiert wird. Eine spätere Post-Revision wird append-only als neue Revision gespeichert; Leser erhalten nur die am Cutoff sichtbare Revision.

## RAG-Kontext

Retrieval liefert keine freie, unbelegte Zusammenfassung, sondern ein ContextBundle mit ContextFact-Werten, historischen Vergleichsereignissen und Evidenz-IDs. Beispiele:

- fuel_cost_share = 31 % für eine Airline
- DEPENDS_ON(company, china_chip_imports)
- SELLS_TO(company, customer_b) mit Umsatzanteil und Gültigkeitszeitraum

Das Modell bewertet daher nicht "Ölpreis steigt" isoliert, sondern dessen wirtschaftliche Bedeutung für die konkrete Firma. Fehlt ein Fallbezug oder die Provenienz, bricht die Pipeline explizit ab. Ein fehlender Kontext wird nie als neutraler Wert 0 ausgegeben.

## Economy Score

Der öffentliche Vertrag ImpactAssessment enthält mindestens:

~~~text
event_id, case_id, horizon,
score, confidence,
assessed_at, context_cutoff,
model_version, feature_schema_version, input_hash, evidence_ids
~~~

score ist der normalisierte inkrementelle wirtschaftliche Effekt relativ zu einer Welt ohne dieses Event. confidence ist optional und vom Score getrennt. Vergleiche sind nur sinnvoll, wenn Horizon, Modell- und Feature-Version zur selben Labeldefinition passen.

Eine tanh-Ausgabe begrenzt Modellwerte technisch auf [-1, +1]; sie ersetzt keine Kalibrierung. Vor produktiver Nutzung sind dokumentierte Labels, zeitgetrennte Evaluation und versionierte Modellartefakte erforderlich.

### Summary-Mode als Modellvertrag

Der EconomyEncoder kann mit zwei expliziten Eingabevertraegen trainiert werden:

- `full`: strukturierte Eventfelder plus freie Event-Summary
- `none`: dieselben strukturierten Felder, aber jede aktuelle und historische Summary wird exakt als `[NO_SUMMARY]` serialisiert

Der Modus ist kein Laufzeit-Toggle fuer Experimente mit demselben Modell. Er
ist Teil des Checkpoints und muss fuer Training, IID-Validation, OOD-Evaluation
und Produktion identisch sein. Der Produktions-Loader liest
`config.summary_mode` und baut den Serializer passend auf; der Modus fliesst
ausserdem in den reproduzierbaren Input-Hash ein. Modell- und
Feature-Schema-Versionen stammen autoritativ aus dem Checkpoint und enthalten
bei neuen Trainingslaeufen den Modus. Vokabular und BPE-Merges werden mit einem
gespeicherten SHA-256-Fingerprint gegen den Checkpoint geprueft.

Damit bleibt die Aufgabenteilung sauber: Der Event Extractor versteht Sprache
und normalisiert sie zu Konzepten. Ein Structured-only EconomyEncoder bewertet
diese Konzepte mit Case- und RAG-Kontext. Ein `full`-Modell darf die Summary als
zusaetzliches Signal verwenden, kann dadurch aber empfindlicher gegen neue
Formulierungen werden.

## Externe Schnittstelle

src.output.assessment_to_record() und assessment_to_json() exportieren den Wert in economy-score-output-v1.

~~~json
{
  "kind": "economic_impact_score",
  "is_market_prediction": false,
  "case_id": "case:airline-x",
  "event_id": "event:oil-price-rise",
  "horizon": "30d",
  "score": -0.78
}
~~~

Die vollständige Evidenz- und Versionskette bleibt im selben Record erhalten. Damit kann ein beliebiges externes System den Score übernehmen, ohne dass MARKTSIGNAL über die Verwendung entscheidet.

## Bewusst außerhalb des Projekts

- Fusion aus Economy, Social, Markt und Makro
- PredictionResult, erwartete Rendite oder Kursrichtung
- technische Indikatoren wie RSI, Momentum, Volumen und Volatilität
- Backtests auf Rendite-Labels und Trading-Ausführung
- eigene Markt-/Makro-Feeds

Diese Dinge können später von einem aufrufenden System mit dem Score kombiniert werden. Sie verändern jedoch weder das Economy-Modell noch dessen öffentlichen Vertrag.
