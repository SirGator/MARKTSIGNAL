# WSBG-Terminal: sinnvolle Source-Grenzen für MARKTSIGNAL

Geprüfte Referenz: [bsommerfeld/wsbg-terminal](https://github.com/bsommerfeld/wsbg-terminal), Branch master, geprüft am 2026-08-22.

WSBG ist Java-basiert und unter [AGPL-3.0](https://github.com/bsommerfeld/wsbg-terminal/blob/master/LICENSE) veröffentlicht. MARKTSIGNAL enthält deshalb keinen kopierten WSBG-Code. Die Python-Implementierung übernimmt nur saubere Verantwortungsgrenzen und ist eine eigenständige Umsetzung.

## Relevante Ideen für den Score-only-Kern

| WSBG-Baustein | Wofür die Grenze nützlich ist | MARKTSIGNAL-Entsprechung |
| --- | --- | --- |
| RedditScraper.scanSubreddit, scanSubredditHot | paginiertes Listing-Polling | RedditJsonCollector.collect() |
| RedditScraper.scanComments, fetchThreadContext | Threads und verschachtelte Kommentare separat abholen | optionaler Thread-Comment-Pfad des Collectors |
| OAuthRedditFetcher / Fallback-Fetcher | Transport von Parser und Fehlerbehandlung trennen | injizierbares HttpTransport mit expliziten HTTP-/Rate-Limit-Fehlern |
| RedditThreadMapper, CommentTreeBuilder | Quell-JSON in stabile Posts/Kommentare normalisieren | RedditJsonParser, RawSocialPost, SocialObservation |
| RedditRepository | Source-Posts vom Rest des Systems getrennt und append-only halten | InMemorySocialObservationStore |
| TickerExtractor, TickerResolver | Alias/Ticker nur als Entity-Kandidaten behandeln | KeywordEntityResolver, EntityResolver |
| StoryClusterer, NearDuplicateGuard | kopierte Quellen nicht als neue wirtschaftliche Ereignisse zählen | Social Content-Hash plus EventMemory-Deduplication |
| MarketEventArchive | Ereignisse mit Evidenz archivieren und später anreichern | EventMemory + WorldGraph + ContextRetriever |

## Bewusst nicht übernommen

YahooMarketClient, MarketSnapshot und marktbezogene GUI-/Signalpfade sind für ein Marktterminal sinnvoll, gehören aber nicht zu MARKTSIGNALs Kern. Dieses Projekt erzeugt keinen technischen Marktindikator und keine Renditeprognose.

Auch aus Reddit-Upvotes, Kommentaranzahl oder Headlines wird hier kein Crowd-State konstruiert. Ein sichtbarer, persistierter Post ist zunächst nur eine Information. Er wird als SourceDocument an die Event Extraction gegeben und erhält erst mit Firmen-/Weltkontext einen Economy Score.

## Ergebnis

~~~text
WSBG-inspirierter Source-Adapter
        ↓
SourceDocument → CanonicalEvent → Event Memory
        ↓
RAG-Kontext für einen Case
        ↓
EconomyImpactModel → Score [-1,+1]
        ↓
externes, aufrufendes System
~~~

Der Score ist damit ein sauberer, auditierbarer Zusatzdatenpunkt. Das externe System entscheidet selbst, ob und wie es ihn mit Markt-, RSI- oder Fundamentaldaten kombiniert.

