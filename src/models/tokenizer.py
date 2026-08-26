"""Byte-Pair-Encoding tokenizer for EconomyEncoder V1.

Learns subword merges from training text. Handles unknown words by
falling back to subword pieces instead of a single [UNK] token.

Special tokens (reserved, never merged, never split by BPE):

    [pad]              = 0
    [cls]              = 1
    [sep]              = 2
    [unk]              = 3
    [mask]             = 4
    [event]            = 5
    [case]             = 6
    [context]          = 7
    [horizon]          = 8
    [historical_event] = 9
    [case_entity]      = 10
    [no_summary]       = 11

Structure tokens like [event], [case], [context] etc. are reserved as
special tokens so BPE never splits them into subwords.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import re
from pathlib import Path
from typing import Sequence


SPECIAL_TOKENS = [
    "[pad]",
    "[cls]",
    "[sep]",
    "[unk]",
    "[mask]",
    "[event]",
    "[case]",
    "[context]",
    "[horizon]",
    "[historical_event]",
    "[case_entity]",
    "[no_summary]",
]
SPECIAL_SET = frozenset(SPECIAL_TOKENS)

_GPT_SPLIT = re.compile(r"\[[A-Z_]+\]|[-+]?\d+(?:[.,]\d+)?|[^\W_]+|[^\s\w]", re.UNICODE)


def _pretokenize(text: str) -> list[str]:
    """Split text into word-level units before BPE merges."""
    return _GPT_SPLIT.findall(text)


def _word_to_symbols(word: str) -> tuple[str, ...]:
    """Split a word into initial symbols (characters + end-of-word marker)."""
    return tuple(word) + ("</w>",)


class BPETokenizer:
    """Byte-Pair-Encoding tokenizer with learn/save/load/encode/decode."""

    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[tuple[str, str]],
    ) -> None:
        self._vocab = dict(vocab)
        self._merges = list(merges)
        self._merge_ranks: dict[tuple[str, str], int] = {
            pair: i for i, pair in enumerate(self._merges)
        }
        self._id_to_token: dict[int, str] = {v: k for k, v in self._vocab.items()}

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    @property
    def merges(self) -> list[tuple[str, str]]:
        return list(self._merges)

    @property
    def fingerprint(self) -> str:
        """SHA-256 identity of the complete token-to-id and merge contract."""

        canonical = json.dumps(
            {
                "merges": [list(pair) for pair in self._merges],
                "vocab": self._vocab,
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def validate_special_tokens(self, *, include_no_summary: bool = True) -> None:
        """Require canonical IDs for the selected serializer contract."""

        expected_tokens = (
            SPECIAL_TOKENS if include_no_summary else SPECIAL_TOKENS[:-1]
        )
        mismatches = {
            token: self._vocab.get(token)
            for expected_id, token in enumerate(expected_tokens)
            if self._vocab.get(token) != expected_id
        }
        if mismatches:
            expected = {
                token: expected_id
                for expected_id, token in enumerate(expected_tokens)
                if token in mismatches
            }
            raise ValueError(
                "tokenizer special-token contract mismatch: "
                f"expected {expected}, got {mismatches}"
            )

    @classmethod
    def train(
        cls,
        texts: Sequence[str],
        vocab_size: int = 24_000,
        verbose: bool = False,
    ) -> "BPETokenizer":
        """Train a BPE tokenizer from raw text.

        Args:
            texts: corpus to learn merges from
            vocab_size: target vocabulary size (including special tokens)
            verbose: print progress
        """
        if vocab_size < len(SPECIAL_TOKENS):
            raise ValueError(f"vocab_size must be at least {len(SPECIAL_TOKENS)}")

        vocab: dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}

        word_freqs: Counter[tuple[str, ...]] = Counter()
        for text in texts:
            for word in _pretokenize(text):
                normalized = word.casefold()
                if normalized in SPECIAL_SET:
                    continue
                word_freqs[_word_to_symbols(normalized)] += 1

        char_set: set[str] = set()
        for symbols in word_freqs:
            char_set.update(symbols)

        for char in sorted(char_set):
            if char not in vocab:
                vocab[char] = len(vocab)
                if len(vocab) >= vocab_size:
                    return cls(vocab, [])

        pair_counts: Counter[tuple[str, str]] = Counter()
        for symbols, freq in word_freqs.items():
            for i in range(len(symbols) - 1):
                pair_counts[(symbols[i], symbols[i + 1])] += freq

        merges: list[tuple[str, str]] = []
        while len(vocab) < vocab_size and pair_counts:
            best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
            if pair_counts[best_pair] < 2:
                break

            new_token = best_pair[0] + best_pair[1]
            vocab[new_token] = len(vocab)
            merges.append(best_pair)

            new_word_freqs: Counter[tuple[str, ...]] = Counter()
            for symbols, freq in word_freqs.items():
                new_symbols: list[str] = []
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                        new_symbols.append(new_token)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                new_word_freqs[tuple(new_symbols)] += freq
            word_freqs = new_word_freqs

            pair_counts = Counter()
            for symbols, freq in word_freqs.items():
                for i in range(len(symbols) - 1):
                    pair_counts[(symbols[i], symbols[i + 1])] += freq

            if verbose and len(merges) % 500 == 0:
                print(f"  merges={len(merges)} vocab={len(vocab)}")

        return cls(vocab, merges)

    def _encode_word(self, word: str) -> list[int]:
        """Encode a single word using learned merges."""
        normalized = word.casefold()
        if normalized in self._vocab:
            return [self._vocab[normalized]]

        symbols = list(_word_to_symbols(normalized))
        while len(symbols) > 1:
            best_rank = None
            best_idx = -1
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self._merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i
            if best_idx < 0:
                break
            symbols[best_idx : best_idx + 2] = [symbols[best_idx] + symbols[best_idx + 1]]

        ids: list[int] = []
        for sym in symbols:
            tid = self._vocab.get(sym, self._vocab["[unk]"])
            ids.append(tid)
        return ids

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs. Special tokens are matched exactly."""
        ids: list[int] = []
        for token in _GPT_SPLIT.findall(text):
            normalized = token.casefold()
            if normalized in SPECIAL_SET:
                # Legacy vocabularies may predate a newly reserved token.  They
                # degrade explicitly to UNK; V2 checkpoint validation prevents
                # this for the current summary contract.
                ids.append(self._vocab.get(normalized, self._vocab["[unk]"]))
            else:
                ids.extend(self._encode_word(token))
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """Decode token IDs back to text."""
        tokens: list[str] = []
        for tid in ids:
            tok = self._id_to_token.get(tid, "[unk]")
            if tok in SPECIAL_SET:
                tokens.append(tok)
            elif tok.endswith("</w>"):
                tokens.append(tok[:-4] + " ")
            else:
                if tokens:
                    tokens[-1] = tokens[-1] + tok
                else:
                    tokens.append(tok)
        return "".join(tokens).strip()

    def save(self, path: Path) -> None:
        """Save tokenizer to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vocab": self._vocab,
            "merges": [list(pair) for pair in self._merges],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        """Load tokenizer from JSON."""
        data = json.loads(path.read_text(encoding="utf-8"))
        vocab = {k: int(v) for k, v in data["vocab"].items()}
        merges = [tuple(pair) for pair in data["merges"]]
        return cls(vocab, merges)
