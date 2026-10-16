"""Probe datasets with mechanically decidable ground truth.

The hard problem in evaluating a decision model is that most ground truth is
opinion. This dataset sidesteps it: every item's truth is computed in Python
from the state itself (integer comparisons, parity, substring containment, word
counts, range membership). No human labels, no judge model, no leaks — and the
same items work against any endpoint.

Two consequences worth stating plainly:

* The tasks are *easy*. They measure whether a model grounds on the state at all,
  not whether it is smart. A model that fails them is broken, not merely weak.
* Difficulty is tunable via the per-group ``difficulty`` knob (see
  :func:`build_noul_items`) so the same harness can be run at a point where
  accuracy is neither 100% nor chance.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

VOWELS = "aeiou"


@dataclass(frozen=True)
class NoulItem:
    id: str
    state: Any
    instructions: str
    truth: bool
    group: str
    criteria: dict[str, str] | None = None
    paraphrase: str | None = None

    @property
    def question(self) -> dict[str, Any]:
        q: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if self.criteria:
            q["criteria"] = self.criteria
        return q


@dataclass(frozen=True)
class ChoiceItem:
    id: str
    state: Any
    instructions: str
    criteria: dict[str, Any]
    truth: str
    group: str

    @property
    def question(self) -> dict[str, Any]:
        return {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": self.criteria,
        }


@dataclass(frozen=True)
class ScoreItem:
    id: str
    state: Any
    instructions: str
    levels: list[str]
    truth_level: int
    group: str

    @property
    def question(self) -> dict[str, Any]:
        return {
            "type": "score",
            "instructions": self.instructions,
            "criteria": list(self.levels),
        }


@dataclass
class Dataset:
    nouls: list[NoulItem] = field(default_factory=list)
    choices: list[ChoiceItem] = field(default_factory=list)
    scores: list[ScoreItem] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.nouls) + len(self.choices) + len(self.scores)


# --------------------------------------------------------------------- generators

_WORDS = (
    "audit ledger payout refund invoice settlement chargeback dispute receipt "
    "gateway transfer balance statement merchant payout fee reversal"
).split()


def _filler(n: int, rng: random.Random) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(n))


def build_noul_items(rng: random.Random, per_group: int = 6, difficulty: str = "mixed") -> list[NoulItem]:
    """Build noul probes across four groups with Python-decided truth.

    ``difficulty`` shifts the margin between the two operands: ``easy`` uses
    far-apart values, ``close`` uses near-ties where a careless model flips.
    """
    items: list[NoulItem] = []
    close = difficulty == "close"

    for i in range(per_group):
        margin = rng.randint(1, 2) if close else rng.randint(5, 90)
        left = rng.randint(-50, 50)
        right = left + (margin if i % 2 == 0 else -margin)
        items.append(
            NoulItem(
                id=f"noul_cmp_{i}",
                state={"left": left, "right": right},
                instructions="Is `left` greater than `right`?",
                truth=left > right,
                group="compare",
                criteria={"true": "left is strictly greater", "false": "left is not greater"},
                paraphrase="Does the number in `left` exceed the number in `right`?",
            )
        )

    for i in range(per_group):
        n = rng.randrange(0, 10_000)
        items.append(
            NoulItem(
                id=f"noul_parity_{i}",
                state={"number": n},
                instructions="Is `number` even?",
                truth=n % 2 == 0,
                group="parity",
                criteria={"true": "divisible by two with no remainder", "false": "not divisible by two"},
            )
        )

    for i in range(per_group):
        needle = rng.choice(_WORDS)
        present = i % 2 == 0
        haystack = _filler(rng.randint(6, 14), rng)
        if present:
            words = haystack.split()
            words.insert(rng.randrange(len(words)), needle)
            haystack = " ".join(words)
        else:
            # Guarantee absence: need{...} never appears in the filler vocabulary.
            needle = needle + "zz"
        items.append(
            NoulItem(
                id=f"noul_substr_{i}",
                state={"haystack": haystack, "needle": needle},
                instructions="Does `haystack` contain the substring `needle`?",
                truth=needle in haystack,
                group="substring",
                paraphrase="Is the text in `needle` present inside `haystack`?",
            )
        )

    for i in range(per_group):
        n_words = rng.randint(4, 20)
        threshold = n_words + rng.choice([-2, -1, 1, 2])
        text = _filler(n_words, rng)
        items.append(
            NoulItem(
                id=f"noul_count_{i}",
                state={"text": text, "threshold": threshold},
                instructions="Does `text` contain more than `threshold` words?",
                truth=len(text.split()) > threshold,
                group="count",
                criteria={
                    "true": "word count is strictly greater than the threshold",
                    "false": "word count is at or below the threshold",
                },
            )
        )

    rng.shuffle(items)
    return items


def build_choice_items(rng: random.Random) -> list[ChoiceItem]:
    items: list[ChoiceItem] = []

    for i in range(4):
        n = rng.choice([-17, -3, 0, 4, 9, 12, 45, 900])
        if n < 0:
            truth = "below zero"
        elif n <= 9:
            truth = "zero to nine"
        else:
            truth = "ten or more"
        items.append(
            ChoiceItem(
                id=f"choice_range_{i}",
                state={"number": n},
                instructions="Which range does `number` fall into?",
                criteria={
                    "below zero": "the number is less than zero",
                    "zero to nine": "the number is from zero through nine inclusive",
                    "ten or more": "the number is ten or greater",
                },
                truth=truth,
                group="range",
            )
        )

    for i in range(4):
        a, b, c = rng.sample(range(1, 99), 3)
        largest = max((a, "a"), (b, "b"), (c, "c"))
        items.append(
            ChoiceItem(
                id=f"choice_largest_{i}",
                state={"a": a, "b": b, "c": c},
                instructions="Which of `a`, `b`, and `c` holds the largest value?",
                criteria={"a": "the value in `a`", "b": "the value in `b`", "c": "the value in `c`"},
                truth=largest[1],
                group="largest",
            )
        )

    for i, word in enumerate(["apple", "otter", "clam", "eagle", "sand", "igloo"]):
        starts_vowel = word[0].lower() in VOWELS
        items.append(
            ChoiceItem(
                id=f"choice_vowel_{i}",
                state={"word": word},
                instructions="Does `word` begin with a vowel?",
                criteria={"yes": "the first letter is a vowel", "no": "the first letter is a consonant"},
                truth="yes" if starts_vowel else "no",
                group="vowel",
            )
        )

    rng.shuffle(items)
    return items


_DENSITY_LEVELS = ["empty", "sparse", "moderate", "dense"]


def _density_level(n_words: int) -> int:
    if n_words == 0:
        return 0
    if n_words <= 3:
        return 1
    if n_words <= 8:
        return 2
    return 3


def build_score_items(rng: random.Random) -> list[ScoreItem]:
    """Ordered-score probes. ``truth_level`` is monotone in word count by construction."""
    items: list[ScoreItem] = []
    counts = [0, 2, 3, 5, 8, 12, 18]
    for i, n_words in enumerate(counts):
        items.append(
            ScoreItem(
                id=f"score_density_{i}",
                state={"text": _filler(n_words, rng)},
                instructions="How dense is the word list in `text`?",
                levels=list(_DENSITY_LEVELS),
                truth_level=_density_level(n_words),
                group="density",
            )
        )
    return items


def build_dataset(
    seed: int = 20261016,
    per_group: int = 6,
    difficulty: str = "mixed",
) -> Dataset:
    rng = random.Random(seed)
    return Dataset(
        nouls=build_noul_items(rng, per_group=per_group, difficulty=difficulty),
        choices=build_choice_items(rng),
        scores=build_score_items(rng),
    )


def monotone_score_pairs(items: list[ScoreItem]) -> list[tuple[ScoreItem, ScoreItem]]:
    """Pairs of score items whose true levels are ordered, for a monotonicity check."""
    ordered = sorted(items, key=lambda it: it.truth_level)
    return list(zip(ordered, ordered[1:]))
