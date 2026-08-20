"""Infrastructure for verifying a quote is a verbatim slice of a source text.

Aligning a model's quote with the card it claims to have read. Every feature
that copies text out of a card rather than describing it needs this same
one-quote-verification concern, so it lives here rather than inside whichever
feature happened to write it first.

This module is the copy-only rule (R3.1) in executable form, and it is
deliberately free of every other concern: no HTTP, no language model, no store.
Give it a card and a quote and it answers with a :class:`Span` or a
:class:`RejectedValue`, and nothing else.

The important property is in :meth:`GroundedCard.find`: when a quote matches, the
text that gets stored is sliced out of the card, not copied from the model's
response. The model locates; it never supplies. That is what makes invention
impossible by construction rather than by good behaviour.

There is no fuzzy-matching tier and one must never be added. research.md 5:
a separately-flagged fuzzy tier is how the copy-only rule quietly stops meaning
anything.
"""

import html
import re
import unicodedata

from app.core.schemas import RejectedValue, Span

ENTITY = re.compile(r"&(?:#\d+|#[xX][0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]*);")
# An HTML tag or comment. Requires a letter or `/` after `<` so that prose
# like "a < b" is left alone.
TAG = re.compile(r"<!--.*?-->|</?[a-zA-Z][^>]*>", re.DOTALL)
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.MULTILINE)
# Built from code points rather than literal characters or \u escapes: both
# forms of zero-width text are unreliable to carry through tooling untouched.
ZERO_WIDTH = frozenset(chr(cp) for cp in (0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF))


def normalise(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Normalise for comparison, keeping a way back to the original offsets.

    Returns the normalised text and, for each of its characters, the ``(start,
    end)`` slice of the original text that produced it. The map is what lets a
    match in normalised space be reported as real offsets into the card.

    The transformations are the ones research.md 5 prescribes, applied
    identically to both sides of every comparison: HTML entities unescaped,
    zero-width characters dropped, whitespace runs collapsed to one space, and
    NFKC applied.

    HTML tags become a single space rather than disappearing. Cards that render
    their results table in HTML put markup inside a value -- Kimi K3's is
    ``<td>Kimi K3<br>(max)</td>`` -- and a model quoting the rendered cell says
    "Kimi K3\n(max)", which is not contiguous in the raw text. A tag is markup,
    not content, so skipping it is the same move as unescaping an entity.

    A *space* and not nothing, because deleting the tag would join
    ``<td>93.5</td><td>92.6</td>`` into "93.592.6" and let a model quote a
    number the card never states. Separating is safe; welding is not.

    ponytail: NFKC is applied per character rather than to the whole string, so
    that one output character always traces to one input span. Whole-string NFKC
    can compose across character boundaries, which would make the map ambiguous.
    The difference only shows up for combining sequences, which model cards do
    not meaningfully contain.
    """
    out: list[str] = []
    offsets: list[tuple[int, int]] = []
    i = 0
    length = len(text)

    while i < length:
        entity = ENTITY.match(text, i)
        if entity:
            for char in html.unescape(entity.group(0)):
                out.append(char)
                offsets.append((i, entity.end()))
            i = entity.end()
            continue

        tag = TAG.match(text, i)
        if tag:
            # Collapses with any adjacent whitespace, as a space would.
            if out and out[-1] != " ":
                out.append(" ")
                offsets.append((i, tag.end()))
            i = tag.end()
            continue

        char = text[i]

        if char in ZERO_WIDTH:
            i += 1
            continue

        if char.isspace():
            run = i
            while run < length and text[run].isspace():
                run += 1
            out.append(" ")
            offsets.append((i, run))
            i = run
            continue

        for produced in unicodedata.normalize("NFKC", char):
            out.append(produced)
            offsets.append((i, i + 1))
        i += 1

    return "".join(out), offsets


class GroundedCard:
    """One model card, normalised once, ready to answer locate requests.

    Normalisation is done in the constructor because a card is searched once per
    extracted field and there can be dozens of them.
    """

    def __init__(self, text: str):
        self.text = text
        self.normalised, self._offsets = normalise(text)
        self._headings = [(m.start(), m.group(2)) for m in HEADING.finditer(text)]

    def section_at(self, offset: int) -> str:
        """R3.3 - the enclosing markdown heading, or ``""`` above the first one."""
        section = ""
        for start, title in self._headings:
            if start > offset:
                break
            section = title
        return section

    def find(self, quote: str, *, field: str) -> Span | RejectedValue:
        """Locate ``quote`` in the card, or say why it could not be located."""
        needle, _ = normalise(quote)
        needle = needle.strip()
        if not needle:
            return RejectedValue(field=field, proposed=quote, reason="empty")

        positions = self._occurrences_of(needle)
        if not positions:
            return RejectedValue(field=field, proposed=quote, reason=self._why_not(needle))

        position = positions[0]
        start = self._offsets[position][0]
        end = self._offsets[position + len(needle) - 1][1]
        return Span(
            text=self.text[start:end],
            start=start,
            end=end,
            section=self.section_at(start),
            occurrences=len(positions),
        )

    def _occurrences_of(self, needle: str) -> list[int]:
        """Every position where ``needle`` appears as a whole token, in order.

        Plain substring matching accepts ``52.8`` against a card that says
        ``52.80``, and ``vLLM`` against ``vLLMv2``. Those are not quotations --
        the model produced a string the card never contains as a unit, and for
        benchmark scores that is precisely how a hallucinated number slips
        through: ``9.5`` nests inside ``19.54``.

        A position counts only when neither edge continues a word: an
        alphanumeric character on the outside touching an alphanumeric character
        on the inside means the token carries on. Punctuation is a real boundary,
        so ``NVFP4`` still matches inside ``NVFP4-A16``.

        Every use of "does this appear in the card" goes through here, so the
        count and the diagnosis can never disagree with the match itself.
        """
        found: list[int] = []
        position = self.normalised.find(needle)
        while position >= 0:
            before = self.normalised[position - 1] if position else ""
            after_index = position + len(needle)
            after = self.normalised[after_index] if after_index < len(self.normalised) else ""
            starts_clean = not (before.isalnum() and needle[0].isalnum())
            ends_clean = not (after.isalnum() and needle[-1].isalnum())
            if starts_clean and ends_clean:
                found.append(position)
            position = self.normalised.find(needle, position + 1)
        return found

    def _why_not(self, needle: str) -> str:
        """Distinguish an invention from a quote stitched out of separate places.

        A model that rejoins two table cells produces a string that never existed
        contiguously, but whose parts are all present. research.md 5 names this
        as a real failure mode, and it deserves a different label from a value
        that simply is not in the card.
        """
        parts = [part for part in needle.split(" ") if part]
        if len(parts) > 1 and all(self._occurrences_of(part) for part in parts):
            return "not_contiguous"
        return "no_match"


def _locate(
    grounded: GroundedCard, value: str | None, field: str, rejected: list[RejectedValue]
) -> Span | None:
    """One verification. Anything that is not a Span is recorded and dropped."""
    if value is None:
        return None
    found = grounded.find(value, field=field)
    if isinstance(found, Span):
        return found
    rejected.append(found)
    return None
