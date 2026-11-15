"""Politically sensitive topic detection used to route edits to AI propaganda review.

This does NOT change an edit's risk score. It only decides which live edits get
sent to the AI analyzer even when their behavioral score is low, because
propaganda edits are often made by established accounts with no other red flags.

Matching is transparent keyword/stem matching over the article title and the
changed text, in the stream's default languages (en, de, fi, ru, sv). Stems
match word prefixes so inflected forms are caught (e.g. "Ukrain" -> Ukraine,
Ukrainian, Ukrainan, Украина via "Украин").
"""

from __future__ import annotations

import re

# Each entry: (topic label, list of word-prefix stems). Stems are matched at a
# word start, case-insensitively. Very short/ambiguous words use WHOLE_WORDS.
_TOPIC_STEMS: list[tuple[str, list[str]]] = [
    ("Russia–Ukraine war", [
        "Ukrain", "Kremlin", "Putin", "Zelensk", "Crimea", "Donbas", "Donetsk", "Luhansk",
        "Mariupol", "Bucha", "Kherson", "Zaporizh", "Lukashenk", "Russo-Ukrain",
        "special military operation", "Spezialoperation", "erityisoperaatio", "specialoperation",
        "Kreml", "Venäj", "Ryssland",
        "Украин", "Кремл", "Путин", "Зеленск", "Крым", "Донбасс", "Донецк", "Луганск",
        "Мариупол", "Буча", "Херсон", "Запорож", "Лукашенк", "спецоперац",
        "специальной военной операц", "специальная военная операц",
    ]),
    ("Israel–Palestine", [
        "Israel", "Palestin", "Palästin", "Palestiin", "Gaza", "Hamas", "Hezbollah",
        "West Bank", "Zionis", "Netanyahu", "October 7",
        "Израил", "Палестин", "сектор Газа", "ХАМАС", "Хезболл",
    ]),
    ("War crimes & atrocities", [
        "genocid", "ethnic cleansing", "war crime", "massacre", "deportation",
        "Völkermord", "Kriegsverbrech", "Massaker", "Vertreibung",
        "kansanmurh", "sotarikos", "joukkomurh",
        "folkmord", "krigsbrott", "massaker",
        "геноцид", "военн преступлен", "военные преступлен", "массов убийств", "депортац",
        "Holocaust", "Holokaust", "Холокост",
    ]),
    ("Invasion & occupation", [
        "invasion", "invaded", "annexation", "annexed", "annexing", "occupied territor", "occupation of",
        "Annexion", "Besatzung", "hyökkäy", "miehity", "annekto", "ockupation", "ockuper", "annekter",
        "вторжени", "аннекс", "оккупац",
    ]),
    ("Authoritarian states & conflicts", [
        "Xinjiang", "Uyghur", "Uigur", "Tibet", "Taiwan", "Hong Kong", "Xi Jinping",
        "Chinese Communist Party", "Tiananmen", "Kashmir", "Nagorno", "Karabakh",
        "Assad", "Khamenei", "Houthi", "Belarus", "Weißrussland", "Valko-Venäj", "Vitryssland",
        "Синьцзян", "Уйгур", "Тайван", "Си Цзиньпин", "Карабах", "Асад", "Беларус", "Белорусс",
    ]),
    ("Extremism & propaganda", [
        "propagand", "Nazi", "neo-Nazi", "fascis", "Faschis", "fasis", "natsi", "nazis",
        "disinformation", "Desinformation", "disinformaatio", "desinformation",
        "нацист", "фашист", "бандеровц", "пропаганд", "дезинформац",
        "terroris", "Terror", "terrori",
        "террорист",
    ]),
    ("Elections & governments", [
        "election", "electoral", "referendum", "coup d",
        "Bundestagswahl", "Wahlkampf", "Wahlbetrug", "Präsidentschaftswahl", "Volksabstimmung",
        "vaali", "kansanäänestys", "eduskuntavaal", "presidentinvaal",
        "riksdagsval", "presidentval", "folkomröstning",
        "выбор президент", "президентские выбор", "парламентские выбор", "референдум",
    ]),
]

# Short words that would over-match as prefixes; matched as whole words only.
_WHOLE_WORDS: list[tuple[str, list[str]]] = [
    ("Russia–Ukraine war", ["Russia", "Russian", "Russland", "Россия", "России", "СВО", "Krim", "Krym"]),
    ("Elections & governments", ["NATO", "Nato", "НАТО", "IDF", "AfD", "ЦАХАЛ",
                                 "regime", "regimes", "Regime", "Regimes"]),
]


# Terms so common in ordinary articles (place lists, biographies, every article on
# Russian Wikipedia) that they only count when they appear in the article TITLE.
_TITLE_ONLY: list[tuple[str, list[str]]] = [
    ("Russia–Ukraine war", ["Russia", "Russian", "Russland", "Россия", "России", "Российск", "Ryssland", "Venäj"]),
    ("Authoritarian states & conflicts", ["Taiwan", "Тайван", "Hong Kong", "Belarus", "Беларус", "Белорусс"]),
]


def _alts(terms: list[str]) -> str:
    return "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))


def _compile(title_only_terms: set[str]) -> list[tuple[str, re.Pattern[str]]]:
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for label, stems in _TOPIC_STEMS:
        stems = [s for s in stems if s not in title_only_terms]
        if not stems:
            continue
        compiled.append((label, re.compile(rf"(?<!\w)(?:{_alts(stems)})", re.IGNORECASE)))
    for label, words in _WHOLE_WORDS:
        words = [w for w in words if w not in title_only_terms]
        if not words:
            continue
        # Case-sensitive: acronyms like NATO/IDF must not match ordinary words.
        compiled.append((label, re.compile(rf"(?<!\w)(?:{_alts(words)})(?!\w)")))
    return compiled


_TITLE_ONLY_TERMS = {t for _, terms in _TITLE_ONLY for t in terms}
_TEXT_PATTERNS = _compile(_TITLE_ONLY_TERMS)
_TITLE_PATTERNS = _compile(set()) + [
    (label, re.compile(rf"(?<!\w)(?:{_alts(terms)})", re.IGNORECASE)) for label, terms in _TITLE_ONLY
]


def changed_segments(revision: dict) -> list[str]:
    """The words that actually changed in a revision (MediaWiki inline diff highlights).

    Falls back to whole changed lines when highlights are unavailable.
    """
    segs = list(revision.get("added_highlights") or []) + list(revision.get("deleted_highlights") or [])
    if not segs:
        segs = list(revision.get("added_lines") or []) + list(revision.get("deleted_lines") or [])
    return [s for s in segs if s and s.strip()]


def detect_sensitive_topics(title: str, changed: list[str] | None = None) -> dict[str, list[str]]:
    """Return {topic_label: [matched terms...]} for the title and the changed wording.

    Only the article title and the words that actually changed are searched, so a
    sensitive word elsewhere in an edited paragraph does not trigger review.
    Empty dict means the edit does not touch a politically sensitive topic.
    """
    found: dict[str, list[str]] = {}
    _scan(title or "", _TITLE_PATTERNS, found)
    _scan("\n".join(changed or []), _TEXT_PATTERNS, found)
    return found


def _scan(haystack: str, patterns: list[tuple[str, re.Pattern[str]]], found: dict[str, list[str]]) -> None:
    if not haystack:
        return
    for label, pattern in patterns:
        for m in pattern.finditer(haystack):
            term = m.group(0)
            terms = found.setdefault(label, [])
            if term.lower() not in (t.lower() for t in terms):
                terms.append(term)
            if len(terms) >= 5:
                break
