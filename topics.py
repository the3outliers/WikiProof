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


def _compile() -> list[tuple[str, re.Pattern[str]]]:
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for label, stems in _TOPIC_STEMS:
        alts = "|".join(re.escape(s) for s in sorted(stems, key=len, reverse=True))
        compiled.append((label, re.compile(rf"(?<!\w)(?:{alts})", re.IGNORECASE)))
    for label, words in _WHOLE_WORDS:
        alts = "|".join(re.escape(w) for w in words)
        # Case-sensitive: acronyms like NATO/IDF must not match ordinary words.
        compiled.append((label, re.compile(rf"(?<!\w)(?:{alts})(?!\w)")))
    return compiled


_PATTERNS = _compile()


def detect_sensitive_topics(title: str, added_text: str = "", deleted_text: str = "") -> dict[str, list[str]]:
    """Return {topic_label: [matched terms...]} for the title and changed text.

    Empty dict means the edit does not touch a politically sensitive topic.
    """
    haystack = "\n".join(part for part in (title, added_text, deleted_text) if part)
    if not haystack:
        return {}
    found: dict[str, list[str]] = {}
    for label, pattern in _PATTERNS:
        for m in pattern.finditer(haystack):
            term = m.group(0)
            terms = found.setdefault(label, [])
            if term.lower() not in (t.lower() for t in terms):
                terms.append(term)
            if len(terms) >= 5:
                break
    return found
