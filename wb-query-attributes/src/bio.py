# -*- coding: utf-8 -*-
"""
Перевод между двумя представлениями разметки: словарь атрибутов
{brand, category, color, gender, size} и последовательность BIO-тегов по токенам.

BIO нужен, потому что NER-модель обучается на разметке уровня токена, а
rule-based экстрактор и ручная разметка работают на уровне атрибута.
"""
import re

from .extractor import (
    _BRAND_LOOKUP,
    _CATEGORY_LOOKUP,
    _COLOR_LOOKUP,
    _GENDER_LOOKUP,
    extract_brand,
    extract_category,
    extract_color,
    extract_gender,
)

ATTRS = ["brand", "category", "color", "gender", "size"]

LABEL_LIST = [
    "O",
    "B-BRAND", "I-BRAND",
    "B-CATEGORY", "I-CATEGORY",
    "B-COLOR", "I-COLOR",
    "B-GENDER", "I-GENDER",
    "B-SIZE", "I-SIZE",
]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

TAG_TO_ATTR = {
    "BRAND": "brand",
    "CATEGORY": "category",
    "COLOR": "color",
    "GENDER": "gender",
    "SIZE": "size",
}

# канонизаторы: приводят найденный в тексте фрагмент к тому же виду, в котором
# значение возвращает rule-based экстрактор ("черные" -> "черный")
CANON_FN = {
    "brand": extract_brand,
    "category": extract_category,
    "color": extract_color,
    "gender": extract_gender,
}


def canonize(attr, value):
    """
    Приводит значение атрибута к канонической форме словаря.

    Нужно при сравнении предсказаний с разметкой: NER возвращает фрагмент
    исходного текста ("черные", "штиль"), а rule-based и ручная разметка
    оперируют канонами ("черный", "stihl"). Без этого шага одно и то же
    значение в разной форме считается ошибкой.

    Для size канонизации нет: там значение и есть текст ("42", "XL").
    """
    if value is None:
        return None
    fn = CANON_FN.get(attr)
    if fn is None:
        return value
    canon = fn(value)
    return canon if canon is not None else value


def _all_spans(lookup, query):
    """
    Все непересекающиеся совпадения словаря в запросе, слева направо.

    При пересечении выигрывает более длинное совпадение: "hugo boss" должно
    размечаться одной сущностью из двух токенов, а не двумя отдельными
    брендами "hugo" и "boss".
    """
    spans = []
    for pattern, _canon in lookup:
        for m in pattern.finditer(query):
            spans.append((m.start(), m.end()))

    # сначала по позиции, при равной позиции — длинное вперёд
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))

    kept = []
    for start, end in spans:
        if any(start < k_end and end > k_start for k_start, k_end in kept):
            continue
        kept.append((start, end))
    return sorted(kept)


def rule_based_to_bio(query):
    """
    Переводит запрос в BIO-разметку по реальным позициям словарных совпадений.

    Позиции берутся из regex-матчей, а не сравнением "канон == токен": иначе
    словоформа "черные" не разметится как COLOR, потому что канон "черный"
    не входит в неё подстрокой.

    Размечаются все непересекающиеся совпадения, а не только первое: запрос
    "белая рубашка с черными пуговицами" содержит два цвета, и разметка
    только первого учила бы модель недобирать сущности.
    """
    tokens = query.split()
    if not tokens:
        return None

    positions, cursor = [], 0
    for tok in tokens:
        start = query.index(tok, cursor)
        positions.append((start, start + len(tok)))
        cursor = start + len(tok)

    tags = ["O"] * len(tokens)

    for lookup, tag_name in (
        (_BRAND_LOOKUP, "BRAND"),
        (_CATEGORY_LOOKUP, "CATEGORY"),
        (_COLOR_LOOKUP, "COLOR"),
        (_GENDER_LOOKUP, "GENDER"),
    ):
        for span_start, span_end in _all_spans(lookup, query):
            first = True
            for i, (tstart, tend) in enumerate(positions):
                if tstart < span_end and tend > span_start and tags[i] == "O":
                    tags[i] = f"{'B' if first else 'I'}-{tag_name}"
                    first = False

    return {"tokens": tokens, "tags": tags}


def bio_to_attr_dict(tokens, tags, apply_canon=True):
    """
    BIO-последовательность -> {attr: value}. Берётся первая сущность каждого типа.
    """
    result = {a: None for a in ATTRS}
    current_attr, current_tokens = None, []

    def flush():
        if current_attr and result[current_attr] is None:
            result[current_attr] = " ".join(current_tokens)

    for tok, tag in list(zip(tokens, tags)) + [("", "O")]:
        if tag.startswith("B-"):
            flush()
            current_attr = TAG_TO_ATTR.get(tag[2:])
            current_tokens = [tok]
        elif tag.startswith("I-") and current_attr == TAG_TO_ATTR.get(tag[2:]):
            current_tokens.append(tok)
        else:
            flush()
            current_attr, current_tokens = None, []

    if apply_canon:
        for attr in ATTRS:
            result[attr] = canonize(attr, result[attr])
    return result


def clean_subword(text):
    """Убирает артефакты subword-токенизатора из фрагмента, возвращённого пайплайном."""
    return re.sub(r"\s*##", "", text).strip()
