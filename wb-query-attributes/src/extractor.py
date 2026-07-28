# -*- coding: utf-8 -*-
"""
Rule-based извлечение атрибутов из поисковых запросов маркетплейса.

Два ключевых решения, к которым пришёл на данных:

1. Сопоставление словарных форм идёт только по границе слова, никогда по
   подстроке. Подстрочный поиск на реальных запросах даёт систематические
   ложные срабатывания: префикс "сер" от цвета "серый" ловит "серьги" и
   "миксер", "бел" ловит "мебели", "син" ловит "бусины" и "синтезатор".

2. Голое число не считается размером без контекста. Одно и то же "42" может
   быть размером обуви, диагональю, моделью или годом. Число трактуется как
   размер только если рядом есть явный контекст ("размер 42") или если
   категория запроса относится к одежде и обуви.

Многословные формы ("hugo boss", "для новорожденных") ищутся целой фразой:
иначе более короткое совпадение по одному слову перехватывает длинное.
"""
import re

from .dictionaries import BRANDS, CATEGORIES, COLORS, GENDER, UNITS


def _compile_lookup(source_dict):
    """
    Из словаря {канон: [формы]} строит список
    (регекс_формы, канон), отсортированный так, что более
    длинные (многословные / более специфичные) формы
    проверяются раньше более коротких. Это гарантирует, что
    "hugo boss" будет найден целиком раньше, чем отдельно
    "boss" внутри другого бренда, и что "для новорожденных"
    не будет "съедено" совпадением по одному слову "новорожденных".
    """
    pairs = []
    for canon, forms in source_dict.items():
        for form in forms:
            pairs.append((form, canon))
    pairs.sort(key=lambda x: -len(x[0]))
    compiled = [
        (re.compile(r"(?<![a-zа-я0-9])" + re.escape(form) + r"(?![a-zа-я0-9])"), canon)
        for form, canon in pairs
    ]
    return compiled


_BRAND_LOOKUP = _compile_lookup(BRANDS)
_CATEGORY_LOOKUP = _compile_lookup(CATEGORIES)
_COLOR_LOOKUP = _compile_lookup(COLORS)
_GENDER_LOOKUP = _compile_lookup(GENDER)

# суффиксы единиц измерения, отсортированные по убыванию длины,
# чтобы "литров" не был перехвачен как "л" на первом же символе
_UNIT_FORMS = sorted(
    ((form, canon) for canon, forms in UNITS.items() for form in forms),
    key=lambda x: -len(x[0]),
)
_UNIT_ALTERNATION = "|".join(re.escape(f) for f, _ in _UNIT_FORMS)
_SIZE_WITH_UNIT_RE = re.compile(
    r"(?<![a-zа-я0-9])(\d+(?:[.,]\d+)?)\s*(" + _UNIT_ALTERNATION + r")(?![a-zа-я])"
)
_UNIT_CANON = {form: canon for form, canon in _UNIT_FORMS}

# контекстные слова, после/перед которыми голое число трактуем как размер
_SIZE_CONTEXT_RE = re.compile(
    r"(размер\w*|рост\w*)\D{0,3}(\d{1,3})|(\d{1,3})\D{0,3}(размер\w*)"
)

# явный триггер "числа-как размера обуви/одежды" — числа в типичном
# диапазоне сеток без единицы измерения, но с категорией обуви/одежды
# рядом (проверяется на уровне extract_all, а не здесь)
_BARE_NUMBER_RE = re.compile(r"(?<![a-zа-я0-9.,])(\d{1,3})(?![a-zа-я0-9.,])")

ADULT_SIZE_RANGE = range(38, 63)      # типичная сетка размеров одежды/обуви взрослых (RU)
KIDS_SHOE_SIZE_RANGE = range(16, 38)  # детская сетка размеров обуви (RU)
LETTER_SIZE_TOKENS = {"xxs", "xs", "s", "m", "l", "xl", "xxl", "xxxl", "4xl", "5xl"}

_SHOE_CLOTHING_CATEGORIES = {
    "кроссовки", "ботинки", "туфли", "сапоги", "балетки", "тапки", "бутсы", "чешки",
    "футболка", "рубашка", "платье", "костюм", "штаны", "шорты", "куртка", "пальто",
    "халат", "свитер", "носки", "колготки", "белье", "купальник", "боди", "комбинезон",
    "гетры",
}


def _find_first(lookup, text):
    """Возвращает (канон, найденная_форма) первого совпадения по позиции в тексте, либо (None, None)."""
    best = None
    for pattern, canon in lookup:
        m = pattern.search(text)
        if m is not None:
            if best is None or m.start() < best[0]:
                best = (m.start(), canon, m.group())
    if best is None:
        return None, None
    return best[1], best[2]

def extract_brand(query: str):
    canon, _ = _find_first(_BRAND_LOOKUP, query)
    return canon


def extract_category(query: str):
    canon, _ = _find_first(_CATEGORY_LOOKUP, query)
    return canon


def extract_color(query: str):
    canon, _ = _find_first(_COLOR_LOOKUP, query)
    return canon


def extract_gender(query: str):
    canon, _ = _find_first(_GENDER_LOOKUP, query)
    return canon

def extract_values_with_unit(query: str):
    """
    Числа с явной единицей измерения: '55 дюймов' -> [('55', 'дюйм')],
    '1.5 кг' -> [('1.5', 'кг')]. Возвращает список всех найденных пар,
    т.к. в одном запросе их может быть несколько (например, габариты
    'шкаф 80 40 200 см').
    """
    results = []
    for match in _SIZE_WITH_UNIT_RE.finditer(query):
        value, unit_form = match.group(1), match.group(2)
        results.append((value, _UNIT_CANON[unit_form]))
    return results


def extract_size(query: str, category: str = None):
    """
    Извлекает размер с приоритетом:
      1. Явный контекст "размер N" / "N размер".
      2. Качественный размер: "большой размер" / "маленький размер"
         (частый паттерн для одежды plus-size — при профилировании
         нашлось ~50 запросов вида "больших размеров" без числа).
      3. Буквенный размер (S, M, L, XL, ...) как отдельный токен.
      4. Голое число в сетке одежды/обуви, ТОЛЬКО если категория
         запроса относится к одежде/обуви — иначе неотличимо от
         цены/модели/года. Отдельно проверяем взрослую (38-62) и
         детскую обувную (16-37) сетки; для детской сетки эвристика
         менее надёжна, т.к. диапазон пересекается с шумом
         (возраст, количество штук) — отмечено в анализе ошибок.
    Числа с единицей измерения (см, кг, л...) сюда не относятся —
    для них есть extract_values_with_unit; это отдельный атрибут
    (например "рост 110 см" для детской одежды по росту).
    """
    ctx_match = _SIZE_CONTEXT_RE.search(query)
    if ctx_match:
        value = ctx_match.group(2) or ctx_match.group(3)
        return value

    if re.search(r"\bбольш\w*\s+размер\w*|\bразмер\w*\s+больш\w*", query):
        return "большой"
    if re.search(r"\bмаленьк\w*\s+размер\w*|\bразмер\w*\s+маленьк\w*", query):
        return "маленький"

    tokens = re.findall(r"[a-zа-я0-9]+", query)

    if category in _SHOE_CLOTHING_CATEGORIES:
        for tok in tokens:
            if tok in LETTER_SIZE_TOKENS:
                return tok.upper()
        for tok in tokens:
            if not tok.isdigit():
                continue
            n = int(tok)
            if n in ADULT_SIZE_RANGE:
                return tok
            if category in {"кроссовки", "ботинки", "туфли", "сапоги", "балетки", "тапки", "бутсы", "чешки"} and n in KIDS_SHOE_SIZE_RANGE:
                return tok

    return None

def extract_all(query_norm: str) -> dict:
    """Запускает все экстракторы над одним нормализованным запросом."""
    category = extract_category(query_norm)
    brand = extract_brand(query_norm)
    color = extract_color(query_norm)
    gender = extract_gender(query_norm)
    size = extract_size(query_norm, category=category)
    values = extract_values_with_unit(query_norm)

    return {
        "brand": brand,
        "category": category,
        "color": color,
        "gender": gender,
        "size": size,
        "values_with_unit": values if values else None,
    }
