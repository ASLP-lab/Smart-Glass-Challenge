# Copyright 2025 SmartGlasses Project Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Text normalization utilities for Chinese ASR evaluation.

This module provides text normalization functions specifically designed for
Chinese automatic speech recognition (ASR) evaluation, including CER (Character
Error Rate) and WER (Word Error Rate) computation preprocessing.

The normalization pipeline applies the following transformations in order:
  1. Unicode NFC normalization — canonical composition of combining characters.
  2. Stripping invisible control characters — zero-width spaces, BOM, etc.
  3. Arabic digit to Chinese character conversion — value-based mapping
     for integer parts (e.g., '200' -> '二百', '12' -> '十二') and
     digit-by-digit mapping for fractional parts (e.g., '.14' -> '一四'),
     matching the convention used in Chinese reference transcripts.
  4. Punctuation removal — both fullwidth Chinese punctuation (，。？！ etc.)
     and halfwidth English punctuation (,.!? etc.).
  5. Special symbol removal — geometric shapes, bullets, etc. commonly found
     in synthetic QA options.
  6. Whitespace normalization — collapse consecutive spaces, trim leading
     and trailing whitespace.

Typical usage:

  >>> from text_normalization import normalize_text
  >>> text = '炒饭用生抽，嗯用生抽，嗯清淡点，这蛋味儿能出来。'
  >>> normalize_text(text, remove_spaces=True)
  '炒饭用生抽嗯用生抽嗯清淡点这蛋味儿能出来'


Reference:
  https://github.com/alanshaoTT/MLC-SLM-2nd-Task1-Baseline/blob/main/text_normalization_2nd.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Chinese fullwidth punctuation to remove.
# Each entry is annotated with its Unicode character name for maintainability.
_CN_PUNCTUATION = (
    # Basic stops
    '\u3001'    # 、  IDEOGRAPHIC COMMA
    '\u3002'    # 。  IDEOGRAPHIC FULL STOP
    '\uff0c'    # ，  FULLWIDTH COMMA
    '\uff1f'    # ？  FULLWIDTH QUESTION MARK
    '\uff01'    # ！  FULLWIDTH EXCLAMATION MARK
    '\uff1a'    # ：  FULLWIDTH COLON
    '\uff1b'    # ；  FULLWIDTH SEMICOLON
    # Quotation marks
    '\u2018'    # '   LEFT SINGLE QUOTATION MARK
    '\u2019'    # '   RIGHT SINGLE QUOTATION MARK
    '\u201c'    # "   LEFT DOUBLE QUOTATION MARK
    '\u201d'    # "   RIGHT DOUBLE QUOTATION MARK
    '\uff08'    # （  FULLWIDTH LEFT PARENTHESIS
    '\uff09'    # ）  FULLWIDTH RIGHT PARENTHESIS
    '\u3010'    # 【  LEFT BLACK LENTICULAR BRACKET
    '\u3011'    # 】  RIGHT BLACK LENTICULAR BRACKET
    '\u300a'    # 《  LEFT DOUBLE ANGLE BRACKET
    '\u300b'    # 》  RIGHT DOUBLE ANGLE BRACKET
    '\u300c'    # 「  LEFT CORNER BRACKET
    '\u300d'    # 」  RIGHT CORNER BRACKET
    '\u300e'    # 『  LEFT WHITE CORNER BRACKET
    '\u300f'    # 』  RIGHT WHITE CORNER BRACKET
    # Connectors and ellipsis
    '\u2014'    # —  EM DASH
    '\u2013'    # –  EN DASH
    '\u2026'    # …  HORIZONTAL ELLIPSIS
    '\uff5e'    # ～  FULLWIDTH TILDE
    '\u00b7'    # ·  MIDDLE DOT
    '\u2022'    # •  BULLET
    '\uff20'    # ＠  FULLWIDTH COMMERCIAL AT
)

# Halfwidth English punctuation. Hyphen (-) and en-dash (–) are included
# because they are not meaningful in Chinese ASR transcripts.
_EN_PUNCTUATION = r'!"#$%&()*+,\-./:;<=>?@[\\\]^_`{|}~–'

# Special geometric / decorative symbols frequently appearing in synthetic
# QA distractors but absent from natural speech transcripts.
_SPECIAL_SYMBOLS = r'●■□◆◇▲△▼▽★☆♀♂⊙◎○●◎※'

# Unicode invisible / zero-width characters that should always be stripped.
_INVISIBLE_CHARS_PATTERN = re.compile(
    '[\u200b\u200c\u200d\u200e\u200f\u2060-\u2064\ufeff]'
)

# Combined punctuation removal pattern (pre-compiled for performance).
_ALL_PUNCTUATION = _CN_PUNCTUATION + _EN_PUNCTUATION + _SPECIAL_SYMBOLS
_PUNCTUATION_PATTERN = re.compile(f'[{re.escape(_ALL_PUNCTUATION)}]')

# Digit-to-Chinese mapping (for fractional part and single digits).
_DIGIT_TO_CHINESE: Dict[str, str] = {
    '0': '零',
    '1': '一',
    '2': '二',
    '3': '三',
    '4': '四',
    '5': '五',
    '6': '六',
    '7': '七',
    '8': '八',
    '9': '九',
}

# Units for value-based number conversion: 十, 百, 千, 万, 亿.
# Each level corresponds to 10^{level+1}.
_CHINESE_UNITS = ['', '十', '百', '千']

# Pattern matching a contiguous digit sequence (integer or decimal).
_NUMBER_PATTERN = re.compile(r'\d+(?:\.\d+)?')

# Pattern matching a single Chinese character (CJK Unified Ideograph).
_CHINESE_CHAR_PATTERN = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')

# Unicode ranges for Chinese characters (CJK Unified Ideographs).
_CJK_RANGES = (
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x3400, 0x4DBF),    # CJK Unified Ideographs Extension A
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _int_to_chinese(num: int) -> str:
    """Convert an integer to value-based Chinese representation.

    Uses the standard Chinese numerical system with units 十/百/千/万/亿.
    Handles zero placeholders (零) correctly for numbers like 101 -> 一百零一.

    Args:
        num: A non-negative integer.

    Returns:
        Chinese string. Examples: 12 -> '十二', 200 -> '二百', 101 -> '一百零一'.

    Raises:
        ValueError: If num is negative.
    """
    if num < 0:
        raise ValueError(f'Cannot convert negative number: {num}')
    if num == 0:
        return '零'

    # Segment-based conversion: process each 4-digit segment (亿/万/个).
    segments: List[str] = []
    segment_names = ['', '万', '亿']
    segment_index = 0

    while num > 0:
        segment = num % 10000
        num //= 10000

        if segment == 0:
            segment_index += 1
            continue

        # Convert a 4-digit segment.
        seg_str = ''
        thousand = segment // 1000
        hundred = (segment % 1000) // 100
        ten = (segment % 100) // 10
        one = segment % 10

        if thousand > 0:
            seg_str += _DIGIT_TO_CHINESE[str(thousand)] + '千'
        if hundred > 0:
            seg_str += _DIGIT_TO_CHINESE[str(hundred)] + '百'
        elif thousand > 0 and (ten > 0 or one > 0):
            # Zero in hundreds place, e.g., 1010 -> 一千零一十
            seg_str += '零'

        if ten > 0:
            seg_str += _DIGIT_TO_CHINESE[str(ten)] + '十'
        elif hundred > 0 and one > 0:
            # Zero in tens place, e.g., 101 -> 一百零一
            seg_str += '零'

        if one > 0:
            seg_str += _DIGIT_TO_CHINESE[str(one)]
        # Handle special case: 10 -> 十 (not 一十), 100 -> 一百
        # But 10 in the middle of a number: 110 -> 一百一十 (correct)

        segment_name = segment_names[segment_index] if segment_index < len(segment_names) else f'段{segment_index}'
        if seg_str:
            segments.append(seg_str + segment_name)

        segment_index += 1

    result = ''.join(reversed(segments))

    # Fix leading 一十 -> 十 (e.g., 12 -> 十二, not 一十二).
    # This applies only when 一十 starts the entire result.
    if result.startswith('一十'):
        result = result[1:]  # 一十二 -> 十二, 一十万 -> 十万

    return result


def _decimal_to_chinese(integer_part: int, fractional_part: str) -> str:
    """Convert a decimal number to Chinese representation.

    The integer part uses value-based conversion, while the fractional part
    uses digit-by-digit mapping (standard Chinese convention).

    Args:
        integer_part: The integer part (before decimal point).
        fractional_part: The fractional part as a string of digits (after '.').

    Returns:
        Chinese string. Example: (3, '14') -> '三点一四'.
    """
    int_str = _int_to_chinese(integer_part) if integer_part > 0 else '零'
    frac_str = ''.join(_DIGIT_TO_CHINESE[d] for d in fractional_part)
    return f'{int_str}点{frac_str}'


def convert_digits(text: str) -> str:
    """Convert Arabic digits to Chinese characters using value-based mapping.

    The integer part of a number is converted using the standard Chinese
    numerical system (e.g., 200 -> '二百', 12 -> '十二'), while the
    fractional part uses digit-by-digit mapping (e.g., .14 -> '一四').

    This matches the convention used in the reference transcripts, where
    numbers are written as Chinese value words (三百, 十二) rather than
    digit sequences.

    Args:
        text: Input text that may contain Arabic digits.

    Returns:
        Text with all Arabic digit sequences converted to Chinese characters.

    Examples:
        >>> convert_digits('200块')
        '二百块'
        >>> convert_digits('12块5毛')
        '十二块五毛'
        >>> convert_digits('3.14')
        '三点一四'
        >>> convert_digits('2026年')
        '二零二六年'
        >>> convert_digits('第1名')
        '第一名'
    """
    def _replace(match: re.Match) -> str:
        num_str = match.group(0)
        if '.' in num_str:
            int_part_str, frac_part = num_str.split('.', 1)
            int_part = int(int_part_str) if int_part_str else 0
            return _decimal_to_chinese(int_part, frac_part)
        else:
            return _int_to_chinese(int(num_str))

    return _NUMBER_PATTERN.sub(_replace, text)


def space_chinese_text(text: str) -> str:
    """Insert a space between every Chinese character.

    Chinese characters are space-separated to facilitate character-level
    alignment for CER evaluation. Non-Chinese tokens (English words, digits)
    remain contiguous.

    Args:
        text: Input text that may contain Chinese characters.

    Returns:
        Text with spaces inserted between Chinese characters. Non-Chinese
        segments are preserved as-is.

    Examples:
        >>> space_chinese_text('炒饭用生抽')
        '炒 饭 用 生 抽'
        >>> space_chinese_text('一共二百元')
        '一 共 二 百 元'
        >>> space_chinese_text('yeah recording uh')
        'yeah recording uh'
        >>> space_chinese_text('一共200元yeah')
        '一 共 200 元 yeah'
    """
    text = _CHINESE_CHAR_PATTERN.sub(r' \g<0> ', text)
    return re.sub(r'\s+', ' ', text).strip()


def normalize_text(
    text: str,
    convert_digit: bool = True,
    space_chinese: bool = True,
    remove_spaces: bool = False,
) -> str:
    """Normalize a single text string for Chinese ASR CER/WER evaluation.

    The normalization pipeline applies steps in a fixed order to ensure
    deterministic and reproducible results. The same function should be applied
    to both reference (ground-truth) and hypothesis (ASR output) texts before
    computing CER or WER.

    Args:
        text: Input text string to normalize. Typically contains Chinese
            dialogue text, ASR transcripts, or QA question/option strings.
        convert_digit: When True (default), Arabic digits are converted to
            Chinese characters via digit-by-digit mapping. Set to False when
            digits should be preserved (e.g., for certain QA analysis tasks).
        space_chinese: When True (default), insert a space between every
            Chinese character for character-level CER alignment.
        remove_spaces: When True, all whitespace is removed after normalization.
            Default is False.

    Returns:
        Normalized text string.

    Examples:
        >>> normalize_text('炒饭用生抽，嗯用生抽。')
        '炒 饭 用 生 抽 嗯 用 生 抽'
        >>> normalize_text('价格是12块5毛')
        '价 格 是 十 二 块 五 毛'
        >>> normalize_text('Hello, 世界！')
        'Hello 世 界'
    """
    # Step 1: Unicode NFC normalization.
    # Combines characters decomposed by NFD (e.g., accented characters,
    # certain CJK compatibility forms) into their canonical composed form.
    text = unicodedata.normalize('NFC', text)

    # Step 2: Strip invisible control characters.
    # These include zero-width spaces, joiners, directional marks,
    # and the byte order mark (BOM). They are not audible and should
    # not affect CER/WER computation.
    text = _INVISIBLE_CHARS_PATTERN.sub('', text)

    # Step 3: Convert Arabic digits to Chinese characters (optional).
    # Performed before punctuation removal so that the decimal point '.'
    # can be properly converted to '点' rather than being stripped.
    if convert_digit:
        text = convert_digits(text)

    # Step 4: Remove punctuation and special symbols.
    # Covers Chinese fullwidth punctuation, English halfwidth punctuation,
    # and decorative symbols commonly found in synthetic QA data.
    text = _PUNCTUATION_PATTERN.sub('', text)

    # Step 5: Optionally insert spaces between Chinese characters.
    # This enables character-level alignment for CER computation.
    if space_chinese:
        text = space_chinese_text(text)

    # Step 6: Collapse consecutive whitespace and trim.
    text = re.sub(r'\s+', ' ', text).strip()

    # Step 7: Optionally remove all whitespace.
    if remove_spaces:
        text = text.replace(' ', '')

    return text


# ---------------------------------------------------------------------------
# JSONL, TextGrid, and STM utilities
# ---------------------------------------------------------------------------


def normalize_jsonl_record(
    record: Dict[str, Any],
    text_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Recursively normalize text fields in a JSONL record.

    This function traverses the record dictionary and applies :func:`normalize_text`
    to all string values whose keys match ``text_fields``. Nested dictionaries
    and lists (e.g., options arrays in QA records) are processed recursively.

    Args:
        record: A dictionary parsed from a JSONL line. Typically represents
            a single data record with fields like ``question_stem``, ``text``, etc.
        text_fields: List of field keys whose string values should be normalized.
            Defaults to ``["question_stem", "text"]`` if not provided.

    Returns:
        A new dictionary with normalized text fields. The original dictionary
        is not mutated.

    Example:
        >>> record = {'id': 'Q001', 'text': '你好，世界！'}
        >>> normalize_jsonl_record(record)
        {'id': 'Q001', 'text': '你好，世界！'}
    """
    if text_fields is None:
        text_fields = ['question_stem', 'text']

    result: Dict[str, Any] = {}
    for key, value in record.items():
        if key in text_fields and isinstance(value, str):
            result[key] = normalize_text(value)
        elif isinstance(value, list):
            result[key] = [
                normalize_jsonl_record(item, text_fields) if isinstance(item, dict)
                else normalize_text(item) if isinstance(item, str)
                else item
                for item in value
            ]
        elif isinstance(value, dict):
            result[key] = normalize_jsonl_record(value, text_fields)
        else:
            result[key] = value
    return result


def extract_textgrid_text(tg_path: str) -> List[str]:
    """Extract all non-empty annotation texts from a Praat TextGrid file.

    Parses the standard Praat TextGrid format and returns all text segments
    that have non-whitespace content. Empty intervals are filtered out.

    Args:
        tg_path: Absolute or relative path to the TextGrid file.

    Returns:
        A list of non-empty text strings, preserving the order in which
        they appear in the TextGrid file.

    Raises:
        FileNotFoundError: If the TextGrid file does not exist.
        UnicodeDecodeError: If the file is not valid UTF-8 encoded text.
    """
    texts: List[str] = []
    with open(tg_path, 'r', encoding='utf-8') as f:
        content = f.read()

    for match in re.finditer(r'text = "([^"]*)"', content):
        text = match.group(1).strip()
        if text:
            texts.append(text)

    return texts


def normalize_textgrid(
    tg_path: str,
    output_path: Optional[str] = None,
) -> List[str]:
    """Read a TextGrid file and normalize all non-empty annotation texts.

    Combines :func:`extract_textgrid_text` and :func:`normalize_text` into
    a single convenience function for batch processing. Each non-empty text
    segment is normalized with ``remove_spaces=True`` (appropriate for CER).

    Args:
        tg_path: Path to the source TextGrid file.
        output_path: Optional output file path. When provided, each normalized
            text segment is written on a separate line. If None (default),
            no file is written.

    Returns:
        A list of normalized text strings.

    Raises:
        FileNotFoundError: If the TextGrid file does not exist.

    Example:
        >>> texts = normalize_textgrid('data/textgrid/chat_0000.TextGrid')
        >>> len(texts)
        83
    """
    texts = extract_textgrid_text(tg_path)
    normalized = [normalize_text(t, remove_spaces=False) for t in texts]

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in normalized:
                f.write(line + '\n')

    return normalized


def normalize_stm(
    stm_path: str,
    output_path: Optional[str] = None,
    convert_digit: bool = True,
    space_chinese: bool = True,
    remove_spaces: bool = False,
) -> List[str]:
    """Read an STM file and normalize the hypothesis (transcript) field.

    The STM (Segments Time Mark) format is widely used in ASR evaluation.
    Each line follows the structure::

        <sessionid> 1 <speakerid> <begintime> <endtime> <hypothesis>

    The first five fields are space-delimited; the hypothesis text starts at
    the sixth field and may contain spaces. This function normalizes only the
    hypothesis field and preserves the original segment metadata.

    Args:
        stm_path: Path to the STM file.
        output_path: Optional output path. When provided, the normalized STM
            (with original metadata preserved) is written to this file.
            If None (default), no file is written.
        convert_digit: Passed through to :func:`normalize_text`. When True,
            Arabic digits in the hypothesis are converted to Chinese characters.
        remove_spaces: Passed through to :func:`normalize_text`. When True,
            all whitespace is removed from the normalized hypothesis.

    Returns:
        A list of normalized hypothesis text strings, in the order they appear
        in the STM file (excluding comment lines starting with ``;`` and
        silence segments with no hypothesis text).

    Raises:
        FileNotFoundError: If the STM file does not exist.

    Example:
        Given an STM line::

            chat_0000 1 spk01 0.37 6.47 炒饭用生抽，嗯用生抽。

        The normalized output (with ``remove_spaces=True``) preserves metadata::

            chat_0000 1 spk01 0.37 6.47 炒饭用生抽嗯用生抽
    """
    normalized_texts: List[str] = []
    output_lines: List[str] = []

    with open(stm_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';'):
                # Preserve blank and comment lines in output if writing file.
                if output_path is not None:
                    output_lines.append(line)
                continue

            parts = line.split(maxsplit=5)
            if len(parts) < 6:
                # Lines with fewer than 6 fields represent silence / empty
                # segments. Preserve the line as-is if writing output.
                if output_path is not None:
                    output_lines.append(line)
                continue

            # Fields: sessionid, channel(1), speakerid, begintime, endtime, hypothesis.
            session_id, channel, speaker_id, begin_time, end_time = parts[:5]
            hypothesis = parts[5]

            normalized = normalize_text(
                hypothesis,
                convert_digit=convert_digit,
                space_chinese=space_chinese,
                remove_spaces=remove_spaces,
            )
            normalized_texts.append(normalized)

            if output_path is not None:
                output_line = f'{session_id} {channel} {speaker_id} {begin_time} {end_time} {normalized}'
                output_lines.append(output_line)

    if output_path and output_lines:
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in output_lines:
                f.write(line + '\n')

    return normalized_texts


# ---------------------------------------------------------------------------
# Analysis utilities
# ---------------------------------------------------------------------------


def _is_chinese_char(ch: str) -> bool:
    """Check whether a character is a CJK Unified Ideograph.

    Args:
        ch: A single character string.

    Returns:
        True if the character falls within the CJK Unified Ideographs range
        (U+4E00-U+9FFF or U+3400-U+4DBF).
    """
    code = ord(ch)
    return any(start <= code <= end for start, end in _CJK_RANGES)


def analyze_text(text: str) -> Dict[str, int]:
    """Analyze the character-type distribution of a text string.

    Useful for diagnosing what types of characters remain after normalization,
    or for understanding the composition of a dataset before preprocessing.

    Args:
        text: Input text string to analyze.

    Returns:
        A dictionary with the following keys:

        - ``total_chars``: Total number of characters.
        - ``chinese_chars``: Count of CJK Unified Ideographs.
        - ``digits``: Count of Arabic digit characters (0-9).
        - ``english_letters``: Count of ASCII English letters (a-z, A-Z).
        - ``punctuation``: Count of punctuation characters (as defined by
          this module's punctuation set).
        - ``spaces``: Count of ASCII space (U+0020) and ideographic space
          (U+3000).
        - ``other``: Count of all other characters.

    Example:
        >>> stats = analyze_text('你好，世界！Hello123')
        >>> stats['chinese_chars']
        5
    """
    stats: Dict[str, int] = {
        'total_chars': len(text),
        'chinese_chars': 0,
        'digits': 0,
        'english_letters': 0,
        'punctuation': 0,
        'spaces': 0,
        'other': 0,
    }

    for ch in text:
        if _is_chinese_char(ch):
            stats['chinese_chars'] += 1
        elif ch.isdigit():
            stats['digits'] += 1
        elif ch.isascii() and ch.isalpha():
            stats['english_letters'] += 1
        elif ch in _ALL_PUNCTUATION or ch == '.':
            stats['punctuation'] += 1
        elif ch == ' ' or ch == '\u3000':
            stats['spaces'] += 1
        else:
            stats['other'] += 1

    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _print_usage() -> None:
    """Print CLI usage information and exit."""
    print('Usage:')
    print('  Normalize text:        python text_normalization.py <text>')
    print('  Normalize JSONL:       python text_normalization.py --jsonl <path> '
          '[--fields f1,f2]')
    print('  Normalize TextGrid:    python text_normalization.py --textgrid <path> '
          '[-o output.txt]')
    print('  Normalize STM:         python text_normalization.py --stm <path> '
          '[-o output.stm] [-d]')
    print('  Analyze text:          python text_normalization.py --analyze <text>')
    print()
    print('Examples:')
    print('  python text_normalization.py "炒饭用生抽，嗯用生抽。"')
    print('  python text_normalization.py --jsonl data.jsonl --fields question_stem,text')
    print('  python text_normalization.py --textgrid data.TextGrid -o normalized.txt')
    print('  python text_normalization.py --stm hyp.stm -o hyp_norm.stm')
    print('  python text_normalization.py --stm hyp.stm -o hyp_norm.stm -d')
    print('    (-d disables digit conversion)')
    print('  python text_normalization.py --analyze "Hello, 世界！"')


def main() -> None:
    """CLI entry point for text normalization.

    Parses command-line arguments and dispatches to the appropriate
    normalization or analysis function.
    """
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        _print_usage()
        sys.exit(1 if len(sys.argv) < 2 else 0)

    command = sys.argv[1]

    if command == '--jsonl':
        _handle_jsonl_command()
    elif command == '--textgrid':
        _handle_textgrid_command()
    elif command == '--stm':
        _handle_stm_command()
    elif command == '--analyze':
        _handle_analyze_command()
    else:
        # Default: treat arguments as raw text to normalize.
        text = ' '.join(sys.argv[1:])
        print(normalize_text(text, remove_spaces=False))


def _handle_jsonl_command() -> None:
    """Handle the ``--jsonl`` CLI command: normalize fields in a JSONL file."""
    path = sys.argv[2]
    fields = ['question_stem', 'text']

    for i, arg in enumerate(sys.argv):
        if arg == '--fields' and i + 1 < len(sys.argv):
            fields = [f.strip() for f in sys.argv[i + 1].split(',')]

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            normalized = normalize_jsonl_record(record, text_fields=fields)
            print(json.dumps(normalized, ensure_ascii=False))


def _handle_textgrid_command() -> None:
    """Handle the ``--textgrid`` CLI command: extract and normalize TextGrid."""
    tg_path = sys.argv[2]
    output_path: Optional[str] = None

    if '-o' in sys.argv:
        idx = sys.argv.index('-o')
        if idx + 1 < len(sys.argv):
            output_path = sys.argv[idx + 1]

    texts = normalize_textgrid(tg_path, output_path)
    print(f'Extracted and normalized {len(texts)} text segments.')
    if output_path:
        print(f'Saved to: {output_path}')


def _handle_stm_command() -> None:
    """Handle the ``--stm`` CLI command: normalize hypothesis text in an STM file."""
    stm_path = sys.argv[2]
    output_path: Optional[str] = None
    convert_digit = True

    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == '-o' and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '-d':
            convert_digit = False
            i += 1
        else:
            i += 1

    texts = normalize_stm(
        stm_path,
        output_path=output_path,
        convert_digit=convert_digit,
        space_chinese=True,
        remove_spaces=False,
    )
    print(f'Normalized {len(texts)} hypothesis segments.')
    if output_path:
        print(f'Saved to: {output_path}')


def _handle_analyze_command() -> None:
    """Handle the ``--analyze`` CLI command: print character distribution."""
    text = ' '.join(sys.argv[2:])
    stats = analyze_text(text)

    print(f'Total characters: {stats["total_chars"]}')
    print(f'  Chinese characters: {stats["chinese_chars"]}')
    print(f'  Digits:             {stats["digits"]}')
    print(f'  English letters:    {stats["english_letters"]}')
    print(f'  Punctuation:        {stats["punctuation"]}')
    print(f'  Spaces:             {stats["spaces"]}')
    print(f'  Other:              {stats["other"]}')


if __name__ == '__main__':
    main()
