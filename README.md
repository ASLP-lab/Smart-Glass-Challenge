# SLT 2026 SmartGlasses Challenge — Text Normalization Toolkit

[![SLT 2026](https://img.shields.io/badge/SLT-2026-blue)](https://aslp-lab.github.io/SmartGlasses/)

This module provides the official text normalization utilities for the
**SLT 2026 SmartGlasses Challenge: Egocentric Speech Interaction on AI Glasses**.
It is designed for preprocessing ASR hypothesis and reference texts before
computing CER (Character Error Rate) and WER (Word Error Rate) in Tasks 1
(TSA-ASR) and Task 2 (SLU).

> **Challenge homepage**: [https://aslp-lab.github.io/SmartGlasses/](https://aslp-lab.github.io/SmartGlasses/)

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Input Format (STM)](#input-format-stm)
- [Normalization Pipeline](#normalization-pipeline)
- [Usage](#usage)
  - [Command Line](#command-line)
  - [Python API](#python-api)
- [Examples](#examples)
- [API Reference](#api-reference)
- [License](#license)

---

## Overview

The SmartGlasses Challenge evaluates two tracks:

| Track | Scenario | Description |
|---|---|---|
| **Track 1** | Dyadic Dialogue | Face-to-face two-person conversations with overlapping speech, background interference, and topic shifts. |
| **Track 2** | Multi-Party Meeting | Multi-speaker meetings with varying participants, frequent turn-taking, long contexts, and domain-specific vocabulary. |

Each track has two tasks:

| Task | Name | Description |
|---|---|---|
| **Task 1** | TSA-ASR | Time-Stamped Speaker-Attributed ASR — transcribe speech with speaker labels and timestamps. |
| **Task 2** | SLU | Spoken Language Understanding — answer multiple-choice questions over the audio. |

This toolkit handles **text normalization** for Task 1 (TSA-ASR) evaluation. It
normalizes hypothesis (system output)
transcripts before  computation, ensuring a fair and consistent scoring.

---

## Installation

The module has **no external dependencies** beyond Python 3.8+.

```bash
# Clone or copy text_normalization.py into your project
# Then simply import:
from text_normalization import normalize_text
```

---

## Input Format (STM)

System outputs for Task 1 must be submitted in **STM (Segments Time Mark)**
format. Each line represents one speech segment:

```
<session_id> 1 <speaker_id> <begin_time> <end_time> <hypothesis_text>
```

| Field | Type | Description |
|---|---|---|
| `session_id` | string | Audio session identifier (e.g., `chat_0000`) |
| `1` | int | Channel (always 1 for this challenge) |
| `speaker_id` | string | Speaker label (e.g., `spk01`, `spk02`) |
| `begin_time` | float | Segment start time in seconds |
| `end_time` | float | Segment end time in seconds |
| `hypothesis_text` | string | ASR transcript for this segment |

**Example:**

```
chat_0000 1 spk01 0.37 6.47 炒饭用生抽嗯用生抽嗯清淡点这蛋味儿能出来
chat_0000 1 spk02 6.47 7.16
chat_0000 1 spk01 7.16 7.99 你干嘛你别碰老抽啊
English_Australian_1127_015_phone 1 2 0.7600 2.9900 yeah recording uh
```

- Lines with fewer than 6 fields are treated as silence (empty segments).
- Lines starting with `;` are treated as comments and preserved as-is.

---

## Normalization Pipeline

The toolkit applies the following transformations **in order** to ensure
deterministic and reproducible results:

```
Raw text
    │
    ▼
1. Unicode NFC Normalization
    │  Canonical composition of combining characters.
    ▼
2. Strip Invisible Control Characters
    │  Zero-width spaces, BOM, directional marks, etc.
    ▼
3. Arabic Digit → Chinese (Value-Based)
    │  Integer part:  value-based conversion  (200 → 二百, 12 → 十二)
    │  Decimal part:  digit-by-digit mapping  (.14 → 一四)
    ▼
4. Remove Punctuation
    │  Chinese: ， 。、？！：；""''（）【】《》「」～·…—
    │  English: !"#$%&()*+,-./:;<=>?@[]^_`{|}~
    │  Symbols: ●■□◆◇▲△▼▽★☆
    ▼
5. Space Chinese Characters
    │  Insert a space between every Chinese character for CER alignment.
    │  Non-Chinese tokens (English words, digits) remain contiguous.
    ▼
6. Collapse & Trim Whitespace
    │
    ▼
Normalized text
```

### Digit Conversion Detail

The integer part of a number uses **value-based conversion** (matching the
convention in Chinese reference transcripts), while the fractional part uses
**digit-by-digit mapping**:

| Input | Output |
|---|---|
| `12` | `十二` |
| `200` | `二百` |
| `100` | `一百` |
| `101` | `一百零一` |
| `98.5` | `九十八点五` |
| `3.14` | `三点一四` |

### Chinese Spacing Detail

After normalization, Chinese characters are space-separated to enable
character-level alignment for CER:

| Input | Output |
|---|---|
| `炒饭用生抽` | `炒 饭 用 生 抽` |
| `一共二百元` | `一 共 二 百 元` |
| `yeah recording uh` | `yeah recording uh` (unchanged) |

---

## Usage

### Command Line

```bash
# Normalize an STM hypothesis file
python text_normalization.py --stm hyp.stm -o hyp_norm.stm

# Normalize with digit conversion disabled
python text_normalization.py --stm hyp.stm -o hyp_norm.stm -d

# Normalize a single text string
python text_normalization.py "炒饭用生抽，嗯用生抽。"

# Analyze character distribution
python text_normalization.py --analyze "你好，世界！Hello123"
```

### Python API

```python
from text_normalization import normalize_text, normalize_stm

# Normalize a single string
text = normalize_text("炒饭用生抽，嗯用生抽。")
# => "炒 饭 用 生 抽 嗯 用 生 抽"

# Normalize with digit conversion disabled
text = normalize_text("价格是12块5毛", convert_digit=False)
# => "价 格 是 12 块 5 毛"

# Normalize an STM file
texts = normalize_stm("hyp.stm", output_path="hyp_norm.stm")
# => Returns list of normalized hypothesis texts,
#    writes full STM with normalized transcripts to hyp_norm.stm
```

---

## Examples

### STM Normalization

**Input:**

```
English_Australian_1127_015_phone 1 2 0.7600 2.9900 yeah recording uh
chat_0000 1 spk01 0.37 6.47 炒饭用生抽，嗯用生抽。
chat_0000 1 spk01 7.16 7.99 你干嘛？你别碰老抽啊！
chat_0001 1 spk02 1.23 4.56 价格是12块5毛，一共200元。
```

**Output:**

```
English_Australian_1127_015_phone 1 2 0.7600 2.9900 yeah recording uh
chat_0000 1 spk01 0.37 6.47 炒 饭 用 生 抽 嗯 用 生 抽
chat_0000 1 spk01 7.16 7.99 你 干 嘛 你 别 碰 老 抽 啊
chat_0001 1 spk02 1.23 4.56 价 格 是 十 二 块 五 毛 一 共 二 百 元
```

### Per-Utterance Comparison

| Input | Normalized |
|---|---|
| `炒饭用生抽，嗯用生抽。` | `炒 饭 用 生 抽 嗯 用 生 抽` |
| `你干嘛？你别碰老抽啊！` | `你 干 嘛 你 别 碰 老 抽 啊` |
| `价格是12块5毛，一共200元。` | `价 格 是 十 二 块 五 毛 一 共 二 百 元` |
| `第1名和第2名` | `第 一 名 和 第 二 名` |
| `yeah recording uh` | `yeah recording uh` |

---

## API Reference

### `normalize_text(text, convert_digit=True, space_chinese=True, remove_spaces=False)`

Normalize a single text string.

- **Args:**
  - `text` (str): Input text.
  - `convert_digit` (bool): Convert Arabic digits to Chinese (default: True).
  - `space_chinese` (bool): Insert spaces between Chinese characters (default: True).
  - `remove_spaces` (bool): Remove all whitespace (default: False).
- **Returns:** Normalized text string.

### `normalize_stm(stm_path, output_path=None, convert_digit=True, space_chinese=True, remove_spaces=False)`

Read an STM file and normalize the hypothesis field.

- **Args:**
  - `stm_path` (str): Path to the STM file.
  - `output_path` (str, optional): Write normalized STM to this path.
  - `convert_digit` (bool): Convert Arabic digits (default: True).
  - `space_chinese` (bool): Insert spaces between Chinese characters (default: True).
  - `remove_spaces` (bool): Remove all whitespace (default: False).
- **Returns:** List of normalized hypothesis text strings.

### `convert_digits(text)`

Convert Arabic digits to Chinese using value-based mapping.

- **Args:** `text` (str): Input with digits.
- **Returns:** Converted text.

### `space_chinese_text(text)`

Insert a space between every Chinese character.

- **Args:** `text` (str): Input text.
- **Returns:** Text with Chinese characters space-separated.

### `analyze_text(text)`

Analyze character-type distribution of a text string.

- **Args:** `text` (str): Input text.
- **Returns:** Dict with counts of chinese_chars, digits, english_letters, punctuation, spaces, other.

---

## License

Apache License 2.0
