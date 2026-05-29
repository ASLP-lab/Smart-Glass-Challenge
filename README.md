# 🏆 SLT 2026 SmartGlasses Challenge: Official Evaluation Toolkit

This repository hosts the official evaluation toolkit of the **SLT 2026 SmartGlasses Challenge — Egocentric Speech Interaction on AI Glasses**. It is intended for participants to score their system outputs locally on the development set in a way that is consistent with the final scoring on the hidden test set. For the call for participation, data description, registration and the full timeline, please refer to the challenge homepage:

- 🌐 **Challenge homepage**: <https://aslp-lab.github.io/SmartGlasses/>

---

## 📖 1. Challenge Overview

Driven by the rapid progress of Large Language Models (LLMs) and Multimodal LLMs, AI-powered smart glasses are becoming a next-generation platform for human–computer interaction. Equipped with microphone arrays and cameras, smart glasses naturally capture the wearer's egocentric (first-person) perspective and enable hands-free multimodal communication in daily life. Compared with stationary devices such as smart speakers or handheld smartphones, smart glasses operate in highly dynamic acoustic environments with environmental noise, user-generated motion noise, and speech from surrounding people, which makes robust speech-centric interaction substantially harder.

The SmartGlasses Challenge introduces a new benchmark for evaluating **Time-Stamped Speaker-Attributed ASR (TSA-ASR)** and **Spoken Language Understanding (SLU)** in real-world egocentric scenarios. The challenge consists of two tracks:

- **Track 1 — Dyadic Dialogue Understanding**: face-to-face two-person conversations in everyday settings, with overlapping speech, background interference and topic shifts.
- **Track 2 — Multi-Party Meeting Understanding**: multi-speaker meetings with varying numbers of participants, frequent turn-taking, long contexts and domain-specific vocabulary.

Each track contains two tasks:

- **Task 1 — TSA-ASR**: speaker-attributed transcription with time alignment in overlapping speech.
- **Task 2 — SLU**: multiple-choice question answering over the conversation/meeting audio.

## 🎯 2. Scope of This Toolkit

**This toolkit currently covers the TSA-ASR task (Task 1) of both Track 1 and Track 2.** The official ASR evaluation pipeline is built on top of [MeetEval](https://github.com/fgnt/meeteval), so all metrics, time-tolerance collars and input formats here are aligned with the final test-set scoring.

**Primary ranking metric** for Task 1 is **tcpWER** (time-constrained minimum-permutation Word Error Rate) with a time tolerance collar of **5 s**. Two diagnostic metrics are additionally reported to help teams locate error sources:

- **cpWER**: minimum-permutation WER without time constraint.
- **DER**: diarization error rate, computed via MeetEval's `dscore` wrapper around `md-eval-22.pl`, with a 0.25 s collar.

For the SLU task (Task 2), please see Section 7 below.

## ⚙️ 3. Installation

The toolkit only depends on `meeteval`. We recommend a clean Python ≥ 3.8 environment.

Install from PyPI:

```bash
pip install meeteval
```

Or install from source (useful if you want the latest changes):

```bash
git clone https://github.com/fgnt/meeteval
pip install -e ./meeteval
```

After installation the following command-line entries should be available:

```bash
meeteval-wer --help
meeteval-der --help
```

Then clone this repository:

```bash
git clone https://github.com/DontPushMeee/Smart-Glass-Challenge.git
cd Smart-Glass-Challenge
```

## 📝 4. Input Format

Both the reference and the hypothesis files **must be in STM (Segmental Time Mark) format**. Each line in an STM file describes one utterance:

```text
<session_id> <channel> <speaker_id> <begin_time> <end_time> <transcript>
```

The fields are space-separated:

- `session_id`: recording / session identifier (must match between ref and hyp).
- `channel`: channel ID (conventionally set to `1`; ignored by MeetEval during scoring).
- `speaker_id`: speaker label or system output stream. Speaker labels do not need to be aligned between ref and hyp — cpWER / tcpWER will find the best permutation automatically.
- `begin_time`, `end_time`: utterance start / end in seconds (float).
- `transcript`: the words of this utterance, separated by spaces.

### Chinese Tokenization

The TSA-ASR task in this challenge is evaluated at the **character level for Chinese**. In every STM line, **Chinese sentences must be tokenized into individual characters separated by single spaces**. Punctuation should be removed. For example, a sentence such as `你好请问你是新来的同事吗` must be written as:

```text
你 好 请 问 你 是 新 来 的 同 事 吗
```

Submissions whose Chinese segments are not character-tokenized will be scored as-is (i.e., the whole sentence will be treated as one token), which is almost always penalised by WER.

### Example Files

Two minimal example files are provided under [`example/`](example):

- [`example/ref.stm`](example/ref.stm): reference transcript.
- [`example/hyp.stm`](example/hyp.stm): a toy hypothesis.

They follow exactly the format expected for an official submission (see Section 6).

## 🚀 5. How to Run

The driver script is [`run.sh`](run.sh). It runs the three official metrics in sequence on the example pair:

```bash
bash run.sh
```

The three commands invoked are:

```bash
# Diarization Error Rate (DER), 0.25 s collar
meeteval-der dscore  -r example/ref.stm -h example/hyp.stm --collar .25

# Concatenated minimum-Permutation WER (cpWER)
meeteval-wer cpwer   -r example/ref.stm -h example/hyp.stm

# Time-Constrained cpWER (tcpWER), 5 s collar
meeteval-wer tcpwer  -r example/ref.stm -h example/hyp.stm --collar 5
```

### Argument Notes

- `-r` / `-h`: path of the reference / hypothesis STM file.
- `--collar`: time tolerance, in seconds. Different metrics use different collars by convention. **For the official ranking metric `tcpWER`, the collar is fixed to `5` seconds.**
- `--collar .25` for DER follows the dscore convention.

To score on your own data, simply replace `example/ref.stm` and `example/hyp.stm` with your reference and your system output. Each invocation will write a JSON file next to the hypothesis (e.g. `hyp_cpwer.json`, `hyp_tcpwer.json`, `hyp_dscore.json`) that contains the detailed statistics.

## ⚠️ 6. Submission and Failure Rules

For Task 1 (TSA-ASR), every audio session in the evaluation set **must have a result that can be parsed as a valid `hyp.stm` file**. Concretely:

- The submission must contain one STM line per recognised utterance, following the format described in Section 4.
- All session IDs that appear in the official reference must also appear in the submission. Sessions for which the system fails to produce any output will be treated as empty hypotheses (i.e. counted as deletions).
- Lines that cannot be parsed (wrong number of fields, non-numeric timestamps, missing speaker id, etc.) are considered invalid. **Submissions that cannot be parsed by this toolkit will not be scored as a valid result.**

The exact submission packaging (file naming, directory layout, manifest) will be released by the organisers ahead of the leaderboard opening, and the official scoring pipeline will be the one in this repository. Teams are strongly encouraged to verify their hypotheses with `bash run.sh` against their own files before submission.

## 🧠 7. SLU Task Evaluation (Task 2)

For the time being, participating teams can evaluate their SLU models locally by simply calculating the **Accuracy** (i.e., the percentage of correctly answered multiple-choice questions) against the provided Dev set references. Since the metric is straightforward, a unified evaluation script is not provided at this stage. Baseline accuracy scores on the Dev set are provided in Section 8.

We will release the standardized submission format requirements and integrate the official SLU scoring pipeline into this repository well before the test phase and leaderboard submission open. Please stay tuned.

## 📊 8. Reference Results

The numbers below are reference baseline scores evaluated on the **official development (dev) set**.

### Task 1 — TSA-ASR

The following scores are obtained by running the metrics in this toolkit on the output of [**VibeVoice-ASR**](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-asr.md).

#### Track 1 — Dyadic Dialogue Understanding

| Metric                  | Value     |
|-------------------------|-----------|
| DER (collar = 0.25 s)   | 9.90 %    |
| cpWER                   | 15.50 %   |
| tcpWER (collar = 5 s)   | 15.95 %   |

#### Track 2 — Multi-Party Meeting Understanding

| Metric                  | Value     |
|-------------------------|-----------|
| DER (collar = 0.25 s)   | 17.39 %   |
| cpWER                   | 30.72 %   |
| tcpWER (collar = 5 s)   | 31.56 %   |

### Task 2 — SLU

The following table reports the **Accuracy (%)** of baseline Large Audio-Language Models on the multiple-choice QA task.

| Model | Track 1 (Dyadic) | Track 2 (Meeting) |
|---|:---:|:---:|
| Qwen2.5-Omni | 63.13 | 54.37 |
| Qwen3-Omni | 74.49 | 75.08 |

Additional reference systems and the corresponding scores will be published in this repository as the challenge progresses.

## 🙏 9. Acknowledgement

This toolkit is built on top of [MeetEval](https://github.com/fgnt/meeteval) by Paderborn University. We thank the authors for releasing and maintaining this excellent meeting-transcription evaluation library.

We also acknowledge the use of [VibeVoice](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-asr.md) as one of our baseline systems for the TSA-ASR task.

## 📧 10. Contact

For questions or issues regarding the evaluation toolkit, please open an issue in this repository or contact the challenge organisers:

- gdh@mail.nwpu.edu.cn
- zxzhao@mail.nwpu.edu.cn
- liaoyujie@mail.nwpu.edu.cn
