## 🎙️ VibeVoice-ASR Fine-tuning Recipe for the SmartGlasses Challenge

This document describes the **reference fine-tuning recipe** used to produce the *VibeVoice-ASR (fine-tuned)* baseline reported in the [main README](../../README.md). It is intended as a reproducible starting point for participants who want to adapt [VibeVoice-ASR](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-asr.md) to the challenge data using the official LoRA fine-tuning pipeline released by Microsoft: <https://github.com/microsoft/VibeVoice/tree/main/finetuning-asr>.

The training launcher used in this recipe is provided in [`finetuning-asr/train.sh`](finetuning-asr/train.sh).

---

### 📖 1. Overview

[VibeVoice-ASR](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-asr.md) is a multi-talker, time-stamped speaker-attributed ASR model released by Microsoft. The model and its official LoRA fine-tuning code (`lora_finetune.py`) live in the `finetuning-asr/` directory of the [VibeVoice](https://github.com/microsoft/VibeVoice) repository.

For the SLT 2026 SmartGlasses Challenge we adapt VibeVoice-ASR to each track **independently**:

- **Track 1 model**: fine-tuned on the Track 1 (Dyadic Dialogue) training set only.
- **Track 2 model**: fine-tuned on the Track 2 (Multi-Party Meeting) training set only.

The same hyper-parameters (see Section 4) are reused for both tracks; only the training data and the output directory change. The resulting checkpoints reproduce the numbers reported in Section 8 of the [main README](../../README.md).

### ⚙️ 2. Installation

We recommend a clean Python ≥ 3.10 environment with a recent CUDA-capable PyTorch (≥ 2.1 + CUDA 12.x).

#### 2.1 Clone VibeVoice

```bash
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice
```

#### 2.2 Install VibeVoice and its fine-tuning dependencies

Follow the upstream instructions in <https://github.com/microsoft/VibeVoice/tree/main/finetuning-asr>. In a typical setup:

```bash
# Core VibeVoice package
pip install -e .

# Extra dependencies required by the fine-tuning recipe
pip install -r finetuning-asr/requirements.txt
```

This installs `transformers`, `peft`, `accelerate`, `deepspeed`, `librosa`, `soundfile`, etc. Please always refer to the upstream README for the most up-to-date dependency list.

#### 2.3 Download the VibeVoice-ASR checkpoint

The base checkpoint is hosted on Hugging Face under [`microsoft/VibeVoice-ASR`](https://huggingface.co/microsoft/VibeVoice-ASR). It will be downloaded automatically the first time `--model_path microsoft/VibeVoice-ASR` is used, or you can pre-fetch it:

```bash
huggingface-cli download microsoft/VibeVoice-ASR --local-dir ./pretrained/VibeVoice-ASR
```

If you pre-download, change `--model_path` in [`finetuning-asr/train.sh`](finetuning-asr/train.sh) to the local path.

#### 2.4 Drop in the training launcher

Copy the launcher provided here into the upstream `finetuning-asr/` folder so it sits next to `lora_finetune.py`:

```bash
cp /path/to/Smart-Glass-Challenge/example/VibeVoice/finetuning-asr/train.sh \
   VibeVoice/finetuning-asr/train.sh
cd VibeVoice/finetuning-asr
```

### 🗂️ 3. Data Preparation

The official VibeVoice-ASR fine-tuning pipeline expects the training data to live under a single root directory (passed via `--data_dir`) and to follow the manifest layout documented in <https://github.com/microsoft/VibeVoice/tree/main/finetuning-asr>. Please refer to the upstream README for the exact JSON / audio layout, naming conventions and expected fields — the SmartGlasses Challenge does **not** redefine this format.

For this recipe we organise the challenge data as follows:

```text
finetuning-asr/
├── lora_finetune.py
├── train.sh
└── Train/                # passed to --data_dir
    ├── track1/           # Track 1 (dyadic) training manifests + audio
    └── track2/           # Track 2 (meeting) training manifests + audio
```

When fine-tuning the Track 1 model, point `--data_dir` to `./Train/track1`; when fine-tuning the Track 2 model, point it to `./Train/track2`. Make sure the references inside the manifests are **character-tokenised for Chinese** (single Chinese characters separated by spaces, no punctuation), exactly as required by the evaluation toolkit (see Section 4 of the [main README](../../README.md)).

### 🚀 4. Fine-tuning

The provided launcher [`finetuning-asr/train.sh`](finetuning-asr/train.sh) wraps `lora_finetune.py` with the hyper-parameters used to obtain the *VibeVoice-ASR (fine-tuned)* baseline:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 lora_finetune.py \
    --model_path microsoft/VibeVoice-ASR \
    --data_dir ./Train \
    --output_dir ./output \
    --num_train_epochs 100 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --gradient_checkpointing \
    --bf16 \
    --report_to none
```

Key arguments:

- `--model_path`: base checkpoint to fine-tune from. Use `microsoft/VibeVoice-ASR` (auto-download) or a local path.
- `--data_dir`: training data root for the current track (`./Train/track1` or `./Train/track2`).
- `--output_dir`: where the LoRA adapter checkpoints will be written. Use a track-specific path, e.g. `./output/track1` and `./output/track2`.
- `--num_train_epochs 100`, `--learning_rate 1e-5`, `--warmup_ratio 0.1`, `--weight_decay 0.01`, `--max_grad_norm 1.0`: optimisation schedule.
- LoRA configuration: `--lora_r 8`, `--lora_alpha 32`, `--lora_dropout 0.05`.
- Effective batch size: `per_device_train_batch_size (1) × gradient_accumulation_steps (16) × nproc_per_node (4) = 64`.
- `--bf16` and `--gradient_checkpointing` are recommended for memory efficiency on Ampere / Hopper GPUs.

#### 4.1 Track 1

```bash
# Edit train.sh (or override on the CLI) to use the Track 1 data and output dir
bash train.sh \
    # equivalent to setting --data_dir ./Train/track1 --output_dir ./output/track1
```

#### 4.2 Track 2

```bash
# Re-run the same launcher with the Track 2 paths
bash train.sh \
    # equivalent to setting --data_dir ./Train/track2 --output_dir ./output/track2
```

Each run produces a LoRA adapter under `--output_dir` that can be merged into / loaded on top of the base VibeVoice-ASR weights at inference time.

### 🔎 5. Inference

Inference uses the standard VibeVoice-ASR decoding pipeline; the only difference is that the LoRA adapter trained above must be loaded on top of the base model. Please follow the inference recipe in <https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-asr.md> and <https://github.com/microsoft/VibeVoice/tree/main/finetuning-asr>, pointing the loader at:

- the base model: `microsoft/VibeVoice-ASR` (or your local copy), and
- the LoRA adapter: e.g. `./output/track1` (for Track 1) or `./output/track2` (for Track 2).

The decoder must produce, **for every session in the dev / test set**, one STM line per recognised utterance, in the exact format described in Section 4 of the [main README](../../README.md):

```text
<session_id> <channel> <speaker_id> <begin_time> <end_time> <transcript>
```

with Chinese transcripts character-tokenised and punctuation removed. Concatenate all sessions into a single `hyp.stm`.

### 📏 6. Local Scoring

Once your `hyp.stm` (and the corresponding `ref.stm` released for the dev set) are ready, score them with the toolkit shipped at the repository root:

```bash
cd /path/to/Smart-Glass-Challenge
bash run.sh   # or invoke meeteval-{wer,der} directly on your own files
```

This reproduces DER / cpWER / tcpWER under exactly the same configuration that will be used on the hidden test set. The expected numbers for this recipe on the official dev set are listed in Section 8 of the [main README](../../README.md).

### 🙏 7. Acknowledgement

The fine-tuning code (`lora_finetune.py`) and the underlying ASR model are released by Microsoft as part of the [VibeVoice](https://github.com/microsoft/VibeVoice) project. We thank the authors for open-sourcing both the model and a reproducible LoRA fine-tuning recipe, which this document is built upon.
