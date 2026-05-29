#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# SLT 2026 SmartGlasses Challenge — TSA-ASR scoring driver
#
# This script runs the three official metrics for Task 1 (TSA-ASR) on the
# toy example shipped in `example/`. To score your own system, replace
# `example/ref.stm` with the official reference and `example/hyp.stm` with
# your hypothesis. Both files must follow the STM format described in the
# README (Section 4); for Chinese, transcripts must be character-tokenised.
#
# Outputs: each command writes a JSON file next to the hypothesis with the
# detailed error statistics (e.g. hyp_cpwer.json, hyp_tcpwer.json,
# hyp_dscore.json).
# ----------------------------------------------------------------------------

# Diarization Error Rate (DER) via MeetEval's dscore wrapper around
# md-eval-22.pl. The 0.25 s collar matches the official setting.
meeteval-der dscore -r example/ref.stm -h example/hyp.stm --collar .25

# Concatenated minimum-Permutation WER (cpWER): permutation-invariant WER
# with no time constraint. Reported as a diagnostic metric.
meeteval-wer cpwer -r example/ref.stm -h example/hyp.stm

# Time-Constrained cpWER (tcpWER): the primary ranking metric of Task 1.
# The collar is fixed to 5 seconds for both Dev and Test scoring.
meeteval-wer tcpwer -r example/ref.stm -h example/hyp.stm --collar 5
