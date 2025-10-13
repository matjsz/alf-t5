---
title: 'ALF-T5: Adaptive Neural Machine Translation Framework for Constructed and Natural Languages'
tags:
  - Python
  - machine learning
  - natural language processing
  - neural machine translation
  - T5
authors:
  - name: Matheus J.G. Silva
    orcid: 0009-0003-5112-7079
    corresponding: true
    affiliation: 1
affiliations:
  - name: Independent Researcher, Brazil
    index: 1
date: 9 October 2025
bibliography: paper.bib
---

# Summary

Neural machine translation (NMT) typically demands abundant parallel corpora, restricting its reach to widely spoken languages and limiting exploration of low-resource and constructed languages (conlangs). ALF-T5 is a Python framework that enables **few-shot**, **bidirectional** translation between English and target languages — natural or constructed — by leveraging transfer learning, parameter-efficient fine-tuning (LoRA via PEFT), and data augmentation. It supports interactive usage, batch translation, automatic evaluation (BLEU, METEOR), and architectural experimentation, making it a useful tool for research, teaching, and prototyping.

# Statement of need

Modern NMT frameworks are biased toward high-resource languages because they demand large parallel corpora and heavy compute. Consequently, research on novel languages, conlangs, or translation under extremely low-data conditions remains underexplored.

ALF-T5 addresses this gap by enabling translation model training from as few as 10–50 sample pairs. Its design emphasizes usability (CLI + Python API), reproducibility (built-in metrics, tests), and modularity (plug in alternative components). Because it integrates with the well-adopted Transformers and PEFT ecosystem, it remains accessible on modest hardware. For educators, it offers an interpretable translation tool on small datasets; for researchers, it provides a testbed for low-resource translation techniques and conlangs.

# Technical approach

## Model & Loss formulation

Let

\[
X = (x_1, x_2, \dots, x_n), \quad
Y = (y_1, y_2, \dots, y_m)
\]

be the source and target sequences, respectively. The model defines:

\[
P(Y \mid X) = \prod_{t=1}^{m} P\bigl(y_t \mid y_{<t}, X \bigr)
\]

We train using cross-entropy loss:

\[
\mathcal{L}_{\mathrm{CE}} = -\sum_{t=1}^{m} \log P\bigl(y_t \mid y_{<t}, X \bigr)
\]

To enable bidirectional translation, we combine both directions:

\[
\mathcal{L}_{\mathrm{bi}} = \mathcal{L}_{\mathrm{CE}}(X \to Y) + \mathcal{L}_{\mathrm{CE}}(Y \to X)
\]

## Parameter-efficient fine-tuning via LoRA / PEFT

ALF-T5 uses **LoRA** (low-rank adaptation) through **PEFT**, which keeps the original pretrained weights **frozen**, and only trains a small set of low-rank adapter matrices. This yields efficient adaptation with lower memory and compute costs compared to full fine-tuning [@hu2021lora].

## Data augmentation

Given the constrained number of training pairs, we generate synthetic variants through:

- Case variation (e.g., uppercase, lowercase, title)  
- Word-order permutations within short phrases  
- Vocabulary recombination, mixing subword parts or synonymous units  

These augmentations help regularize the model and improve performance in data-scarce regimes.

# Usage & Features

- **Interactive mode:** CLI prompt with optional confidence  
- **Batch mode:** Translate lists or files in parallel  
- **Automatic metrics:** BLEU and METEOR scoring  
- **Experiment support:** Grid search over hyperparameters (LoRA rank, learning rate, etc.), logging  
- **Model inference:** Load and translate new text using the same API

# Validation & Examples

In a constructed language experiment with **45 training pairs**, ALF-T5 generated bidirectional translation outputs (see Figure \ref{fig:showcase}). On Portuguese ↔ English (Pirá dataset), ALF-T5 achieves BLEU and METEOR scores comparable to small baselines (see Figure \ref{fig:performance}).

*(Insert exact numeric results here.)*

# Citations

Key works and software used by ALF-T5:

- Raffel et al. (2020) — T5 model  
- Vaswani et al. (2017) — Transformer architecture  
- Hu et al. — LoRA: low-rank adaptation  
- Papineni et al. (2002) — BLEU metric  
- Banerjee & Lavie (2005) — METEOR metric  
- Wolf et al. — Hugging Face Transformers library

# Figures

![Few-shot translation in a constructed language (45 pairs).](https://raw.githubusercontent.com/matjsz/alf-t5/refs/heads/main/docs/showcase.png){#fig:showcase}

![Portuguese ↔ English translation performance (BLEU/METEOR).](https://raw.githubusercontent.com/matjsz/alf-t5/refs/heads/main/docs/showcase-2.png){#fig:performance}

# Acknowledgements

We thank the authors of T5 and LoRA, and the teams behind Transformers and PEFT. Special thanks to Paulo Pirozelli for the Pirá dataset, and to early users who tested edge cases.

