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
  - name: Anhanguera University, Brazil
    index: 1
date: 9 October 2025
bibliography: paper.bib
---

# Summary

Neural machine translation (NMT) typically demands abundant parallel corpora, restricting its reach to widely spoken languages and limiting exploration of low-resource and constructed languages (conlangs). **ALF-T5** is a Python framework that enables **few-shot**, **bidirectional** translation between English and target languages — natural or constructed — by leveraging transfer learning, parameter-efficient fine-tuning (LoRA via PEFT), and data augmentation. It supports interactive usage, batch translation, automatic evaluation (BLEU, METEOR), and architectural experimentation, making it useful for research, teaching, and prototype development.

# Statement of need

Most modern NMT toolkits assume large parallel corpora and substantial compute. This restricts experimentation with under-resourced languages and constructed languages, and limits exploration of translation strategies in extremely low-data regimes.

ALF-T5 fills this gap by enabling translation model training from as few as **10–50 example pairs**. The project emphasizes usability (CLI + Python API), reproducibility (built-in metrics and tests), and modularity (pluggable augmentation and experiment components). Because it integrates with the widely used Transformers and PEFT ecosystems, ALF-T5 is usable on modest hardware and useful for both educators and researchers.

# Technical approach

## Model and loss formulation

Let the source sequence be $X = (x_1, x_2, \dots, x_n)$ and the target sequence be $Y = (y_1, y_2, \dots, y_m)$. The model factorizes the conditional probability as

$$
P(Y \mid X) \;=\; \prod_{t=1}^{m} P\bigl(y_t \mid y_{1:t-1}, X\bigr).
$$

We train using the standard cross-entropy loss:

$$
\mathcal{L}_{\mathrm{CE}}(X,Y) \;=\; -\sum_{t=1}^{m} \log P\bigl(y_t \mid y_{1:t-1}, X\bigr).
$$

To support bidirectional translation, we train both directions and combine losses:

$$
\mathcal{L}_{\mathrm{bi}} \;=\; \mathcal{L}_{\mathrm{CE}}(X\to Y) \;+\; \mathcal{L}_{\mathrm{CE}}(Y\to X).
$$

(Using $y_{1:t-1}$ instead of $y_{<t}$ avoids characters that can be converted or escaped by the toolchain.)

## Parameter-efficient tuning (LoRA / PEFT)

ALF-T5 uses LoRA (low-rank adaptation) implemented through the PEFT framework: the pretrained base model weights remain frozen and we learn small low-rank adapter matrices. This drastically reduces the number of trainable parameters and lowers memory / compute requirements compared to full fine-tuning [@hu2021lora]. The adapters are inserted at attention / projection layers and trained while the backbone stays unchanged.

## Data augmentation

To amplify scarce parallel data we use lightweight augmentations:

- Case variation (lowercase / uppercase / title)  
- Word-order permutations within short phrases  
- Vocabulary recombination (mixing subword segments or analogous tokens)

These synthetic variants increase the effective dataset size and help generalization in low-data settings.

# Usage & features

- **Interactive mode**: CLI REPL for single-sentence translation with optional confidence scores.  
- **Batch mode**: File and list translation for large-scale processing.  
- **Automatic evaluation**: BLEU and METEOR computed during or after training.  
- **Experiment framework**: Grid search over hyperparameters (LoRA rank, learning rate) with result logging.  
- **Model inference**: Load a saved translator and translate new text via a simple API.

# Validation & examples

- **Constructed language (few-shot):** ALF-T5 learned a small constructed language from **45** training pairs and produced coherent bidirectional outputs (see Figure \ref{fig:showcase}).  
- **Portuguese-English:** On the Pirá dataset (Portuguese↔English QA/translation subset) ALF-T5 matches or outperforms small baselines on BLEU and METEOR for comparable model sizes (see Figure \ref{fig:performance}).

# Citations

This project builds on a number of foundational works and libraries, including T5 [@raffel2020exploring], the Transformer architecture [@vaswani2017attention], LoRA [@hu2021lora], BLEU [@papineni2002bleu], METEOR [@banerjee2005meteor], and the Hugging Face Transformers library [@wolf2020transformers].

# Figures

![Few-shot translation example in a constructed language (45 training pairs).](https://raw.githubusercontent.com/matjsz/alf-t5/refs/heads/main/docs/showcase.png){#fig:showcase}

![Portuguese-English translation performance (BLEU / METEOR).](https://raw.githubusercontent.com/matjsz/alf-t5/refs/heads/main/docs/showcase-2.png){#fig:performance}

# Acknowledgements

Thanks to the authors of T5 and LoRA, and to the Hugging Face and PEFT teams. Special thanks to Paulo Pirozelli for providing the Pirá dataset used in evaluation, and to early testers who provided feedback.

