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

Neural machine translation (NMT) has become a cornerstone of natural language processing. However, training high-quality translation models often requires vast parallel corpora, which limits applications in low-resource languages or constructed languages (conlangs). 

ALF-T5 is a Python framework that enables few-shot, bidirectional translation of both natural and constructed languages. Built on top of Google's T5 model, it uses transfer learning, parameter-efficient fine-tuning (LoRA), and data augmentation to achieve high-quality translations from minimal examples. The framework also supports interactive translation, batch processing, automatic evaluation metrics (BLEU, METEOR), and model architecture experimentation, making it suitable for research, education, and rapid prototyping.

# Statement of need

Existing NMT frameworks focus primarily on high-resource natural languages and often require extensive training data and computational resources. This limits exploration in underrepresented languages, constructed languages, or experimental research into novel translation architectures.

ALF-T5 addresses this gap by providing a lightweight, flexible, and efficient Python framework that can learn new languages from tens of examples while supporting scalable experimentation. Its CLI, Python API, and automatic evaluation metrics enable both developers and researchers to quickly train, test, and deploy custom translation models. 

The framework leverages Hugging Face's Transformers library and PEFT for low-rank adaptation, making it computationally feasible to fine-tune large models with limited resources. By supporting both small-scale experimental languages and medium-scale natural language translation tasks, ALF-T5 empowers educational, research, and hobbyist projects alike.

# Mathematics

The ALF-T5 translation model is based on the standard encoder-decoder formulation for sequence-to-sequence learning. For an input sequence $(X = (x_1, x_2, \dots, x_n))$ in the source language, the conditional probability of the target sequence $(Y = (y_1, y_2, \dots, y_m))$ is given by:

$$
P(Y|X) = \prod_{t=1}^{m} P(y_t | y_{<t}, X)
$$

The training objective minimizes the cross-entropy loss:

$$
\mathcal{L} = - \sum_{t=1}^{m} \log P(y_t | y_{<t}, X)
$$

Data augmentation strategies, including case variations, word order shuffling, and vocabulary recombination, increase the effective dataset size, enabling better generalization for low-resource languages.

# Citations

ALF-T5 relies on several foundational libraries and research:

- [@t5:2019] – T5: Text-to-Text Transfer Transformer
- [@peft:2023] – Parameter-Efficient Fine-Tuning with LoRA
- [@huggingface:2020] – Hugging Face Transformers library

# Figures

![Example translation with 45 examples demonstrating few-shot learning in a constructed language.\label{fig}](https://raw.githubusercontent.com/matjsz/alf-t5/refs/heads/main/docs/showcase.png)

![Training on a medium Portuguese-English dataset demonstrating BLEU and METEOR metrics.\label{fig}](https://raw.githubusercontent.com/matjsz/alf-t5/refs/heads/main/docs/showcase-2.png)

# Acknowledgements

We acknowledge the authors of T5 and LoRA for their research and foundational libraries. We also thank the developers of Hugging Face Transformers and PEFT for enabling accessible and efficient model training. Special thanks to Paulo Pirozelli for providing the Pirá Portuguese-English dataset used in model evaluation.

```

Software can be cited using:

```bibtex
@software{alft5,
  author = {Matheus J.G. Silva},
  title = {ALF-T5: Adaptive Neural Machine Translation Framework for Constructed and Natural Languages},
  year = {2025},
  url = {https://github.com/matjsz/alf-t5}
}
