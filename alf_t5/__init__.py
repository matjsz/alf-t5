"""
ALF-T5: Adaptative Language Framework for T5
==============================================

A framework for training and evaluating language translation models,
particularly focused on constructed languages (conlangs).

Main components:
- ALFT5Translator: Core translation class based on T5
- ALFT5Dataset: ALF-T5 Dataset class
- ModelExperiment: Hyperparameter optimization
- Data: Data handling tools for language pairs
- Evaluation: BLEU and METEOR score calculation
"""

__version__ = "1.1.2"

from alf_t5.translator import ALFT5Translator
from alf_t5.dataset import ALFDataset
from alf_t5.experiment import ModelExperiment
import alf_t5.data
import alf_t5.evaluation

__all__ = [
    "ALFT5Translator",
    "ALFDataset",
    "ModelExperiment",
    "data",
    "evaluation"
]