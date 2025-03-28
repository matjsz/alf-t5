import json
from typing import List, Dict, Optional, Any

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import numpy as np
from typing import List, Dict, Any, Optional

from alf_t5.data import parse_language_data

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

def evaluate_bleu(hypothesis: str, reference: str, weights=(0.25, 0.25, 0.25, 0.25)) -> float:
    """
    Calculate BLEU score for a single translation.
    
    Args:
        hypothesis: Model's translation
        reference: Reference (ground truth) translation
        weights: Weights for unigrams, bigrams, trigrams, and 4-grams
        
    Returns:
        BLEU score (0-1)
    """
    # Tokenize sentences
    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    
    # Apply smoothing for short sentences
    smoothing = SmoothingFunction().method1
    
    # Calculate BLEU score
    return sentence_bleu([ref_tokens], hyp_tokens, weights=weights, smoothing_function=smoothing)

def evaluate_model_bleu(model, test_data_file, output_file=None, direction="t2b"):
    """
    Evaluate a trained model using BLEU score.
    
    Args:
        model: The trained model
        test_data_file: Path to test data file (format: language|english)
        output_file: Path to save detailed results (optional)
        direction: Translation direction ('c2e' or 'e2c')
    """
    # Load model
    translator = model
    
    # Load test data
    with open(test_data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    test_data = []
    for line in lines:
        line = line.strip()
        if '|' in line:
            language, english = line.split('|', 1)
            test_data.append((language.strip(), english.strip()))
    
    print(f"Loaded {len(test_data)} test examples")
    
    # Evaluate BLEU
    results = translator.evaluate_bleu(test_data, direction=direction)
    
    # Print results
    print(f"Corpus BLEU: {results['corpus_bleu']:.4f}")
    print(f"Mean BLEU: {results['mean_bleu']:.4f}")
    print(f"Median BLEU: {results['median_bleu']:.4f}")
    print(f"Min BLEU: {results['min_bleu']:.4f}")
    print(f"Max BLEU: {results['max_bleu']:.4f}")
    
    # Print example translations
    print("\nExample Translations:")
    for i, example in enumerate(results["examples"]):
        if direction == "t2b":
            print(f"Example {i+1}:")
            print(f"Conlang: {example['language']}")
            print(f"Reference: {example['reference']}")
            print(f"Translation: {example['translation']}")
            print(f"BLEU: {example['bleu']:.4f}")
        else:
            print(f"Example {i+1}:")
            print(f"English: {example['english']}")
            print(f"Reference: {example['reference']}")
            print(f"Translation: {example['translation']}")
            print(f"BLEU: {example['bleu']:.4f}")
        print()
    
    # Save detailed results
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to {output_file}")
    
    return results

def interpret_bleu_score(bleu_score, dataset_size=None):
    """
    Interpret BLEU score for language translation.
    
    Args:
        bleu_score: BLEU score (0-1)
        dataset_size: Size of training dataset (optional)
        
    Returns:
        Dictionary with interpretation
    """
    # Base interpretation
    if bleu_score < 0.10:
        quality = "Poor"
        description = "Translations are mostly incorrect or nonsensical."
    elif bleu_score < 0.20:
        quality = "Fair"
        description = "Some words are translated correctly, but grammar is incorrect."
    elif bleu_score < 0.30:
        quality = "Moderate"
        description = "Translations are understandable but contain significant errors."
    elif bleu_score < 0.40:
        quality = "Good"
        description = "Translations are mostly correct with some minor errors."
    elif bleu_score < 0.50:
        quality = "Very Good"
        description = "Translations are fluent and accurate with few errors."
    else:
        quality = "Excellent"
        description = "Translations are nearly perfect."
    
    # Adjust interpretation based on dataset size
    if dataset_size is not None:
        if dataset_size < 50:
            relative_quality = "Excellent" if bleu_score > 0.25 else \
                              "Very Good" if bleu_score > 0.20 else \
                              "Good" if bleu_score > 0.15 else \
                              "Moderate" if bleu_score > 0.10 else \
                              "Fair" if bleu_score > 0.05 else "Poor"
            
            context = f"For a small dataset of {dataset_size} examples, this is {relative_quality}."
        elif dataset_size < 200:
            relative_quality = "Excellent" if bleu_score > 0.35 else \
                              "Very Good" if bleu_score > 0.30 else \
                              "Good" if bleu_score > 0.25 else \
                              "Moderate" if bleu_score > 0.15 else \
                              "Fair" if bleu_score > 0.10 else "Poor"
            
            context = f"For a medium dataset of {dataset_size} examples, this is {relative_quality}."
        else:
            relative_quality = "Excellent" if bleu_score > 0.45 else \
                              "Very Good" if bleu_score > 0.40 else \
                              "Good" if bleu_score > 0.30 else \
                              "Moderate" if bleu_score > 0.20 else \
                              "Fair" if bleu_score > 0.15 else "Poor"
            
            context = f"For a large dataset of {dataset_size} examples, this is {relative_quality}."
    else:
        context = None
    
    return {
        "score": bleu_score,
        "quality": quality,
        "description": description,
        "context": context
    }

# New function to evaluate METEOR score
def evaluate_meteor(references: List[List[str]], hypotheses: List[str]) -> Dict[str, float]:
    """
    Evaluate translations using METEOR score
    
    Args:
        references: List of reference translations (tokenized)
        hypotheses: List of hypothesis translations (tokenized)
        
    Returns:
        Dictionary containing METEOR scores (corpus and per-sentence)
    """
    # Initialize smoothing function for METEOR
    per_sentence_scores = []
    
    # Calculate METEOR for each sentence
    for i, hyp in enumerate(hypotheses):
        refs = references[i]
        score = meteor_score(refs, hyp)
        per_sentence_scores.append(score)
    
    # Calculate corpus-level METEOR score
    corpus_meteor_score = sum(per_sentence_scores) / len(per_sentence_scores) if per_sentence_scores else 0
    
    return {
        "corpus_meteor": corpus_meteor_score,
        "sentence_meteor": per_sentence_scores
    }

def interpret_meteor_score(meteor_score: float, dataset_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Interpret a METEOR score, providing human-readable quality assessment
    
    Args:
        meteor_score: METEOR score (0-1)
        dataset_size: Optional size of the dataset that produced this score
        
    Returns:
        Dictionary with interpretation details
    """
    # Basic quality assessment
    if meteor_score < 0.20:
        quality = "Poor"
        description = "Translations are mostly incorrect or nonsensical."
    elif meteor_score < 0.30:
        quality = "Fair"
        description = "Some words are translated correctly, but grammar is incorrect."
    elif meteor_score < 0.40:
        quality = "Moderate"
        description = "Translations are understandable but contain significant errors."
    elif meteor_score < 0.50:
        quality = "Good"
        description = "Translations are mostly correct with some minor errors."
    elif meteor_score < 0.60:
        quality = "Very Good"
        description = "Translations are fluent and accurate with few errors."
    else:
        quality = "Excellent"
        description = "Translations are nearly perfect."
    
    # Adjust interpretation based on dataset size
    context = ""
    if dataset_size is not None:
        if dataset_size < 50:
            relative_quality = "Excellent" if meteor_score > 0.25 else \
                              "Very Good" if meteor_score > 0.20 else \
                              "Good" if meteor_score > 0.15 else \
                              "Moderate" if meteor_score > 0.10 else \
                              "Fair" if meteor_score > 0.05 else "Poor"
            context = f"For a small dataset of {dataset_size} examples, this is {relative_quality}."
        elif dataset_size < 200:
            relative_quality = "Excellent" if meteor_score > 0.35 else \
                              "Very Good" if meteor_score > 0.30 else \
                              "Good" if meteor_score > 0.25 else \
                              "Moderate" if meteor_score > 0.15 else \
                              "Fair" if meteor_score > 0.10 else "Poor"
            context = f"For a medium dataset of {dataset_size} examples, this is {relative_quality}."
        else:
            relative_quality = "Excellent" if meteor_score > 0.45 else \
                              "Very Good" if meteor_score > 0.40 else \
                              "Good" if meteor_score > 0.30 else \
                              "Moderate" if meteor_score > 0.20 else \
                              "Fair" if meteor_score > 0.15 else "Poor"
            context = f"For a large dataset of {dataset_size} examples, this is {relative_quality}."
    
    return {
        "score": meteor_score,
        "quality": quality,
        "description": description,
        "context": context
    }

# New function to evaluate model with METEOR score
def evaluate_model_meteor(model, test_data_file, output_file=None, direction="t2b"):
    """
    Evaluate a trained model using METEOR score on a test dataset
    
    Args:
        model: The trained model
        test_data_file: File containing test data in format "source|target"
        output_file: Optional file to save evaluation results
        direction: Translation direction ('c2e' or 'e2c')
        
    Returns:
        Dictionary containing METEOR scores and examples
    """
    # Load the model
    translator = model
    
    # Load test data
    with open(test_data_file, 'r', encoding='utf-8') as f:
        test_data = f.read()
    
    test_pairs = parse_language_data(test_data)
    
    # Prepare for evaluation
    references = []
    hypotheses = []
    examples = []
    
    # Process depending on direction
    for language, english in test_pairs:
        if direction == "t2b":
            source, reference = language, english
        else:  # e2c
            source, reference = english, language
        
        # Translate
        translation = translator.translate(source, direction=direction)
        
        # Tokenize for METEOR calculation
        reference_tokens = nltk.word_tokenize(reference.lower())
        translation_tokens = nltk.word_tokenize(translation.lower())
        
        # Add to references and hypotheses
        references.append([reference_tokens])  # METEOR expects a list of reference lists
        hypotheses.append(translation_tokens)
        
        # Store example
        examples.append({
            "source": source,
            "reference": reference,
            "translation": translation
        })
    
    # Calculate METEOR scores
    meteor_results = evaluate_meteor(references, hypotheses)
    
    # Prepare results
    results = {
        "direction": direction,
        "num_examples": len(test_pairs),
        "corpus_meteor": meteor_results["corpus_meteor"],
        "examples": examples[:10]  # First 10 examples
    }
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
    
    return results

def evaluate(
    hypotheses: List[str],
    references: List[List[str]],
    metrics: List[str] = ["bleu", "meteor"]
) -> Dict[str, Any]:
    """
    Evaluate translations using multiple metrics
    
    Args:
        hypotheses: List of hypothesis translations (tokenized)
        references: List of reference translations (tokenized)
        metrics: List of metrics to calculate. Options: "bleu", "meteor"
        
    Returns:
        Dictionary containing evaluation scores
    """
    results = {}
    
    # Calculate requested metrics
    if "bleu" in metrics:
        bleu_results = evaluate_bleu(hypotheses, references)
        results.update({
            "corpus_bleu": bleu_results["corpus_bleu"],
            "sentence_bleu": bleu_results["sentence_bleu"]
        })
    
    if "meteor" in metrics:
        meteor_results = evaluate_meteor(references, hypotheses)
        results.update({
            "corpus_meteor": meteor_results["corpus_meteor"],
            "sentence_meteor": meteor_results["sentence_meteor"]
        })
    
    return results