import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
import os
import json
import time
import random
from typing import List, Tuple, Dict, Optional, Any, Union
from collections import defaultdict, Counter
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import itertools
import concurrent.futures
import pandas as pd

import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Any, Optional

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

# For pre-trained models
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorForSeq2Seq
)

# For PEFT (Parameter-Efficient Fine-Tuning)
from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType, 
    PeftModel
)

# For mixed precision training
from torch.cuda.amp import autocast, GradScaler

# Set random seeds for reproducibility
def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Parse language data
def parse_language_data(data_string: str) -> List[Tuple[str, str]]:
    """Parse language data in the format language|translation."""
    pairs = []
    for line in data_string.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
            
        language, english = line.split("|")
        pairs.append((language, english))
    return pairs

# Data augmentation techniques
def augment_data(data_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Augment data with various techniques."""
    augmented_pairs = data_pairs.copy()
    
    # 1. Add capitalized versions
    for language, english in data_pairs:
        if len(language) > 0 and len(english) > 0:
            augmented_pairs.append((language.capitalize(), english.capitalize()))
    
    # 2. Add reversed word order for multi-word phrases
    for language, english in data_pairs:
        language_words = language.split()
        english_words = english.split()
        
        if len(language_words) > 1 and len(english_words) > 1:
            reversed_language = ' '.join(language_words[::-1])
            reversed_english = ' '.join(english_words[::-1])
            augmented_pairs.append((reversed_language, reversed_english))
    
    # 3. Create new combinations from existing vocabulary
    language_word_map = {}
    english_word_map = {}
    
    # Build word mappings
    for language, english in data_pairs:
        language_words = language.split()
        english_words = english.split()
        
        if len(language_words) == len(english_words):
            for c_word, e_word in zip(language_words, english_words):
                if c_word not in language_word_map:
                    language_word_map[c_word] = []
                if e_word not in english_word_map:
                    english_word_map[e_word] = []
                
                language_word_map[c_word].append(e_word)
                english_word_map[e_word].append(c_word)
    
    # Create new combinations
    for language, english in data_pairs:
        language_words = language.split()
        english_words = english.split()
        
        if len(language_words) > 1 and len(english_words) > 1:
            # Swap one word
            for i in range(len(language_words)):
                if language_words[i] in language_word_map and len(language_word_map[language_words[i]]) > 1:
                    for alt_english in language_word_map[language_words[i]]:
                        if alt_english != english_words[i]:
                            new_english_words = english_words.copy()
                            new_english_words[i] = alt_english
                            augmented_pairs.append((language, ' '.join(new_english_words)))
                            break
    
    return augmented_pairs

def calculate_bleu(reference: str, hypothesis: str, weights=(0.25, 0.25, 0.25, 0.25)) -> float:
    """
    Calculate BLEU score for a single translation.
    
    Args:
        reference: Reference (ground truth) translation
        hypothesis: Model's translation
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

def evaluate_model_bleu(model_path, test_data_file, output_file=None, direction="c2e"):
    """
    Evaluate a trained model using BLEU score.
    
    Args:
        model_path: Path to the trained model
        test_data_file: Path to test data file (format: language|english)
        output_file: Path to save detailed results (optional)
        direction: Translation direction ('c2e' or 'e2c')
    """
    # Load model
    translator = ALFT5Translator.load(model_path)
    print(f"Model loaded from {model_path}")
    
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
        if direction == "c2e":
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

# Dataset for Conlang Translation
class ALFDataset(Dataset):
    """Dataset for language translation pairs."""
    def __init__(
        self, 
        data_pairs: List[Tuple[str, str]], 
        tokenizer,
        max_length: int = 128,
        direction: str = "c2e"
    ):
        self.data_pairs = data_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.direction = direction
        
        # Set prefix based on direction
        if direction == "c2e":
            self.prefix = "translate language to english: "
        else:  # "e2c"
            self.prefix = "translate english to language: "
    
    def __len__(self) -> int:
        return len(self.data_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a data item."""
        src, tgt = self.data_pairs[idx]
        
        # Swap source and target if direction is e2c
        if self.direction == "e2c":
            src, tgt = tgt, src
        
        # Add prefix
        src_text = f"{self.prefix}{src}"
        
        # Tokenize source and target
        src_encoding = self.tokenizer(
            src_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        tgt_encoding = self.tokenizer(
            tgt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        src_ids = src_encoding["input_ids"].squeeze()
        src_mask = src_encoding["attention_mask"].squeeze()
        tgt_ids = tgt_encoding["input_ids"].squeeze()
        
        # Replace padding token id with -100 for loss calculation
        labels = tgt_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": src_ids,
            "attention_mask": src_mask,
            "labels": labels,
            "src_text": src,
            "tgt_text": tgt
        }

# T5 ALF Implementation
class ALFT5Translator:
    """ALF translator using T5 model."""
    def __init__(
        self,
        model_name: str = "t5-small",
        use_peft: bool = True,
        peft_r: int = 8,
        peft_lora_alpha: int = 32,
        peft_lora_dropout: float = 0.1,
        batch_size: int = 8,
        lr: float = 5e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 20,
        max_length: int = 128,
        use_fp16: bool = True,
        warmup_ratio: float = 0.1,
        output_dir: str = "alf_t5_translator",
        eval_bleu: bool = True,
        eval_meteor: bool = True
    ):
        self.model_name = model_name
        self.use_peft = use_peft
        self.peft_r = peft_r
        self.peft_lora_alpha = peft_lora_alpha
        self.peft_lora_dropout = peft_lora_dropout
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.warmup_ratio = warmup_ratio
        self.output_dir = output_dir
        self.eval_bleu = eval_bleu
        self.eval_meteor = eval_meteor
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler() if use_fp16 else None
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": [],
            "bleu_scores": [],
            "meteor_scores": []
        }
    
    def _initialize_model_and_tokenizer(self):
        """Initialize model and tokenizer."""
        print(f"Initializing model and tokenizer from {self.model_name}...")
        
        # Load tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        
        # Load model
        base_model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        
        # Apply PEFT if requested
        if self.use_peft:
            print("Applying LoRA for parameter-efficient fine-tuning...")
            
            # Configure PEFT (LoRA)
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=self.peft_r,
                lora_alpha=self.peft_lora_alpha,
                lora_dropout=self.peft_lora_dropout,
                target_modules=["q", "v"],  # Target attention modules
                bias="none"
            )
            
            # Create model with PEFT
            self.model = get_peft_model(base_model, peft_config)
            self.model.print_trainable_parameters()
        else:
            self.model = base_model
        
        # Move model to device
        self.model.to(self.device)
    
    def _initialize_optimizer_and_scheduler(self, num_training_steps):
        """Initialize optimizer and scheduler."""
        # Prepare optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.lr)
        
        # Prepare scheduler
        warmup_steps = int(num_training_steps * self.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def evaluate_bleu(
        self,
        test_data: List[Tuple[str, str]],
        direction: str = "c2e",
        batch_size: int = 8,
        ngram_weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    ) -> Dict[str, Any]:
        """Evaluate model using BLEU score on test data."""
        self.model.eval()
        
        all_sources = []
        all_targets = []
        all_translations = []
        
        # Process in batches
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i+batch_size]
            
            # Prepare source and reference texts
            if direction == "c2e":
                sources = [pair[0] for pair in batch]
                targets = [pair[1] for pair in batch]
            else:  # "e2c"
                sources = [pair[1] for pair in batch]
                targets = [pair[0] for pair in batch]
            
            all_sources.extend(sources)
            all_targets.extend(targets)
            
            # Generate translations
            translations = self.batch_translate(sources, direction)
            all_translations.extend(translations)
        
        # Prepare references and hypotheses for BLEU calculation
        references = [[nltk.word_tokenize(target.lower())] for target in all_targets]
        hypotheses = [nltk.word_tokenize(translation.lower()) for translation in all_translations]
        
        # Calculate corpus BLEU
        smoothing = SmoothingFunction().method1
        corpus_bleu_score = corpus_bleu(references, hypotheses, weights=ngram_weights, smoothing_function=smoothing)
        
        # Calculate individual BLEU scores
        individual_scores = [
            sentence_bleu(ref, hyp, weights=ngram_weights, smoothing_function=smoothing)
            for ref, hyp in zip(references, hypotheses)
        ]
        
        # Calculate statistics
        mean_score = np.mean(individual_scores)
        median_score = np.median(individual_scores)
        min_score = np.min(individual_scores)
        max_score = np.max(individual_scores)
        
        # Prepare example results
        examples = []
        for i in range(min(10, len(test_data))):
            examples.append({
                "source": all_sources[i],
                "reference": all_targets[i],
                "translation": all_translations[i],
                "bleu": individual_scores[i]
            })
        
        return {
            "corpus_bleu": corpus_bleu_score,
            "mean_bleu": mean_score,
            "median_bleu": median_score,
            "min_bleu": min_score,
            "max_bleu": max_score,
            "examples": examples
        }

    def batch_translate(
        self,
        texts: List[str],
        direction: str = "c2e",
        max_length: int = None,
        num_beams: int = 5
    ) -> List[str]:
        """Translate a batch of texts."""
        if max_length is None:
            max_length = self.max_length
        
        # Set prefix based on direction
        if direction == "c2e":
            prefix = "translate language to english: "
        else:  # "e2c"
            prefix = "translate english to language: "
        
        # Prepare inputs
        input_texts = [f"{prefix}{text}" for text in texts]
        
        # Tokenize inputs
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Move to device
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Generate translations
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )
        
        # Decode outputs
        translations = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        return translations
    
    def train(
        self,
        data_string: str,
        test_size: float = 0.1,
        augment: bool = True,
        early_stopping_patience: int = 5,
        eval_bleu: bool = True,
        bleu_eval_steps: int = 5  # Evaluate BLEU every N epochs
    ):
        """Train the translator on language data with BLEU evaluation."""
        # Parse data
        data_pairs = parse_language_data(data_string)
        if not data_pairs:
            raise ValueError("No valid data pairs found in the input string")
        
        print(f"Parsed {len(data_pairs)} data pairs")
        
        # Augment data if requested
        if augment:
            data_pairs = augment_data(data_pairs)
            print(f"Augmented to {len(data_pairs)} data pairs")
        
        # Split data into train and test
        random.shuffle(data_pairs)
        split = int((1 - test_size) * len(data_pairs))
        train_data = data_pairs[:split]
        test_data = data_pairs[split:]
        
        print(f"Training on {len(train_data)} examples, testing on {len(test_data)} examples")
        
        # Initialize model and tokenizer
        self._initialize_model_and_tokenizer()
        
        # Create datasets for both directions (c2e and e2c)
        train_dataset_c2e = ALFDataset(
            train_data,
            self.tokenizer,
            max_length=self.max_length,
            direction="c2e"
        )
        
        train_dataset_e2c = ALFDataset(
            train_data,
            self.tokenizer,
            max_length=self.max_length,
            direction="e2c"
        )
        
        # Combine datasets for bidirectional training
        train_dataset = torch.utils.data.ConcatDataset([train_dataset_c2e, train_dataset_e2c])
        
        # Create validation datasets
        val_dataset_c2e = ALFDataset(
            test_data,
            self.tokenizer,
            max_length=self.max_length,
            direction="c2e"
        )
        
        val_dataset_e2c = ALFDataset(
            test_data,
            self.tokenizer,
            max_length=self.max_length,
            direction="e2c"
        )
        
        # Combine validation datasets
        val_dataset = torch.utils.data.ConcatDataset([val_dataset_c2e, val_dataset_e2c])
        
        # Create data loaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        # Initialize optimizer and scheduler
        num_training_steps = len(train_dataloader) * self.num_epochs
        self._initialize_optimizer_and_scheduler(num_training_steps)
        
        # Training loop
        best_val_loss = float("inf")
        best_bleu_score = 0.0
        patience_counter = 0
        
        print(f"Starting training for {self.num_epochs} epochs...")
        
        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_steps = 0
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
            
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass with mixed precision
                if self.use_fp16:
                    with autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss
                    
                    # Backward pass with gradient scaling
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard training without mixed precision
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                # Update scheduler
                self.scheduler.step()
                
                # Update progress bar
                train_loss += loss.item()
                train_steps += 1
                progress_bar.set_postfix({"loss": loss.item()})
                
                # Record learning rate
                self.history["learning_rates"].append(self.scheduler.get_last_lr()[0])
            
            # Calculate average training loss
            avg_train_loss = train_loss / train_steps
            self.history["train_loss"].append(avg_train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_steps = 0
            
            progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Val]")
            
            with torch.no_grad():
                for batch in progress_bar:
                    # Move batch to device
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    # Update validation loss
                    val_loss += outputs.loss.item()
                    val_steps += 1
                    progress_bar.set_postfix({"loss": outputs.loss.item()})
            
            # Calculate average validation loss
            avg_val_loss = val_loss / val_steps
            self.history["val_loss"].append(avg_val_loss)
            
            if eval_bleu and (epoch + 1) % bleu_eval_steps == 0:
                print(f"Evaluating BLEU score after epoch {epoch+1}...")
                
                # Evaluate on test data
                bleu_results = self.evaluate_bleu(test_data, direction="c2e")
                
                # Store in history
                if "bleu_scores" not in self.history:
                    self.history["bleu_scores"] = []
                    
                self.history["bleu_scores"].append({
                    "epoch": epoch + 1,
                    "corpus_bleu": bleu_results["corpus_bleu"],
                    "mean_bleu": bleu_results["mean_bleu"]
                })
                
                print(f"Epoch {epoch+1}: BLEU Score: {bleu_results['corpus_bleu']:.4f}")
            else:
                # Print epoch summary
                print(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Check for improvement
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save best model
                self._save_checkpoint(f"{self.output_dir}/best_model")
                print(f"New best model saved with validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epochs")
                
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0 or (epoch + 1) == self.num_epochs:
                self._save_checkpoint(f"{self.output_dir}/checkpoint-epoch-{epoch+1}")
            
            # Add meteor score tracking
            best_meteor_score = 0.0
            
            # Add METEOR evaluation
            if self.eval_meteor:
                if "meteor_scores" not in self.history:
                    self.history["meteor_scores"] = []
                    
                # Calculate METEOR scores
                meteor_results = self.evaluate_meteor(test_data)
                
                # Record METEOR scores
                self.history["meteor_scores"].append({
                    "epoch": epoch,
                    "corpus_meteor": meteor_results["corpus_meteor"],
                    "batch_size": self.batch_size,
                    "learning_rate": self.lr
                })
                
                # Update best METEOR score
                if meteor_results["corpus_meteor"] > best_meteor_score:
                    best_meteor_score = meteor_results["corpus_meteor"]
                    if self.save_best_meteor:
                        self.save(f"{self.output_dir}/best_meteor_model")
                    
                # Log METEOR score
                print(f"METEOR Score: {meteor_results['corpus_meteor']:.4f}")
        
        if eval_bleu:
            print("Performing final BLEU evaluation...")
            final_bleu_results = self.evaluate_bleu(test_data, direction="c2e")
            print(f"Final BLEU Score: {final_bleu_results['corpus_bleu']:.4f}")
            
            # Save detailed BLEU results
            with open(f"{self.output_dir}/bleu_results.json", "w") as f:
                json.dump(final_bleu_results, f, indent=2)

        # Save final model
        self._save_checkpoint(f"{self.output_dir}/final_model")
        
        # Plot training history
        self._plot_training_history()
        
        print("Training completed!")
    
    def translate(
        self,
        text: str,
        direction: str = "c2e",
        max_length: int = None,
        num_beams: int = 5,
        temperature: float = 1.0,
        top_p: float = None,
        do_sample: bool = False,
        return_confidence: bool = False
    ) -> Union[str, Tuple[str, float]]:
        """Translate text."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not initialized. Call train() first or load a saved model.")
        
        # Set max length if not provided
        if max_length is None:
            max_length = self.max_length
        
        # Set prefix based on direction
        if direction == "c2e":
            prefix = "translate language to english: "
        else:  # "e2c"
            prefix = "translate english to language: "
        
        # Prepare input
        input_text = f"{prefix}{text}"
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Move to device
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Set generation parameters
        gen_kwargs = {
            "max_length": max_length,
            "num_beams": num_beams,
            "early_stopping": True
        }
        
        # Add parameters for confidence calculation if needed
        if return_confidence:
            gen_kwargs.update({
                "return_dict_in_generate": True,
                "output_scores": True
            })
        
        if do_sample:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": temperature
            })
            
            if top_p is not None:
                gen_kwargs["top_p"] = top_p
        
        # Generate translation
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs
            )
        
        # Calculate confidence score if requested
        confidence_score = None
        if return_confidence:
            sequences = outputs.sequences
            scores = outputs.scores
            sequence = sequences[0]
            confidence_score = self._calculate_confidence_score(sequence, scores)
            translation = self.tokenizer.decode(sequence, skip_special_tokens=True)
        else:
            if hasattr(outputs, 'sequences'):
                translation = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            else:
                translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if return_confidence:
            return translation, confidence_score
        return translation
    
    def _calculate_confidence_score(self, sequence, scores):
        """Calculate confidence score from token probabilities."""
        sequence = sequence[1:]
        
        if not scores or len(scores) == 0:
            return 1.0
        
        token_probs = []
        for i, token_scores in enumerate(scores):
            if i >= len(sequence) - 1:
                break
                
            token_idx = sequence[i + 1].item()
            token_probs_dist = torch.nn.functional.softmax(token_scores[0], dim=-1)
            token_prob = token_probs_dist[token_idx].item()
            token_probs.append(token_prob)
        
        if token_probs:
            avg_prob = sum(token_probs) / len(token_probs)
            return avg_prob
        return 1.0
    
    def _save_checkpoint(self, path: str):
        """Save model checkpoint."""
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(path)
        
        # Save training history
        with open(f"{path}/training_history.json", "w") as f:
            json.dump(self.history, f)
        
        # Save training arguments
        with open(f"{path}/training_args.json", "w") as f:
            json.dump({
                "model_name": self.model_name,
                "use_peft": self.use_peft,
                "peft_r": self.peft_r,
                "peft_lora_alpha": self.peft_lora_alpha,
                "peft_lora_dropout": self.peft_lora_dropout,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "num_epochs": self.num_epochs,
                "max_length": self.max_length,
                "use_fp16": self.use_fp16,
                "warmup_ratio": self.warmup_ratio
            }, f)
    
    def _plot_training_history(self):
        """Plot training history."""
        if "bleu_scores" in self.history and self.history["bleu_scores"]:
            # Create a figure with 3 subplots
            plt.figure(figsize=(15, 5))
            
            # Plot losses
            plt.subplot(1, 3, 1)
            plt.plot(self.history["train_loss"], label="Train Loss")
            plt.plot(self.history["val_loss"], label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
            plt.grid(True)
            
            # Plot learning rate
            plt.subplot(1, 3, 2)
            plt.plot(self.history["learning_rates"])
            plt.xlabel("Step")
            plt.ylabel("Learning Rate")
            plt.title("Learning Rate Schedule")
            plt.grid(True)
            
            # Plot BLEU scores
            plt.subplot(1, 3, 3)
            epochs = [item["epoch"] for item in self.history["bleu_scores"]]
            bleu_scores = [item["corpus_bleu"] for item in self.history["bleu_scores"]]
            plt.plot(epochs, bleu_scores, marker='o')
            plt.xlabel("Epoch")
            plt.ylabel("BLEU Score")
            plt.title("BLEU Score Progress")
            plt.grid(True)
        else:
            # Original plotting code for just losses and learning rate
            plt.figure(figsize=(12, 5))
            
            # Plot losses
            plt.subplot(1, 2, 1)
            plt.plot(self.history["train_loss"], label="Train Loss")
            plt.plot(self.history["val_loss"], label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
            plt.grid(True)
            
            # Plot learning rate
            plt.subplot(1, 2, 2)
            plt.plot(self.history["learning_rates"])
            plt.xlabel("Step")
            plt.ylabel("Learning Rate")
            plt.title("Learning Rate Schedule")
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/training_history.png")
        plt.close()
    
    @classmethod
    def load(cls, model_path: str):
        """Load a saved model."""
        # Load training arguments
        with open(f"{model_path}/training_args.json", "r") as f:
            training_args = json.load(f)
        
        # Create instance with loaded arguments
        translator = cls(**training_args)
        
        # Load tokenizer
        translator.tokenizer = T5Tokenizer.from_pretrained(model_path)
        
        # Check if it's a PEFT model
        peft_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(peft_config_path):
            # Load base model
            base_model = T5ForConditionalGeneration.from_pretrained(training_args["model_name"])
            
            # Load PEFT model
            translator.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            # Load regular model
            translator.model = T5ForConditionalGeneration.from_pretrained(model_path)
        
        # Move model to device
        translator.model.to(translator.device)
        
        # Load training history if available
        history_path = f"{model_path}/training_history.json"
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                translator.history = json.load(f)
        
        return translator

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
def evaluate_model_meteor(model_path, test_data_file, output_file=None, direction="c2e"):
    """
    Evaluate a trained model using METEOR score on a test dataset
    
    Args:
        model_path: Path to the trained model
        test_data_file: File containing test data in format "source|target"
        output_file: Optional file to save evaluation results
        direction: Translation direction ('c2e' or 'e2c')
        
    Returns:
        Dictionary containing METEOR scores and examples
    """
    # Load the model
    translator = ALFT5Translator.load(model_path)
    
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
        if direction == "c2e":
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

class ModelExperiment:
    """Class for experimenting with different model architectures and hyperparameters"""
    
    def __init__(
        self,
        base_output_dir: str = "alf_t5_experiments",
        data_string: str = None,
        data_file: str = None,
        test_size: float = 0.2,
        augment: bool = True,
        metrics: List[str] = ["bleu", "meteor"],
        parallel_experiments: int = 1,
        experiment_timeout: int = 7200  # 2 hours default timeout per experiment
    ):
        """
        Initialize the model experiment framework
        
        Args:
            base_output_dir: Base directory to store experiment outputs
            data_string: String containing the training data
            data_file: File containing the training data (alternative to data_string)
            test_size: Proportion of data to use for validation
            augment: Whether to augment the training data
            metrics: List of metrics to evaluate models on
            parallel_experiments: Number of experiments to run in parallel (use with caution)
            experiment_timeout: Maximum time allowed per experiment in seconds
        """
        self.base_output_dir = base_output_dir
        self.data_string = data_string
        self.data_file = data_file
        self.test_size = test_size
        self.augment = augment
        self.metrics = metrics
        self.parallel_experiments = parallel_experiments
        self.experiment_timeout = experiment_timeout
        
        # Load data if file is provided
        if data_file and not data_string:
            with open(data_file, 'r', encoding='utf-8') as f:
                self.data_string = f.read()
        
        # Store experiment results
        self.experiments = []
        self.results = {}
        
        # Create base directory
        os.makedirs(base_output_dir, exist_ok=True)
    
    def add_experiment(
        self,
        name: str,
        model_name: str = "t5-small",
        **model_params
    ):
        """
        Add an experiment configuration
        
        Args:
            name: Name of the experiment
            model_name: Name of the T5 model to use
            **model_params: Parameters to pass to ALFT5Translator
        """
        experiment = {
            "name": name,
            "model_name": model_name,
            "model_params": model_params
        }
        self.experiments.append(experiment)
        return self
    
    def add_grid_search(
        self,
        name_prefix: str,
        model_names: List[str] = ["t5-small"],
        **param_grid
    ):
        """
        Add multiple experiments via grid search over parameters
        
        Args:
            name_prefix: Prefix for experiment names
            model_names: List of model names to try
            **param_grid: Parameter grid with lists of values to try
        
        Example:
            add_grid_search(
                "peft_experiment",
                model_names=["t5-small", "t5-base"],
                peft_r=[4, 8, 16],
                learning_rate=[1e-4, 3e-4, 5e-4]
            )
        """
        # Generate all combinations of parameters
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for model_name in model_names:
            for i, values in enumerate(itertools.product(*param_values)):
                params = dict(zip(param_names, values))
                
                # Create experiment name
                param_str = "_".join([f"{k}={v}" for k, v in params.items()])
                name = f"{name_prefix}_{model_name}_{param_str}"
                
                # Add the experiment
                self.add_experiment(
                    name=name,
                    model_name=model_name,
                    **params
                )
        
        return self
    
    def _run_experiment(self, experiment):
        """
        Run a single experiment
        
        Args:
            experiment: Experiment configuration dictionary
        
        Returns:
            Dictionary with experiment results
        """
        name = experiment["name"]
        model_name = experiment["model_name"]
        model_params = experiment["model_params"]
        
        print(f"Running experiment: {name}")
        
        # Create output directory for this experiment
        output_dir = os.path.join(self.base_output_dir, name)
        os.makedirs(output_dir, exist_ok=True)
        
        start_time = time.time()
        
        try:
            # Initialize translator
            translator = ALFT5Translator(
                model_name=model_name,
                output_dir=output_dir,
                eval_bleu="bleu" in self.metrics,
                eval_meteor="meteor" in self.metrics,
                **model_params
            )
            
            # Train the model
            translator.train(
                data_string=self.data_string,
                test_size=self.test_size,
                augment=self.augment
            )
            
            end_time = time.time()
            training_time = end_time - start_time
            
            # Get the results
            train_loss = min(translator.history["train_loss"]) if translator.history["train_loss"] else float('inf')
            val_loss = min(translator.history["val_loss"]) if translator.history["val_loss"] else float('inf')
            
            bleu_score = max([item["corpus_bleu"] for item in translator.history["bleu_scores"]]) if "bleu_scores" in translator.history and translator.history["bleu_scores"] else 0.0
            
            meteor_score = max([item["corpus_meteor"] for item in translator.history["meteor_scores"]]) if "meteor_scores" in translator.history and translator.history["meteor_scores"] else 0.0
            
            # Save results
            results = {
                "name": name,
                "model_name": model_name,
                "params": model_params,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "bleu_score": bleu_score,
                "meteor_score": meteor_score,
                "training_time": training_time,
                "successful": True
            }
            
            # Save experiment config and results
            with open(os.path.join(output_dir, "experiment_results.json"), 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4)
            
            return results
            
        except Exception as e:
            end_time = time.time()
            training_time = end_time - start_time
            
            error_message = str(e)
            print(f"Error in experiment {name}: {error_message}")
            
            # Save error information
            results = {
                "name": name,
                "model_name": model_name,
                "params": model_params,
                "error": error_message,
                "training_time": training_time,
                "successful": False
            }
            
            # Save error information
            with open(os.path.join(output_dir, "experiment_error.json"), 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4)
            
            return results
    
    def run_experiments(self):
        """
        Run all configured experiments
        
        Returns:
            DataFrame with experiment results
        """
        if not self.experiments:
            raise ValueError("No experiments configured. Add experiments with add_experiment() or add_grid_search().")
        
        if not self.data_string:
            raise ValueError("No training data provided. Set data_string or data_file.")
        
        results = []
        
        if self.parallel_experiments > 1:
            # Run experiments in parallel
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.parallel_experiments) as executor:
                future_to_exp = {
                    executor.submit(self._run_experiment, exp): exp 
                    for exp in self.experiments
                }
                
                for future in concurrent.futures.as_completed(future_to_exp):
                    exp = future_to_exp[future]
                    try:
                        result = future.result(timeout=self.experiment_timeout)
                        results.append(result)
                    except concurrent.futures.TimeoutError:
                        print(f"Experiment {exp['name']} timed out after {self.experiment_timeout} seconds")
                        results.append({
                            "name": exp["name"],
                            "model_name": exp["model_name"],
                            "params": exp["model_params"],
                            "error": "Experiment timed out",
                            "training_time": self.experiment_timeout,
                            "successful": False
                        })
        else:
            # Run experiments sequentially
            for exp in self.experiments:
                result = self._run_experiment(exp)
                results.append(result)
        
        # Store results
        self.results = {result["name"]: result for result in results}
        
        # Create a DataFrame for analysis
        results_df = pd.DataFrame(results)
        
        # Save overall results
        results_df.to_csv(os.path.join(self.base_output_dir, "experiment_results.csv"), index=False)
        
        with open(os.path.join(self.base_output_dir, "experiment_results.json"), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        
        return results_df
    
    def get_best_experiment(self, metric="val_loss", higher_is_better=False):
        """
        Get the best experiment according to a metric
        
        Args:
            metric: Metric to use for comparison ('val_loss', 'bleu_score', 'meteor_score')
            higher_is_better: Whether higher values of the metric are better
            
        Returns:
            Dictionary with best experiment details
        """
        if not self.results:
            raise ValueError("No experiment results available. Run experiments first.")
        
        # Filter experiments that completed successfully
        successful_experiments = [exp for exp in self.results.values() if exp["successful"]]
        
        if not successful_experiments:
            raise ValueError("No successful experiments to compare.")
        
        # Sort by metric
        if higher_is_better:
            best_exp = max(successful_experiments, key=lambda x: x.get(metric, float('-inf')))
        else:
            best_exp = min(successful_experiments, key=lambda x: x.get(metric, float('inf')))
            
        return best_exp
    
    def plot_experiment_results(self, metrics=None, sort_by=None, ascending=True):
        """
        Plot experiment results
        
        Args:
            metrics: List of metrics to plot (default: ['train_loss', 'val_loss', 'bleu_score', 'meteor_score'])
            sort_by: Column to sort results by
            ascending: Whether to sort in ascending order
        """
        if not self.results:
            raise ValueError("No experiment results available. Run experiments first.")
        
        # Default metrics to plot
        if metrics is None:
            metrics = ['train_loss', 'val_loss', 'bleu_score', 'meteor_score']
        
        # Create DataFrame from results
        df = pd.DataFrame([
            {**result, **result["params"]} 
            for result in self.results.values() 
            if result["successful"]
        ])
        
        # Drop the params column as we've expanded it
        if "params" in df.columns:
            df = df.drop(columns=["params"])
        
        # Sort if requested
        if sort_by:
            df = df.sort_values(by=sort_by, ascending=ascending)
        
        # Plot each metric
        for metric in metrics:
            if metric in df.columns:
                plt.figure(figsize=(12, 6))
                ax = df.plot(kind='bar', x='name', y=metric, legend=False)
                plt.title(f'{metric} by Experiment')
                plt.xlabel('Experiment')
                plt.ylabel(metric)
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.savefig(os.path.join(self.base_output_dir, f"{metric}_comparison.png"))
                plt.close()
        
        # Plot training time
        if 'training_time' in df.columns:
            plt.figure(figsize=(12, 6))
            ax = df.plot(kind='bar', x='name', y='training_time', legend=False)
            plt.title('Training Time by Experiment')
            plt.xlabel('Experiment')
            plt.ylabel('Training Time (s)')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(self.base_output_dir, "training_time_comparison.png"))
            plt.close()
        
        # Create correlation matrix
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            plt.figure(figsize=(10, 8))
            plt.imshow(corr, cmap='coolwarm', interpolation='none', aspect='auto')
            plt.colorbar()
            plt.xticks(range(len(corr)), corr.columns, rotation=90)
            plt.yticks(range(len(corr)), corr.columns)
            plt.title('Parameter Correlation Matrix')
            
            # Add correlation values
            for i in range(len(corr)):
                for j in range(len(corr)):
                    text = plt.text(j, i, f'{corr.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black")
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.base_output_dir, "correlation_matrix.png"))
            plt.close()
        
        return df
