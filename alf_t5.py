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

import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
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
    translator = T5ConlangTranslator.load(model_path)
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
        output_dir: str = "alf_t5_translator"
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
            "learning_rates": []
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
        
        # Add BLEU tracking to history
        self.history["bleu_scores"] = []
        
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