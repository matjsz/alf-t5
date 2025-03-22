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
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict, Counter
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

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

# Parse conlang data
def parse_conlang_data(data_string: str) -> List[Tuple[str, str]]:
    """Parse conlang data in the format conlang|translationd."""
    pairs = []
    for line in data_string.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
            
        conlang, english = line.split("|")
        pairs.append((conlang, english))
    return pairs

# Data augmentation techniques
def augment_data(data_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Augment data with various techniques."""
    augmented_pairs = data_pairs.copy()
    
    # 1. Add capitalized versions
    for conlang, english in data_pairs:
        if len(conlang) > 0 and len(english) > 0:
            augmented_pairs.append((conlang.capitalize(), english.capitalize()))
    
    # 2. Add reversed word order for multi-word phrases
    for conlang, english in data_pairs:
        conlang_words = conlang.split()
        english_words = english.split()
        
        if len(conlang_words) > 1 and len(english_words) > 1:
            reversed_conlang = ' '.join(conlang_words[::-1])
            reversed_english = ' '.join(english_words[::-1])
            augmented_pairs.append((reversed_conlang, reversed_english))
    
    # 3. Create new combinations from existing vocabulary
    conlang_word_map = {}
    english_word_map = {}
    
    # Build word mappings
    for conlang, english in data_pairs:
        conlang_words = conlang.split()
        english_words = english.split()
        
        if len(conlang_words) == len(english_words):
            for c_word, e_word in zip(conlang_words, english_words):
                if c_word not in conlang_word_map:
                    conlang_word_map[c_word] = []
                if e_word not in english_word_map:
                    english_word_map[e_word] = []
                
                conlang_word_map[c_word].append(e_word)
                english_word_map[e_word].append(c_word)
    
    # Create new combinations
    for conlang, english in data_pairs:
        conlang_words = conlang.split()
        english_words = english.split()
        
        if len(conlang_words) > 1 and len(english_words) > 1:
            # Swap one word
            for i in range(len(conlang_words)):
                if conlang_words[i] in conlang_word_map and len(conlang_word_map[conlang_words[i]]) > 1:
                    for alt_english in conlang_word_map[conlang_words[i]]:
                        if alt_english != english_words[i]:
                            new_english_words = english_words.copy()
                            new_english_words[i] = alt_english
                            augmented_pairs.append((conlang, ' '.join(new_english_words)))
                            break
    
    return augmented_pairs

# Dataset for Conlang Translation
class CoALDataset(Dataset):
    """Dataset for conlang translation pairs."""
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
            self.prefix = "translate conlang to english: "
        else:  # "e2c"
            self.prefix = "translate english to conlang: "
    
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

# T5 CoAL Implementation
class CoALT5Translator:
    """CoAL translator using T5 model."""
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
        output_dir: str = "coal_t5_translator"
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
    
    def train(
        self,
        data_string: str,
        test_size: float = 0.1,
        augment: bool = True,
        early_stopping_patience: int = 5
    ):
        """Train the translator on conlang data."""
        # Parse data
        data_pairs = parse_conlang_data(data_string)
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
        train_dataset_c2e = CoALDataset(
            train_data,
            self.tokenizer,
            max_length=self.max_length,
            direction="c2e"
        )
        
        train_dataset_e2c = CoALDataset(
            train_data,
            self.tokenizer,
            max_length=self.max_length,
            direction="e2c"
        )
        
        # Combine datasets for bidirectional training
        train_dataset = torch.utils.data.ConcatDataset([train_dataset_c2e, train_dataset_e2c])
        
        # Create validation datasets
        val_dataset_c2e = CoALDataset(
            test_data,
            self.tokenizer,
            max_length=self.max_length,
            direction="c2e"
        )
        
        val_dataset_e2c = CoALDataset(
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
        do_sample: bool = False
    ) -> str:
        """Translate text."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not initialized. Call train() first or load a saved model.")
        
        # Set max length if not provided
        if max_length is None:
            max_length = self.max_length
        
        # Set prefix based on direction
        if direction == "c2e":
            prefix = "translate conlang to english: "
        else:  # "e2c"
            prefix = "translate english to conlang: "
        
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
        
        # Decode output
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return translation
    
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