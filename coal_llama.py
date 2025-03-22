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
    AutoModelForCausalLM, AutoTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling
)

# For PEFT (Parameter-Efficient Fine-Tuning)
from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType, 
    PeftModel,
    prepare_model_for_kbit_training
)

# For mixed precision training
from torch.cuda.amp import autocast, GradScaler

# For quantization
import bitsandbytes as bnb

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

# Dataset for CoAL Llama Translation
class LlamaCoALDataset(Dataset):
    """Dataset for conlang translation using Llama."""
    def __init__(
        self, 
        data_pairs: List[Tuple[str, str]], 
        tokenizer,
        max_length: int = 512,
        include_both_directions: bool = True
    ):
        self.data_pairs = data_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_both_directions = include_both_directions
        
        # Format prompts
        self.formatted_examples = []
        
        for conlang, english in data_pairs:
            # Format for conlang to english
            c2e_prompt = f"Translate from conlang to English:\nConlang: {conlang}\nEnglish:"
            c2e_completion = f" {english}</s>"
            self.formatted_examples.append((c2e_prompt, c2e_completion))
            
            # Format for english to conlang if requested
            if include_both_directions:
                e2c_prompt = f"Translate from English to conlang:\nEnglish: {english}\nConlang:"
                e2c_completion = f" {conlang}</s>"
                self.formatted_examples.append((e2c_prompt, e2c_completion))
    
    def __len__(self) -> int:
        return len(self.formatted_examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a data item."""
        prompt, completion = self.formatted_examples[idx]
        
        # Combine prompt and completion for training
        full_text = prompt + completion
        
        # Tokenize
        encodings = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Create labels (same as input_ids for causal LM)
        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()
        
        # Create labels with -100 for prompt tokens
        labels = input_ids.clone()
        
        # Find the position where completion starts
        prompt_len = len(self.tokenizer(prompt, return_tensors="pt")["input_ids"][0])
        labels[:prompt_len] = -100  # Mask prompt tokens
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# CoAL Llama Translator
class LlamaCoALTranslator:
    """Conlang translator using Llama model."""
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        use_4bit: bool = True,
        use_peft: bool = True,
        peft_r: int = 16,
        peft_lora_alpha: int = 32,
        peft_lora_dropout: float = 0.05,
        batch_size: int = 1,
        lr: float = 2e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 3,
        max_length: int = 512,
        warmup_ratio: float = 0.03,
        output_dir: str = "llama_conlang_translator"
    ):
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.use_peft = use_peft
        self.peft_r = peft_r
        self.peft_lora_alpha = peft_lora_alpha
        self.peft_lora_dropout = peft_lora_dropout
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.max_length = max_length
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
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Ensure the tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization if requested
        if self.use_4bit:
            print("Loading model with 4-bit quantization...")
            quantization_config = {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True
            }
            
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
            
            # Prepare model for k-bit training
            base_model = prepare_model_for_kbit_training(base_model)
        else:
            base_model = AutoModelForCausalLM.from_pretrained(self.model_name)
            base_model.to(self.device)
        
        # Apply PEFT if requested
        if self.use_peft:
            print("Applying LoRA for parameter-efficient fine-tuning...")
            
            # Configure PEFT (LoRA)
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
            
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.peft_r,
                lora_alpha=self.peft_lora_alpha,
                lora_dropout=self.peft_lora_dropout,
                target_modules=target_modules,
                bias="none"
            )
            
            # Create model with PEFT
            self.model = get_peft_model(base_model, peft_config)
            self.model.print_trainable_parameters()
        else:
            self.model = base_model
    
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
        early_stopping_patience: int = 3
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
        
        # Create datasets
        train_dataset = LlamaCoALDataset(
            train_data,
            self.tokenizer,
            max_length=self.max_length
        )
        
        val_dataset = LlamaCoALDataset(
            test_data,
            self.tokenizer,
            max_length=self.max_length
        )
        
        # Create data loaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
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
                
                # Forward pass
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
            if (epoch + 1) % 1 == 0 or (epoch + 1) == self.num_epochs:
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
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """Translate text."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not initialized. Call train() first or load a saved model.")
        
        # Format prompt based on direction
        if direction == "c2e":
            prompt = f"Translate from conlang to English:\nConlang: {text}\nEnglish:"
        else:  # "e2c"
            prompt = f"Translate from English to conlang:\nEnglish: {text}\nConlang:"
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        
        # Set generation parameters
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id
        }
        
        if do_sample:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p
            })
        
        # Generate translation
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(input_ids, **gen_kwargs)
        
        # Decode output
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract translation from output
        if direction == "c2e":
            translation = full_output.split("English:")[1].strip()
        else:  # "e2c"
            translation = full_output.split("Conlang:")[1].strip()
        
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
                "use_4bit": self.use_4bit,
                "use_peft": self.use_peft,
                "peft_r": self.peft_r,
                "peft_lora_alpha": self.peft_lora_alpha,
                "peft_lora_dropout": self.peft_lora_dropout,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "num_epochs": self.num_epochs,
                "max_length": self.max_length,
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
        translator.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Check if it's a PEFT model
        peft_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(peft_config_path):
            # Load base model with quantization if requested
            if translator.use_4bit:
                quantization_config = {
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": torch.float16,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": True
                }
                
                base_model = AutoModelForCausalLM.from_pretrained(
                    translator.model_name,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
            else:
                base_model = AutoModelForCausalLM.from_pretrained(translator.model_name)
                base_model.to(translator.device)
            
            # Load PEFT model
            translator.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            # Load regular model
            translator.model = AutoModelForCausalLM.from_pretrained(model_path)
            translator.model.to(translator.device)
        
        # Load training history if available
        history_path = f"{model_path}/training_history.json"
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                translator.history = json.load(f)
        
        return translator