from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any

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