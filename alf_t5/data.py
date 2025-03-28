from typing import List, Tuple

import nltk
import numpy as np
from typing import List, Tuple

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
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