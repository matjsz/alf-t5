<div align="center">
    <h1>CoAL-1</h1>
    <img src="docs/CoAL-1 Logo.png" width="100rem" />
    <p>Conlang Auto Learner</p>
    <div style="display: flex; justify-content: center; gap: .5rem;">
        <img src="https://img.shields.io/badge/3.12-blue?style=plastic&label=python" />
        <img src="https://img.shields.io/badge/2.6-orange?style=plastic&label=torch" />
        <img src="https://img.shields.io/badge/1.0-green?style=plastic&label=stable" />
    </div>
</div>

---

A neural machine translation system that automatically learns and translates constructed languages (conlangs) based on a small set of translation examples. This project uses transfer learning with pre-trained language models to achieve high-quality translations with minimal training data.

## Features

- **Automatic Language Learning**: Learns conlang structure, vocabulary, and grammar from translation pairs
- **Bidirectional Translation**: Translates from conlang to English and English to conlang
- **Few-Shot Learning**: Requires only a small dataset of translation examples
- **Parameter-Efficient Fine-Tuning**: Uses LoRA to fine-tune only a small subset of model parameters
- **Data Augmentation**: Automatically expands limited training data
- **Interactive Mode**: Command-line interface for real-time translations
- **Batch Processing**: Support for translating multiple texts or files

## Installation

### Start locally

```
# Clone the repository
git clone https://github.com/matjsz/coal.git
cd coal

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch transformers peft tqdm matplotlib
```

## Quick Start

### Training a New Translator

```python
from coal_t5 import CoALT5Translator, extract_dataset

# Example conlang data
# .txt, .csv
data = extract_dataset("my_data.txt")
# Create and train translator
translator = CoALT5Translator(
    model_name="t5-small",
    use_peft=True,
    batch_size=4,
    num_epochs=20,
    output_dir="coal_t5_translator"
)

# Train the model
translator.train(
    data_string=data,
    test_size=0.2,
    augment=True
)
```

### Testing The Translator

```python
english = translator.translate("thou drinkth waterth", direction="c2e")
conlang = translator.translate("you drink water", direction="e2c")

print(f"English: {english}")
print(f"Conlang: {conlang}")
```

### Using a Trained Translator

```python
from coal import CoALT5Translator

# Load a trained model
translator = CoALT5Translator.load("coal_t5_translator/final_model")

# Translate from conlang to English
english = translator.translate("thou eath thy appleth", direction="c2e")
print(f"English: {english}")

# Translate from English to conlang
conlang = translator.translate("I see you", direction="e2c")
print(f"Conlang: {conlang}")
```

### Interactive Mode

```shellscript
python coal_app.py --model coal_t5_translator/final_model --mode interactive
```

## Data Format

The translator accepts conlang data in the following format:

```plaintext
conlang_text|english_translation
```

Each line contains a pair of conlang text and its English translation, separated by a pipe character (`|`).

## Technical Details

### Architecture

**CoAL-1** uses the **T5** (Text-to-Text Transfer Transformer) model as the foundation for the translation system. T5 is an encoder-decoder model specifically designed for sequence-to-sequence tasks like translation.

Key components:

- **Base Model**: T5-small (60M parameters) by default, but can be configured to use larger variants
- **Fine-Tuning**: Parameter-Efficient Fine-Tuning (PEFT) with LoRA (Low-Rank Adaptation)
- **Tokenization**: Uses T5's subword tokenizer
- **Training**: Bidirectional training (conlang→English and English→conlang)
- **Generation**: Beam search with configurable parameters


### Why T5 over Causal Language Models?

For conlang translation, T5's encoder-decoder architecture offers several advantages:

1. **Bidirectional Context**: The encoder processes the entire source sentence bidirectionally
2. **Parameter Efficiency**: More efficient for translation tasks than causal LMs
3. **Training Stability**: More stable during fine-tuning for translation tasks
4. **Resource Requirements**: Requires less computational resources


### Data Augmentation

To maximize learning from limited examples, the system employs several data augmentation techniques:

1. **Case Variations**: Adding capitalized versions of examples
2. **Word Order Variations**: Adding reversed word order for multi-word phrases
3. **Vocabulary Recombination**: Creating new combinations from existing vocabulary mappings

Thought for 3 seconds### Conlang Translator

## Project Structure

```plaintext
conlang-translator/
├── coal_t5.py     # Main implementation with T5
├── coal_llama.py  # Alternative implementation with Llama
├── simple_usage.py         # Simple script to use a trained model
├── coal_app.py    # Advanced application with multiple modes
├── README.md                    # This file
└── coal_t5_translator/       # Directory for trained models
    ├── final_model/             # Saved model after training
    ├── best_model/              # Best model based on validation loss
    └── training_history.png     # Plot of training metrics
```

## Training Process

The training process involves:

1. **Data Parsing**: Converting the input format into training pairs
2. **Data Augmentation**: Expanding the training data
3. **Tokenization**: Converting text to token IDs
4. **Model Initialization**: Loading the pre-trained T5 model
5. **PEFT Setup**: Configuring LoRA for parameter-efficient fine-tuning
6. **Training Loop**: Fine-tuning with early stopping based on validation loss
7. **Model Saving**: Saving checkpoints and the final model

### Training Parameters

Key parameters that can be configured:

- `model_name`: Base pre-trained model (default: "t5-small")
- `use_peft`: Whether to use parameter-efficient fine-tuning (default: True)
- `peft_r`: LoRA rank (default: 8)
- `batch_size`: Batch size for training (default: 8)
- `lr`: Learning rate (default: 5e-4)
- `num_epochs`: Maximum number of training epochs (default: 20)
- `max_length`: Maximum sequence length (default: 128)


## Translation Parameters

When translating text, you can configure:

- `direction`: Translation direction ("c2e" or "e2c")
- `num_beams`: Number of beams for beam search (default: 5)
- `temperature`: Sampling temperature (default: 1.0)
- `top_p`: Nucleus sampling parameter (default: None)
- `do_sample`: Whether to use sampling instead of greedy decoding (default: False)

Thought for 3 seconds### Conlang Translator

A neural machine translation system that automatically learns and translates constructed languages (conlangs) based on a small set of translation examples. This project uses transfer learning with pre-trained language models to achieve high-quality translations with minimal training data.

## Advanced Usage

### Batch Translation

```python
texts = ["thou eath", "thou walkth toth", "Ith eath thy appleth"]
directions = ["c2e", "c2e", "c2e"]

app = ConlangTranslatorApp("coal_t5_translator/final_model")
results = app.batch_translate(texts, directions)

for result in results:
    print(f"Source: {result['source']}")
    print(f"Translation: {result['translation']}")
```

### File Translation

```shellscript
python coal_app.py --model coal_t5_translator/final_model --mode file --input input.txt --output output.txt --direction c2e
```

## Performance Considerations

- **GPU Acceleration**: Training and inference are significantly faster with a GPU
- **Model Size**: Larger T5 models (t5-base, t5-large) may provide better results but require more resources
- **Training Data**: More diverse examples generally lead to better generalization
- **Hyperparameters**: Adjust batch size, learning rate, and LoRA parameters based on your dataset size

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research or work, please cite:

```plaintext
@software{coal,
  author = {Matheus J.G. Silva},
  title = {CoAL-1: Neural Machine Translation for Constructed Languages},
  year = {2025},
  url = {https://github.com/matjsz/coal}
}
```

## Acknowledgments

- This project uses Hugging Face's Transformers library
- The PEFT implementation is based on the PEFT library
- Special thanks to the T5 and LoRA authors for their research