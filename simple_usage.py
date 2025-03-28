from alf_t5 import ALFT5Translator

# Path to your saved model
model_path = "alf_t5_translator/final_model"  # or "alf_t5_translator/best_model"

print(f"Loading model from {model_path}...")

# Load the trained model
translator = ALFT5Translator.load(model_path)
print("Model loaded successfully!")

# Example translations
examples = [
    # Language to English
    ("Ith goeth", "t2b"),
    ("Thouth goeth", "t2b"),
    ("Heth eath", "t2b"),
    ("Sheth eath", "t2b"),
    
    # English to Language
    ("I go", "b2t"),
    ("you go", "b2t"),
    ("he eats", "b2t"),
    ("she eats", "b2t")
]

print("\nPerforming translations:")
print("-" * 50)

for text, direction in examples:
    # Translate the text
    translation = translator.translate(
        text=text,
        direction=direction,
        num_beams=5,          # Beam search for better quality
        temperature=1.0,      # 1.0 = no temperature (use model logits directly)
        do_sample=False       # Set to True for more creative translations
    )
    
    # Print the result
    src_lang = "Language" if direction == "t2b" else "English"
    tgt_lang = "English" if direction == "t2b" else "Language"
    print(f"{src_lang}: {text}")
    print(f"{tgt_lang}: {translation}")
    print("-" * 50)

# Interactive mode
print("\nInteractive Translation Mode (type 'quit' to exit)")
print("Format: text | direction (c2e or e2c)")

while True:
    user_input = input("\nEnter text to translate: ")
    if user_input.lower() == 'quit':
        break
        
    parts = user_input.split('|')
    if len(parts) != 2:
        print("Invalid format. Please use: text|direction (c2e or e2c)")
        continue
        
    text = parts[0].strip()
    direction = parts[1].strip()
    
    if direction not in ["t2b", "b2t"]:
        print("Invalid direction. Use 'c2e' for language to English or 'e2c' for English to language")
        continue
        
    try:
        translation = translator.translate(
            text=text,
            direction=direction,
            num_beams=5,
            temperature=1.0,
            do_sample=False
        )
        
        src_lang = "Language" if direction == "t2b" else "English"
        tgt_lang = "English" if direction == "t2b" else "Language"
        print(f"{src_lang}: {text}")
        print(f"{tgt_lang}: {translation}")
    except Exception as e:
        print(f"Error during translation: {e}")
