from alf_t5 import ALFT5Translator

def main():
    model_path = "alf_t5_translator/final_model"  # or "alf_t5_translator/best_model"
    
    print(f"Loading model from {model_path}...")
    
    translator = ALFT5Translator.load(model_path)
    print("Model loaded successfully!")
    
    examples = [
        ("Ith goeth", "c2e"),
        ("Thouth goeth", "c2e"),
        ("Heth eath", "c2e"),
        ("Sheth eath", "c2e"),
        
        ("I go", "e2c"),
        ("you go", "e2c"),
        ("he eats", "e2c"),
        ("she eats", "e2c")
    ]
    
    print("\nPerforming translations with confidence scores:")
    print("-" * 60)
    
    for text, direction in examples:
        translation, confidence = translator.translate(
            text=text,
            direction=direction,
            num_beams=5,
            temperature=1.0,
            do_sample=False,
            return_confidence=True
        )
        
        src_lang = "Language" if direction == "c2e" else "English"
        tgt_lang = "English" if direction == "c2e" else "Language"
        print(f"{src_lang}: {text}")
        print(f"{tgt_lang}: {translation}")
        print(f"Confidence: {confidence:.4f}")
        
        if confidence > 0.9:
            interpretation = "Very high confidence"
        elif confidence > 0.75:
            interpretation = "High confidence"
        elif confidence > 0.5:
            interpretation = "Moderate confidence"
        elif confidence > 0.3:
            interpretation = "Low confidence"
        else:
            interpretation = "Very low confidence"
            
        print(f"Interpretation: {interpretation}")
        print("-" * 60)

if __name__ == "__main__":
    main() 