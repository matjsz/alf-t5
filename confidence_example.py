from alf_t5 import ALFT5Translator

def main():
    model_path = "alf_t5_translator/final_model"  # or "alf_t5_translator/best_model"
    
    print(f"Loading model from {model_path}...")
    
    translator = ALFT5Translator.load(model_path)
    print("Model loaded successfully!")
    
    examples = [
        ("Ith goeth", "t2b"),
        ("Thouth goeth", "t2b"),
        ("Heth eath", "t2b"),
        ("Sheth eath", "t2b"),
        
        ("I go", "b2t"),
        ("you go", "b2t"),
        ("he eats", "b2t"),
        ("she eats", "b2t")
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
        
        src_lang = "Language" if direction == "t2b" else "English"
        tgt_lang = "English" if direction == "t2b" else "Language"
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