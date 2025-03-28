import argparse
import json
from alf_t5.translator import ALFT5Translator

class ALFTranslatorApp:
    def __init__(self, model_path):
        """Initialize the translator app with a trained model."""
        self.model_path = model_path
        self.translator = self._load_model(model_path)
        
        # Default generation parameters 
        self.default_params = {
            "num_beams": 5,
            "temperature": 1.0,
            "top_p": None,
            "do_sample": False,
            "max_length": 128
        }
    
    def _load_model(self, model_path):
        """Load the trained model from the specified path."""
        try:
            translator = ALFT5Translator.load(model_path)
            print(f"Model loaded successfully from {model_path}")
            return translator
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def translate_text(self, text, direction="t2b", include_confidence=False, **kwargs):
        """Translate a single text."""
        params = {**self.default_params, **kwargs}
        
        try:
            result = self.translator.translate(
                text=text,
                direction=direction,
                return_confidence=include_confidence,
                **params
            )
            
            if include_confidence:
                translation, confidence = result
                return {
                    "text": text,
                    "translation": translation,
                    "confidence": confidence,
                    "direction": direction
                }
            else:
                return result
        except Exception as e:
            print(f"Error translating '{text}': {e}")
            return f"[Translation error: {str(e)}]"
    
    def batch_translate(self, texts, directions, include_confidence=False, **kwargs):
        """Translate a batch of texts."""
        results = []
        
        for text, direction in zip(texts, directions):
            result = self.translate_text(text, direction, include_confidence=include_confidence, **kwargs)
            
            if include_confidence:
                results.append(result)
            else:
                results.append({
                    "text": text,
                    "translation": result,
                    "direction": direction
                })
        
        return results
    
    def translate_file(self, input_file, output_file, direction="t2b", include_confidence=False, **kwargs):
        """Translate texts from a file and save results to another file."""
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        translations = []
        confidences = []
        for line in lines:
            result = self.translate_text(line, direction, include_confidence=include_confidence, **kwargs)
            
            if include_confidence:
                translations.append(result["translation"])
                confidences.append(result["confidence"])
            else:
                translations.append(result)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, (original, translation) in enumerate(zip(lines, translations)):
                f.write(f"Source: {original}\n")
                f.write(f"Translation: {translation}\n")
                if include_confidence:
                    f.write(f"Confidence: {confidences[i]:.4f}\n")
                f.write("\n")
        
        print(f"Translated {len(lines)} lines from {input_file} to {output_file}")
        
        if include_confidence:
            return list(zip(translations, confidences))
        return translations
    
    def save_dictionary(self, output_file):
        """Extract and save a dictionary of known translations."""
        # This is a simplified approach - in a real implementation,
        # you would extract this from your training data
        examples = [
            ("Ith", "I"),
            ("thou", "you"),
            ("sheth", "he/she"),
            ("wyth", "we"),
        ]
        
        dictionary = {
            "conlang_to_english": {conlang: english for conlang, english in examples},
            "english_to_conlang": {english: conlang for conlang, english in examples}
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dictionary, f, indent=2, ensure_ascii=False)
        
        print(f"Dictionary saved to {output_file}")
        return dictionary
    
    def interactive_mode(self):
        """Run an interactive translation session."""
        print("\nALF Interactive Mode")
        print("Type 'quit' to exit, 'help' for commands")
        
        # Add confidence to default parameters
        self.show_confidence = False
        
        while True:
            user_input = input("\n> ").strip()
            
            if user_input.lower() == 'quit':
                print("Exiting interactive mode")
                break
            
            if user_input.lower() == 'help':
                print("\nCommands:")
                print("  text | direction - Translate text (direction: c2e or e2c)")
                print("  params - Show current translation parameters")
                print("  set param value - Set a translation parameter")
                print("  confidence on/off - Toggle confidence score display")
                print("  quit - Exit interactive mode")
                continue
            
            if user_input.lower() == 'params':
                print("\nCurrent translation parameters:")
                for param, value in self.default_params.items():
                    print(f"  {param}: {value}")
                print(f"  confidence: {self.show_confidence}")
                continue
            
            if user_input.lower().startswith('confidence '):
                option = user_input.lower().split()[1]
                if option in ['on', 'true', 'yes', '1']:
                    self.show_confidence = True
                    print("Confidence scores enabled")
                elif option in ['off', 'false', 'no', '0']:
                    self.show_confidence = False
                    print("Confidence scores disabled")
                else:
                    print("Invalid option. Use 'confidence on' or 'confidence off'")
                continue
            
            if user_input.lower().startswith('set '):
                parts = user_input[4:].split()
                if len(parts) != 2:
                    print("Invalid format. Use: set param value")
                    continue
                
                param, value = parts
                if param not in self.default_params:
                    print(f"Unknown parameter: {param}")
                    continue
                
                try:
                    # Convert value to appropriate type
                    if param in ['num_beams', 'max_length']:
                        value = int(value)
                    elif param in ['temperature', 'top_p']:
                        value = float(value)
                    elif param == 'do_sample':
                        value = value.lower() in ['true', 'yes', '1']
                    
                    self.default_params[param] = value
                    print(f"Set {param} to {value}")
                except ValueError:
                    print(f"Invalid value for {param}: {value}")
                continue
            
            # Process translation request
            if '|' in user_input:
                parts = user_input.split('|')
                if len(parts) != 2:
                    print("Invalid format. Please use: text | direction (c2e or e2c)")
                    continue
                
                text = parts[0].strip()
                direction = parts[1].strip()
                
                if direction not in ["t2b", "b2t"]:
                    print("Invalid direction. Use 'c2e' for conlang to English or 'e2c' for English to conlang")
                    continue
                
                try:
                    result = self.translate_text(text, direction, include_confidence=self.show_confidence)
                    src_lang = "Conlang" if direction == "t2b" else "English"
                    tgt_lang = "English" if direction == "t2b" else "Conlang"
                    
                    print(f"{src_lang}: {text}")
                    
                    if self.show_confidence:
                        print(f"{tgt_lang}: {result['translation']}")
                        print(f"Confidence: {result['confidence']:.4f}")
                    else:
                        print(f"{tgt_lang}: {result}")
                except Exception as e:
                    print(f"Error during translation: {e}")
            else:
                print("Invalid format. Please use: text | direction (c2e or e2c)")


def run():
    parser = argparse.ArgumentParser(description="ALF-T5")
    parser.add_argument("--model", type=str, default="alf_t5_translator/final_model",
                        help="Path to the trained model directory")
    parser.add_argument("--mode", type=str, choices=["interactive", "file", "batch"], 
                        default="interactive", help="Operation mode")
    parser.add_argument("--input", type=str, help="Input file for file mode")
    parser.add_argument("--output", type=str, help="Output file for file mode")
    parser.add_argument("--direction", type=str, choices=["t2b", "b2t"], 
                        default="t2b", help="Translation direction")
    parser.add_argument("--confidence", action="store_true",
                        help="Include confidence scores in the output")
    
    args = parser.parse_args()

    try:
        # Create the translator app
        app = ALFTranslatorApp(args.model)
        
        # Run in the specified mode
        if args.mode == "interactive":
            app.interactive_mode()
        elif args.mode == "file":
            if not args.input or not args.output:
                print("Error: --input and --output are required for file mode")
                return
            app.translate_file(args.input, args.output, args.direction, include_confidence=args.confidence)
        elif args.mode == "batch":
            # Example batch translation
            texts = ["Ith eath", "Thou eath", "Heth eath", "Sheth eath"]
            directions = ["t2b"] * len(texts)
            results = app.batch_translate(texts, directions, include_confidence=args.confidence)
            
            for result in results:
                if args.confidence:
                    print(f"Text: {result['text']}")
                    print(f"Translation: {result['translation']}")
                    print(f"Confidence: {result['confidence']:.4f}")
                else:
                    print(f"Text: {result['text']}")
                    print(f"Translation: {result['translation']}")
                print()
    except:
        print("Error: it wasn't possible to start ALF-T5 app, did you train the model? Otherwise, check the --help command.")