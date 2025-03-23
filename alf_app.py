import torch
import argparse
import json
from pathlib import Path
from alf_t5 import ALFT5Translator

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
    
    def translate_text(self, text, direction="c2e", **kwargs):
        """Translate a single text."""
        # Merge default parameters with any provided kwargs
        params = {**self.default_params, **kwargs}
        
        try:
            translation = self.translator.translate(
                text=text,
                direction=direction,
                **params
            )
            return translation
        except Exception as e:
            print(f"Error translating '{text}': {e}")
            return f"[Translation error: {str(e)}]"
    
    def batch_translate(self, texts, directions, **kwargs):
        """Translate a batch of texts."""
        results = []
        
        for text, direction in zip(texts, directions):
            translation = self.translate_text(text, direction, **kwargs)
            results.append({
                "source": text,
                "direction": direction,
                "translation": translation
            })
        
        return results
    
    def translate_file(self, input_file, output_file, direction="c2e", **kwargs):
        """Translate texts from a file and save results to another file."""
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # Translate each line
        translations = []
        for line in lines:
            translation = self.translate_text(line, direction, **kwargs)
            translations.append(translation)
        
        # Write to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            for original, translation in zip(lines, translations):
                f.write(f"Source: {original}\n")
                f.write(f"Translation: {translation}\n\n")
        
        print(f"Translated {len(lines)} lines from {input_file} to {output_file}")
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
                print("  quit - Exit interactive mode")
                continue
            
            if user_input.lower() == 'params':
                print("\nCurrent translation parameters:")
                for param, value in self.default_params.items():
                    print(f"  {param}: {value}")
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
                
                if direction not in ["c2e", "e2c"]:
                    print("Invalid direction. Use 'c2e' for conlang to English or 'e2c' for English to conlang")
                    continue
                
                try:
                    translation = self.translate_text(text, direction)
                    src_lang = "Conlang" if direction == "c2e" else "English"
                    tgt_lang = "English" if direction == "c2e" else "Conlang"
                    print(f"{src_lang}: {text}")
                    print(f"{tgt_lang}: {translation}")
                except Exception as e:
                    print(f"Error during translation: {e}")
            else:
                print("Invalid format. Please use: text | direction (c2e or e2c)")


def main():
    parser = argparse.ArgumentParser(description="ALF-1")
    parser.add_argument("--model", type=str, default="alf_t5_translator/final_model",
                        help="Path to the trained model directory")
    parser.add_argument("--mode", type=str, choices=["interactive", "file", "batch"], 
                        default="interactive", help="Operation mode")
    parser.add_argument("--input", type=str, help="Input file for file mode")
    parser.add_argument("--output", type=str, help="Output file for file mode")
    parser.add_argument("--direction", type=str, choices=["c2e", "e2c"], 
                        default="c2e", help="Translation direction")
    
    args = parser.parse_args()
    
    # Create the translator app
    app = ALFTranslatorApp(args.model)
    
    # Run in the specified mode
    if args.mode == "interactive":
        app.interactive_mode()
    elif args.mode == "file":
        if not args.input or not args.output:
            print("Error: --input and --output are required for file mode")
            return
        app.translate_file(args.input, args.output, args.direction)
    elif args.mode == "batch":
        # Example batch translation
        texts = ["Ith eath", "Thou eath", "Heth eath", "Sheth eath"]
        directions = ["c2e"] * len(texts)
        results = app.batch_translate(texts, directions)
        
        for result in results:
            print(f"Source: {result['source']}")
            print(f"Translation: {result['translation']}")
            print()


if __name__ == "__main__":
    main()