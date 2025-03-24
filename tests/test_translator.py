import unittest
import sys
import os
import shutil
import tempfile
from typing import List, Dict, Tuple

# Add parent directory to path to import alf_t5 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alf_t5 import (
    ALFT5Translator,
    parse_language_data,
    augment_data
)

class TestALFT5Translator(unittest.TestCase):
    """Test cases for ALFT5Translator functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        
        self.sample_data = """hello|hi
goodbye|bye
good morning|morning
thank you|thanks"""
        
        self.parsed_data = parse_language_data(self.sample_data)
    
    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_parse_language_data(self):
        """Test parsing of language data"""
        parsed = parse_language_data(self.sample_data)
        
        self.assertEqual(len(parsed), 4)
        
        self.assertEqual(parsed[0], ("hello", "hi"))
        self.assertEqual(parsed[1], ("goodbye", "bye"))
        self.assertEqual(parsed[2], ("good morning", "morning"))
        self.assertEqual(parsed[3], ("thank you", "thanks"))
    
    def test_augment_data(self):
        """Test data augmentation"""
        augmented = augment_data(self.parsed_data)
        
        self.assertGreater(len(augmented), len(self.parsed_data))
        
        for pair in self.parsed_data:
            self.assertIn(pair, augmented)
        
        self.assertIn(("Hello", "Hi"), augmented)
        
        self.assertIn(("morning good", "morning"), augmented)
    
    def test_translator_initialization(self):
        """Test translator initialization"""
        translator = ALFT5Translator(
            model_name="t5-small",
            output_dir=self.test_dir
        )
        
        self.assertEqual(translator.model_name, "t5-small")
        self.assertEqual(translator.output_dir, self.test_dir)
        self.assertTrue(translator.use_peft)  # Default should be True
    
    def test_model_architecture_config(self):
        """Test model architecture configuration"""
        configs = [
            {"model_name": "t5-small", "use_peft": True, "peft_r": 8},
            {"model_name": "t5-small", "use_peft": False},
            {"model_name": "t5-small", "peft_r": 16, "peft_lora_alpha": 32}
        ]
        
        for config in configs:
            translator = ALFT5Translator(
                output_dir=self.test_dir,
                **config
            )
            
            for key, value in config.items():
                self.assertEqual(getattr(translator, key), value)
    
    @unittest.skipIf("CI" in os.environ, "Skipping in CI environment")
    def test_small_training_session(self):
        """Test a small training session"""
        translator = ALFT5Translator(
            model_name="t5-small",
            output_dir=self.test_dir,
            num_epochs=1,  # Just one epoch for testing
            batch_size=2,
            eval_bleu=True,
            eval_meteor=True
        )
        
        translator.train(
            data_string=self.sample_data,
            test_size=0.5,
            augment=True
        )
        
        self.assertIn("train_loss", translator.history)
        self.assertIn("val_loss", translator.history)
        self.assertGreater(len(translator.history["train_loss"]), 0)
        
        save_path = os.path.join(self.test_dir, "test_model")
        translator.save(save_path)
        self.assertTrue(os.path.exists(save_path))
        
        translation = translator.translate("hello", direction="c2e")
        self.assertIsInstance(translation, str)
        
        translation, confidence = translator.translate(
            "hello", 
            direction="c2e",
            return_confidence=True
        )
        self.assertIsInstance(translation, str)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)


if __name__ == "__main__":
    unittest.main() 