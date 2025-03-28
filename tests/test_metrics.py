import unittest
import sys
import os
from typing import List

# Add parent directory to path to import alf_t5 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alf_t5.evaluation import (
    evaluate_bleu,
    evaluate_meteor,
    interpret_bleu_score,
    interpret_meteor_score
)

class TestMetrics(unittest.TestCase):
    """Test cases for translation metrics evaluation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.perfect_reference = [["this", "is", "a", "test"]]
        self.perfect_hypothesis = ["this", "is", "a", "test"]
        
        self.partial_reference = [["this", "is", "a", "good", "test"]]
        self.partial_hypothesis = ["this", "is", "a", "great", "test"]
        
        self.no_match_reference = [["this", "is", "a", "test"]]
        self.no_match_hypothesis = ["completely", "different", "words", "here"]
        
        self.multi_reference = [
            ["this", "is", "a", "test"],
            ["this", "is", "a", "trial"],
            ["this", "is", "an", "experiment"]
        ]
        
        self.references_list = [
            [["this", "is", "a", "test"]],
            [["the", "cat", "is", "black"]],
            [["i", "like", "to", "read", "books"]]
        ]
        
        self.hypotheses_list = [
            ["this", "is", "a", "test"],
            ["the", "cat", "is", "white"],
            ["i", "enjoy", "reading", "books"]
        ]
    
    def test_bleu_perfect_match(self):
        """Test BLEU score calculation with perfect match"""
        result = evaluate_bleu([self.perfect_hypothesis], [self.perfect_reference])
        self.assertEqual(result["corpus_bleu"], 1.0)
    
    def test_bleu_partial_match(self):
        """Test BLEU score calculation with partial match"""
        result = evaluate_bleu([self.partial_hypothesis], [self.partial_reference])
        self.assertGreater(result["corpus_bleu"], 0.0)
        self.assertLess(result["corpus_bleu"], 1.0)
    
    def test_bleu_no_match(self):
        """Test BLEU score calculation with no match"""
        result = evaluate_bleu([self.no_match_hypothesis], [self.no_match_reference])
        self.assertLess(result["corpus_bleu"], 0.1)
    
    def test_bleu_corpus_level(self):
        """Test corpus-level BLEU calculation"""
        result = evaluate_bleu(self.hypotheses_list, self.references_list)
        self.assertIn("corpus_bleu", result)
        self.assertIn("sentence_bleu", result)
        self.assertEqual(len(result["sentence_bleu"]), len(self.hypotheses_list))
    
    def test_meteor_perfect_match(self):
        """Test METEOR score calculation with perfect match"""
        result = evaluate_meteor([self.perfect_reference], [self.perfect_hypothesis])
        self.assertEqual(result["corpus_meteor"], 1.0)
    
    def test_meteor_partial_match(self):
        """Test METEOR score calculation with partial match"""
        result = evaluate_meteor([self.partial_reference], [self.partial_hypothesis])
        self.assertGreater(result["corpus_meteor"], 0.0)
        self.assertLess(result["corpus_meteor"], 1.0)
    
    def test_meteor_no_match(self):
        """Test METEOR score calculation with no match"""
        result = evaluate_meteor([self.no_match_reference], [self.no_match_hypothesis])
        self.assertLess(result["corpus_meteor"], 0.1)
    
    def test_meteor_corpus_level(self):
        """Test corpus-level METEOR calculation"""
        result = evaluate_meteor(self.references_list, self.hypotheses_list)
        self.assertIn("corpus_meteor", result)
        self.assertIn("sentence_meteor", result)
        self.assertEqual(len(result["sentence_meteor"]), len(self.hypotheses_list))
    
    def test_interpret_bleu_score(self):
        """Test BLEU score interpretation"""
        perfect = interpret_bleu_score(1.0)
        good = interpret_bleu_score(0.42)
        moderate = interpret_bleu_score(0.25)
        poor = interpret_bleu_score(0.05)
        
        self.assertEqual(perfect["quality"], "Excellent")
        self.assertEqual(good["quality"], "Good")
        self.assertEqual(moderate["quality"], "Moderate")
        self.assertEqual(poor["quality"], "Poor")
        
        small_dataset = interpret_bleu_score(0.30, 40)
        medium_dataset = interpret_bleu_score(0.30, 150)
        large_dataset = interpret_bleu_score(0.30, 500)
        
        self.assertIn("small dataset of 40", small_dataset["context"])
        self.assertIn("medium dataset of 150", medium_dataset["context"])
        self.assertIn("large dataset of 500", large_dataset["context"])
    
    def test_interpret_meteor_score(self):
        """Test METEOR score interpretation"""
        perfect = interpret_meteor_score(1.0)
        good = interpret_meteor_score(0.45)
        moderate = interpret_meteor_score(0.35)
        poor = interpret_meteor_score(0.15)
        
        self.assertEqual(perfect["quality"], "Excellent")
        self.assertEqual(good["quality"], "Good")
        self.assertEqual(moderate["quality"], "Moderate")
        self.assertEqual(poor["quality"], "Poor")
        
        small_dataset = interpret_meteor_score(0.30, 40)
        medium_dataset = interpret_meteor_score(0.30, 150)
        large_dataset = interpret_meteor_score(0.30, 500)
        
        self.assertIn("small dataset of 40", small_dataset["context"])
        self.assertIn("medium dataset of 150", medium_dataset["context"])
        self.assertIn("large dataset of 500", large_dataset["context"])


if __name__ == "__main__":
    unittest.main() 