import unittest
import sys
import os
import shutil
import tempfile
from typing import List, Dict, Tuple

# Add parent directory to path to import alf_t5 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alf_t5 import (
    ModelExperiment,
    parse_language_data
)

class TestModelExperimentation(unittest.TestCase):
    """Test cases for model experimentation functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        
        self.sample_data = """hello|hi
goodbye|bye
good morning|morning
thank you|thanks"""
    
    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_experiment_initialization(self):
        """Test experiment initialization"""
        experiment = ModelExperiment(
            base_output_dir=self.test_dir,
            data_string=self.sample_data,
            test_size=0.2,
            augment=True
        )
        
        self.assertEqual(experiment.base_output_dir, self.test_dir)
        self.assertEqual(experiment.data_string, self.sample_data)
        self.assertEqual(experiment.test_size, 0.2)
        self.assertTrue(experiment.augment)
        self.assertEqual(experiment.experiments, [])
        self.assertEqual(experiment.results, {})
    
    def test_add_experiment(self):
        """Test adding a single experiment"""
        experiment = ModelExperiment(
            base_output_dir=self.test_dir,
            data_string=self.sample_data
        )
        
        experiment.add_experiment(
            name="test_experiment",
            model_name="t5-small",
            num_epochs=1,
            batch_size=2
        )
        
        self.assertEqual(len(experiment.experiments), 1)
        self.assertEqual(experiment.experiments[0]["name"], "test_experiment")
        self.assertEqual(experiment.experiments[0]["model_name"], "t5-small")
        self.assertEqual(experiment.experiments[0]["model_params"]["num_epochs"], 1)
        self.assertEqual(experiment.experiments[0]["model_params"]["batch_size"], 2)
    
    def test_add_grid_search(self):
        """Test adding experiments via grid search"""
        experiment = ModelExperiment(
            base_output_dir=self.test_dir,
            data_string=self.sample_data
        )
        
        experiment.add_grid_search(
            name_prefix="grid_experiment",
            model_names=["t5-small"],
            num_epochs=[1, 2],
            batch_size=[2, 4],
            learning_rate=[1e-4, 3e-4]
        )
        
        self.assertEqual(len(experiment.experiments), 8)
        
        self.assertTrue(any("grid_experiment_t5-small_num_epochs=1_batch_size=2_learning_rate=0.0001" == exp["name"] for exp in experiment.experiments))
        
        param_combinations = []
        for exp in experiment.experiments:
            params = exp["model_params"]
            combo = (params["num_epochs"], params["batch_size"], params["learning_rate"])
            param_combinations.append(combo)
        
        expected_combinations = [
            (1, 2, 1e-4), (1, 2, 3e-4),
            (1, 4, 1e-4), (1, 4, 3e-4),
            (2, 2, 1e-4), (2, 2, 3e-4),
            (2, 4, 1e-4), (2, 4, 3e-4)
        ]
        
        for combo in expected_combinations:
            self.assertIn(combo, param_combinations)
    
    @unittest.skipIf("CI" in os.environ, "Skipping in CI environment")
    def test_run_single_experiment(self):
        """Test running a single experiment"""
        experiment = ModelExperiment(
            base_output_dir=self.test_dir,
            data_string=self.sample_data
        )
        
        experiment.add_experiment(
            name="mini_experiment",
            model_name="t5-small",
            num_epochs=1,  # Just one epoch for testing
            batch_size=2
        )
        
        results_df = experiment.run_experiments()
        
        self.assertIn("mini_experiment", experiment.results)
        self.assertTrue(experiment.results["mini_experiment"]["successful"])
        
        # Check output files
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "experiment_results.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "experiment_results.json")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "mini_experiment", "experiment_results.json")))
    
    def test_get_best_experiment(self):
        """Test getting best experiment from results"""
        experiment = ModelExperiment(
            base_output_dir=self.test_dir,
            data_string=self.sample_data
        )
        
        # Mock results
        experiment.results = {
            "exp1": {
                "name": "exp1",
                "val_loss": 1.5,
                "bleu_score": 0.3,
                "meteor_score": 0.4,
                "successful": True
            },
            "exp2": {
                "name": "exp2",
                "val_loss": 1.2,
                "bleu_score": 0.4,
                "meteor_score": 0.5,
                "successful": True
            },
            "exp3": {
                "name": "exp3",
                "val_loss": 1.8,
                "bleu_score": 0.2,
                "meteor_score": 0.3,
                "successful": True
            },
            "exp_failed": {
                "name": "exp_failed",
                "error": "Some error",
                "successful": False
            }
        }
        
        # Test with val_loss (lower is better)
        best = experiment.get_best_experiment(metric="val_loss", higher_is_better=False)
        self.assertEqual(best["name"], "exp2")
        
        # Test with bleu_score (higher is better)
        best = experiment.get_best_experiment(metric="bleu_score", higher_is_better=True)
        self.assertEqual(best["name"], "exp2")
        
        # Test with meteor_score (higher is better)
        best = experiment.get_best_experiment(metric="meteor_score", higher_is_better=True)
        self.assertEqual(best["name"], "exp2")
    
    @unittest.skipIf("CI" in os.environ, "Skipping in CI environment")
    def test_plot_experiment_results(self):
        """Test plotting experiment results"""
        experiment = ModelExperiment(
            base_output_dir=self.test_dir,
            data_string=self.sample_data
        )
        
        experiment.results = {
            "exp1": {
                "name": "exp1",
                "val_loss": 1.5,
                "bleu_score": 0.3,
                "meteor_score": 0.4,
                "training_time": 60,
                "params": {"learning_rate": 1e-4, "batch_size": 4},
                "successful": True
            },
            "exp2": {
                "name": "exp2",
                "val_loss": 1.2,
                "bleu_score": 0.4,
                "meteor_score": 0.5,
                "training_time": 90,
                "params": {"learning_rate": 3e-4, "batch_size": 8},
                "successful": True
            }
        }
        
        df = experiment.plot_experiment_results()
        
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "val_loss_comparison.png")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "bleu_score_comparison.png")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "meteor_score_comparison.png")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "training_time_comparison.png")))
        
        # Check DataFrame
        self.assertEqual(len(df), 2)
        self.assertIn("learning_rate", df.columns)
        self.assertIn("batch_size", df.columns)


if __name__ == "__main__":
    unittest.main() 