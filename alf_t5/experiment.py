import os
import json
import time
import matplotlib.pyplot as plt
import itertools
import concurrent.futures
import pandas as pd

import nltk
import matplotlib.pyplot as plt
from typing import List

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

from alf_t5.translator import ALFT5Translator

class ModelExperiment:
    """Class for experimenting with different model architectures and hyperparameters"""
    
    def __init__(
        self,
        base_output_dir: str = "alf_t5_experiments",
        data_string: str = None,
        data_file: str = None,
        test_size: float = 0.2,
        augment: bool = True,
        metrics: List[str] = ["bleu", "meteor"],
        parallel_experiments: int = 1,
        experiment_timeout: int = 7200  # 2 hours default timeout per experiment
    ):
        """
        Initialize the model experiment framework
        
        Args:
            base_output_dir: Base directory to store experiment outputs
            data_string: String containing the training data
            data_file: File containing the training data (alternative to data_string)
            test_size: Proportion of data to use for validation
            augment: Whether to augment the training data
            metrics: List of metrics to evaluate models on
            parallel_experiments: Number of experiments to run in parallel (use with caution)
            experiment_timeout: Maximum time allowed per experiment in seconds
        """
        self.base_output_dir = base_output_dir
        self.data_string = data_string
        self.data_file = data_file
        self.test_size = test_size
        self.augment = augment
        self.metrics = metrics
        self.parallel_experiments = parallel_experiments
        self.experiment_timeout = experiment_timeout
        
        # Load data if file is provided
        if data_file and not data_string:
            with open(data_file, 'r', encoding='utf-8') as f:
                self.data_string = f.read()
        
        # Store experiment results
        self.experiments = []
        self.results = {}
        
        # Create base directory
        os.makedirs(base_output_dir, exist_ok=True)
    
    def add_experiment(
        self,
        name: str,
        model_name: str = "t5-small",
        **model_params
    ):
        """
        Add an experiment configuration
        
        Args:
            name: Name of the experiment
            model_name: Name of the T5 model to use
            **model_params: Parameters to pass to ALFT5Translator
        """
        experiment = {
            "name": name,
            "model_name": model_name,
            "model_params": model_params
        }
        self.experiments.append(experiment)
        return self
    
    def add_grid_search(
        self,
        name_prefix: str,
        model_names: List[str] = ["t5-small"],
        **param_grid
    ):
        """
        Add multiple experiments via grid search over parameters
        
        Args:
            name_prefix: Prefix for experiment names
            model_names: List of model names to try
            **param_grid: Parameter grid with lists of values to try
        
        Example:
            add_grid_search(
                "peft_experiment",
                model_names=["t5-small", "t5-base"],
                peft_r=[4, 8, 16],
                learning_rate=[1e-4, 3e-4, 5e-4]
            )
        """
        # Generate all combinations of parameters
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for model_name in model_names:
            for i, values in enumerate(itertools.product(*param_values)):
                params = dict(zip(param_names, values))
                
                # Create experiment name
                param_str = "_".join([f"{k}={v}" for k, v in params.items()])
                name = f"{name_prefix}_{model_name}_{param_str}"
                
                # Add the experiment
                self.add_experiment(
                    name=name,
                    model_name=model_name,
                    **params
                )
        
        return self
    
    def _run_experiment(self, experiment):
        """
        Run a single experiment
        
        Args:
            experiment: Experiment configuration dictionary
        
        Returns:
            Dictionary with experiment results
        """
        name = experiment["name"]
        model_name = experiment["model_name"]
        model_params = experiment["model_params"]
        
        print(f"Running experiment: {name}")
        
        # Create output directory for this experiment
        output_dir = os.path.join(self.base_output_dir, name)
        os.makedirs(output_dir, exist_ok=True)
        
        start_time = time.time()
        
        try:
            # Initialize translator
            translator = ALFT5Translator(
                model_name=model_name,
                output_dir=output_dir,
                eval_bleu="bleu" in self.metrics,
                eval_meteor="meteor" in self.metrics,
                **model_params
            )
            
            # Train the model
            translator.train(
                data_string=self.data_string,
                test_size=self.test_size,
                augment=self.augment
            )
            
            end_time = time.time()
            training_time = end_time - start_time
            
            # Get the results
            train_loss = min(translator.history["train_loss"]) if translator.history["train_loss"] else float('inf')
            val_loss = min(translator.history["val_loss"]) if translator.history["val_loss"] else float('inf')
            
            bleu_score = max([item["corpus_bleu"] for item in translator.history["bleu_scores"]]) if "bleu_scores" in translator.history and translator.history["bleu_scores"] else 0.0
            
            meteor_score = max([item["corpus_meteor"] for item in translator.history["meteor_scores"]]) if "meteor_scores" in translator.history and translator.history["meteor_scores"] else 0.0
            
            # Save results
            results = {
                "name": name,
                "model_name": model_name,
                "params": model_params,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "bleu_score": bleu_score,
                "meteor_score": meteor_score,
                "training_time": training_time,
                "successful": True
            }
            
            # Save experiment config and results
            with open(os.path.join(output_dir, "experiment_results.json"), 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4)
            
            return results
            
        except Exception as e:
            end_time = time.time()
            training_time = end_time - start_time
            
            error_message = str(e)
            print(f"Error in experiment {name}: {error_message}")
            
            # Save error information
            results = {
                "name": name,
                "model_name": model_name,
                "params": model_params,
                "error": error_message,
                "training_time": training_time,
                "successful": False
            }
            
            # Save error information
            with open(os.path.join(output_dir, "experiment_error.json"), 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4)
            
            return results
    
    def run_experiments(self):
        """
        Run all configured experiments
        
        Returns:
            DataFrame with experiment results
        """
        if not self.experiments:
            raise ValueError("No experiments configured. Add experiments with add_experiment() or add_grid_search().")
        
        if not self.data_string:
            raise ValueError("No training data provided. Set data_string or data_file.")
        
        results = []
        
        if self.parallel_experiments > 1:
            # Run experiments in parallel
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.parallel_experiments) as executor:
                future_to_exp = {
                    executor.submit(self._run_experiment, exp): exp 
                    for exp in self.experiments
                }
                
                for future in concurrent.futures.as_completed(future_to_exp):
                    exp = future_to_exp[future]
                    try:
                        result = future.result(timeout=self.experiment_timeout)
                        results.append(result)
                    except concurrent.futures.TimeoutError:
                        print(f"Experiment {exp['name']} timed out after {self.experiment_timeout} seconds")
                        results.append({
                            "name": exp["name"],
                            "model_name": exp["model_name"],
                            "params": exp["model_params"],
                            "error": "Experiment timed out",
                            "training_time": self.experiment_timeout,
                            "successful": False
                        })
        else:
            # Run experiments sequentially
            for exp in self.experiments:
                result = self._run_experiment(exp)
                results.append(result)
        
        # Store results
        self.results = {result["name"]: result for result in results}
        
        # Create a DataFrame for analysis
        results_df = pd.DataFrame(results)
        
        # Save overall results
        results_df.to_csv(os.path.join(self.base_output_dir, "experiment_results.csv"), index=False)
        
        with open(os.path.join(self.base_output_dir, "experiment_results.json"), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        
        return results_df
    
    def get_best_experiment(self, metric="val_loss", higher_is_better=False):
        """
        Get the best experiment according to a metric
        
        Args:
            metric: Metric to use for comparison ('val_loss', 'bleu_score', 'meteor_score')
            higher_is_better: Whether higher values of the metric are better
            
        Returns:
            Dictionary with best experiment details
        """
        if not self.results:
            raise ValueError("No experiment results available. Run experiments first.")
        
        # Filter experiments that completed successfully
        successful_experiments = [exp for exp in self.results.values() if exp["successful"]]
        
        if not successful_experiments:
            raise ValueError("No successful experiments to compare.")
        
        # Sort by metric
        if higher_is_better:
            best_exp = max(successful_experiments, key=lambda x: x.get(metric, float('-inf')))
        else:
            best_exp = min(successful_experiments, key=lambda x: x.get(metric, float('inf')))
            
        return best_exp
    
    def plot_experiment_results(self, metrics=None, sort_by=None, ascending=True):
        """
        Plot experiment results
        
        Args:
            metrics: List of metrics to plot (default: ['train_loss', 'val_loss', 'bleu_score', 'meteor_score'])
            sort_by: Column to sort results by
            ascending: Whether to sort in ascending order
        """
        if not self.results:
            raise ValueError("No experiment results available. Run experiments first.")
        
        # Default metrics to plot
        if metrics is None:
            metrics = ['train_loss', 'val_loss', 'bleu_score', 'meteor_score']
        
        # Create DataFrame from results
        df = pd.DataFrame([
            {**result, **result["params"]} 
            for result in self.results.values() 
            if result["successful"]
        ])
        
        # Drop the params column as we've expanded it
        if "params" in df.columns:
            df = df.drop(columns=["params"])
        
        # Sort if requested
        if sort_by:
            df = df.sort_values(by=sort_by, ascending=ascending)
        
        # Plot each metric
        for metric in metrics:
            if metric in df.columns:
                plt.figure(figsize=(12, 6))
                ax = df.plot(kind='bar', x='name', y=metric, legend=False)
                plt.title(f'{metric} by Experiment')
                plt.xlabel('Experiment')
                plt.ylabel(metric)
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.savefig(os.path.join(self.base_output_dir, f"{metric}_comparison.png"))
                plt.close()
        
        # Plot training time
        if 'training_time' in df.columns:
            plt.figure(figsize=(12, 6))
            ax = df.plot(kind='bar', x='name', y='training_time', legend=False)
            plt.title('Training Time by Experiment')
            plt.xlabel('Experiment')
            plt.ylabel('Training Time (s)')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(self.base_output_dir, "training_time_comparison.png"))
            plt.close()
        
        # Create correlation matrix
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            plt.figure(figsize=(10, 8))
            plt.imshow(corr, cmap='coolwarm', interpolation='none', aspect='auto')
            plt.colorbar()
            plt.xticks(range(len(corr)), corr.columns, rotation=90)
            plt.yticks(range(len(corr)), corr.columns)
            plt.title('Parameter Correlation Matrix')
            
            # Add correlation values
            for i in range(len(corr)):
                for j in range(len(corr)):
                    text = plt.text(j, i, f'{corr.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black")
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.base_output_dir, "correlation_matrix.png"))
            plt.close()
        
        return df