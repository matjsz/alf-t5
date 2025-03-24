#!/usr/bin/env python
"""
Example script demonstrating the model architecture experimentation functionality
"""

import os
import sys
from alf_t5 import ModelExperiment

def main():
    """Run model experimentation example"""
    
    # Check if data file argument is provided
    if len(sys.argv) < 2:
        print("Usage: python model_experiment_example.py <data_file>")
        print("Example: python model_experiment_example.py my_language_data.txt")
        sys.exit(1)
    
    data_file = sys.argv[1]
    
    # Create output directory
    output_dir = "model_experiments"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the experiment framework
    experiment = ModelExperiment(
        base_output_dir=output_dir,
        data_file=data_file,
        test_size=0.2,
        augment=True,
        metrics=["bleu", "meteor"],
        parallel_experiments=1  # Set to higher value to run experiments in parallel
    )
    
    # Add a simple experiment with default parameters
    experiment.add_experiment(
        name="baseline",
        model_name="t5-small",
        num_epochs=20,
        batch_size=16,
        learning_rate=3e-4
    )
    
    # Add a grid search experiment to explore parameter space
    experiment.add_grid_search(
        name_prefix="peft_config",
        model_names=["t5-small"],
        peft_r=[4, 8],  # LoRA rank
        peft_lora_alpha=[16, 32],  # LoRA alpha parameter
        learning_rate=[1e-4, 3e-4]
    )
    
    # Add a grid search for different training configurations
    experiment.add_grid_search(
        name_prefix="training_config",
        model_names=["t5-small"],
        batch_size=[8, 16, 32],
        num_epochs=[15, 20, 25]
    )
    
    # Run all experiments
    print(f"Running {len(experiment.experiments)} experiments...")
    results_df = experiment.run_experiments()
    
    # Print results summary
    print("\nExperiments completed!")
    print(f"Results saved to {output_dir}/experiment_results.csv")
    
    # Get the best experiment by validation loss
    best_val_loss = experiment.get_best_experiment(metric="val_loss", higher_is_better=False)
    print(f"\nBest experiment by validation loss: {best_val_loss['name']}")
    print(f"  Val Loss: {best_val_loss['val_loss']:.4f}")
    
    # Get the best experiment by BLEU score
    best_bleu = experiment.get_best_experiment(metric="bleu_score", higher_is_better=True)
    print(f"\nBest experiment by BLEU score: {best_bleu['name']}")
    print(f"  BLEU Score: {best_bleu['bleu_score']:.4f}")
    
    # Get the best experiment by METEOR score
    best_meteor = experiment.get_best_experiment(metric="meteor_score", higher_is_better=True)
    print(f"\nBest experiment by METEOR score: {best_meteor['name']}")
    print(f"  METEOR Score: {best_meteor['meteor_score']:.4f}")
    
    # Plot experiment results
    print("\nGenerating result plots...")
    experiment.plot_experiment_results()
    print(f"Plots saved to {output_dir}/")

if __name__ == "__main__":
    main() 