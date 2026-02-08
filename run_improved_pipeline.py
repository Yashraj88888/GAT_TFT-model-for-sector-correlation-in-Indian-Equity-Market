#!/usr/bin/env python3
"""
Run Improved Model Pipeline v2
Usage: python run_improved_pipeline.py
"""

import os
import sys
import subprocess
import time

SCRIPTS = [
    ("training_v2.py", "Training improved model v2..."),
    ("evaluation_v2.py", "Evaluating model on test set..."),
]

def run_script(script_path: str, description: str) -> bool:
    """Run a single script"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"📄 Running: {script_path}")
    print(f"{'='*60}\n")
    
    if not os.path.exists(script_path):
        print(f"❌ ERROR: {script_path} not found!")
        return False
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if result.returncode != 0:
            print(f"\n❌ ERROR in {script_path}")
            return False
        
        elapsed = time.time() - start_time
        print(f"\n✅ Completed in {elapsed:.1f} seconds")
        return True
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {str(e)}")
        return False

def main():
    """Run the improved pipeline"""
    print("\n" + "🌟" * 30)
    print("IMPROVED GAT-TFT MODEL PIPELINE v2")
    print("🌟" * 30)
    
    print("\nThis pipeline will:")
    for i, (script, desc) in enumerate(SCRIPTS, 1):
        print(f"  {i}. {desc}")
    
    print("\nPrerequisites:")
    print("  - Run 01_data_download.py first (if not done)")
    print("  - Run 02_data_preprocessing.py")
    print("  - Run 03_data_windowing.py")
    print("  - Data should be in data/processed/indian_windowed.pkl")
    
    response = input("\nContinue? (y/n): ").strip().lower()
    
    if response not in ['y', 'yes', '']:
        print("Exiting.")
        return
    
    # Run each script
    for script, description in SCRIPTS:
        success = run_script(script, description)
        if not success:
            print(f"\n❌ Pipeline failed at: {script}")
            sys.exit(1)
        time.sleep(1)
    
    # Final message
    print("\n" + "🎉" * 30)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("🎉" * 30)
    print("\nResults available in:")
    print("  • models/India/best_model_v2.pt     - Trained model")
    print("  • results/evaluation_results_v2.json - Metrics")
    print("  • results/evaluation_plots_v2.png   - Visualizations")
    print("  • results/predictions_v2.json       - Raw predictions")
    print("  • logs/training_v2.log              - Training log")

if __name__ == "__main__":
    main()
