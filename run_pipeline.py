#!/usr/bin/env python3
"""
Quick Start Script for GAT-TFT Pipeline
Run with: python quick_start.py
"""

import os
import sys
import subprocess
import time

# List of scripts to run in order
SCRIPTS = [
    ("_00_setup_environment.py", "Setting up environment..."),
    ("01_data_download.py", "Downloading stock data..."),
    ("02_data_preprocessing.py", "Preprocessing data..."),
    ("03_data_windowing.py", "Creating temporal windows..."),
    ("training.py", "Training GAT-TFT model..."),
    ("06_evaluation.py", "Evaluating model..."),
]

def run_script(script_path, description):
    """Run a single script"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"📄 Running: {script_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(script_path):
        print(f"❌ ERROR: {script_path} not found!")
        return False
    
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True
        )
        
        # Print any output
        if result.stdout:
            print(result.stdout)
        
        # Check for errors
        if result.returncode != 0:
            print(f"\n❌ ERROR in {script_path}:")
            print(f"Return code: {result.returncode}")
            if result.stderr:
                print("Error output:")
                print(result.stderr[:500])  # Print first 500 chars
            return False
        
        elapsed = time.time() - start_time
        print(f"\n✅ Completed in {elapsed:.1f} seconds")
        return True
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {str(e)}")
        return False

def main():
    """Run all scripts in order"""
    print("\n" + "🌟" * 30)
    print("GAT-TFT STOCK PREDICTION PIPELINE")
    print("🌟" * 30)
    
    print("\nThis will run the following steps:")
    for i, (script, desc) in enumerate(SCRIPTS, 1):
        print(f"{i}. {desc}")
    
    print("\nEstimated time: 30-60 minutes")
    response = input("\nContinue? (y/n): ").strip().lower()
    
    if response not in ['y', 'yes']:
        print("Exiting.")
        return
    
    # Run each script
    for script, description in SCRIPTS:
        success = run_script(script, description)
        if not success:
            print(f"\n❌ Pipeline failed at: {script}")
            print("Please check the error message above.")
            sys.exit(1)
        
        # Small delay between scripts
        time.sleep(2)
    
    # Final message
    print("\n" + "🎉" * 30)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("🎉" * 30)
    print("\nResults are available in:")
    print("  • models/India/best_model.pt  - Trained model")
    print("  • results/                    - Evaluation results")
    print("  • logs/                       - Log files")
    print("\nTo rerun the pipeline, delete the output files or use:")
    print("  python run_pipeline.py --force")

if __name__ == "__main__":
    main()