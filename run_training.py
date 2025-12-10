#!/usr/bin/env python3
"""
NBA Player Props Model - Training Script
=========================================
Run this script to train the model through December 6, 2025.

Usage:
    python run_training.py

The script will:
1. Fetch data from api.balldontlie.io through 12/6/2025
2. Engineer features with strict temporal validation (no leakage)
3. Train ensemble models for PTS, REB, AST, STL, BLK
4. Calculate SGP covariance matrices
5. Save all outputs for December 7, 2025 predictions

Author: Portfolio Project
Date: December 2025
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nba_props_model import run_full_pipeline, API_KEY, TRAINING_CUTOFF_DATE, TEST_DATE

def main():
    print("="*70)
    print(f"NBA PLAYER PROPS MODEL - TRAINING FOR {TEST_DATE} PREDICTIONS")
    print("="*70)
    print()
    print("Configuration:")
    print(f"  Training data cutoff: {TRAINING_CUTOFF_DATE}")
    print(f"  Test date: {TEST_DATE}")
    print(f"  API endpoint: https://api.balldontlie.io/v1")
    print()
    
    # Run the full pipeline
    try:
        result = run_full_pipeline(
            api_key=API_KEY,
            training_cutoff=TRAINING_CUTOFF_DATE,
            output_dir="."
        )
        
        print("\n" + "="*70)
        print("SUCCESS - MODEL TRAINED AND READY")
        print("="*70)
        print()
        print("Model Performance Summary:")
        for stat, metrics in result['results'].items():
            print(f"\n  {stat.upper()}:")
            print(f"    MAE: {metrics['mae']:.3f}")
            print(f"    RMSE: {metrics['rmse']:.3f}")
            print(f"    Within 2: {metrics['within_2']:.1f}%")
            print(f"    Within 3: {metrics['within_3']:.1f}%")
            
        print("\n" + "-"*70)
        print("Files created:")
        print("  - trained_model.pkl (main model)")
        print("  - SGP_COVARIANCE_MATRIX.pkl (for same-game parlays)")
        print("  - previous_results.pkl (performance metrics)")
        print("  - training_data.pkl (processed training data)")
        print()
        print(f"Ready to predict for {TEST_DATE}!")
        
    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
