#!/usr/bin/env python3
"""
=============================================================================
DAILY TRAINING PIPELINE
=============================================================================
Run this script daily (via GitHub Actions at 6 AM EST) to:
1. Fetch latest data from BallDontLie API
2. Engineer features
3. Train/update ensemble model
4. Calculate SGP correlations
5. Save artifacts

Usage:
    python run_daily_training.py
    
Environment:
    BALLDONTLIE_API_KEY - Your API key
=============================================================================
"""

import os
import sys
from datetime import datetime, date, timedelta

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nba_model_core import (
    BallDontLieAPI, 
    DataProcessor, 
    AdvancedFeatureEngineer,
    Config,
    CONFIG
)
from nba_model_training import WorldClassEnsemble, SGPCorrelationCalculator


def run_daily_training():
    """
    Main daily training pipeline.
    """
    print("="*70)
    print("NBA PLAYER PROPS - DAILY TRAINING PIPELINE")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Initialize API client
    api = BallDontLieAPI()
    
    # Calculate training cutoff (yesterday)
    today = date.today()
    training_cutoff = (today - timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"\nTraining cutoff date: {training_cutoff}")
    print(f"Predictions will be for: {today.strftime('%Y-%m-%d')}")
    
    # =================================================================
    # STEP 1: FETCH DATA
    # =================================================================
    print("\n" + "="*50)
    print("STEP 1: FETCHING DATA")
    print("="*50)
    
    # Determine seasons to fetch
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    if current_month >= 10:  # Oct-Dec: current season started
        seasons = [current_year - 1, current_year]
    else:  # Jan-Sep: current season is previous year
        seasons = [current_year - 2, current_year - 1]
    
    print(f"Fetching seasons: {seasons}")
    
    all_stats = []
    for season in seasons:
        print(f"\n  Fetching season {season}...")
        stats = api.get_stats(seasons=[season], end_date=training_cutoff)
        print(f"    Got {len(stats)} game records")
        all_stats.extend(stats)
    
    print(f"\nTotal records fetched: {len(all_stats)}")
    
    # Convert to DataFrame
    df = DataProcessor.stats_to_dataframe(all_stats)
    print(f"DataFrame shape: {df.shape}")
    
    if len(df) < 1000:
        print("ERROR: Insufficient data for training!")
        return False
    
    # =================================================================
    # STEP 2: FEATURE ENGINEERING
    # =================================================================
    print("\n" + "="*50)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*50)
    
    feature_engineer = AdvancedFeatureEngineer()
    
    # Calculate team defensive stats
    print("\nCalculating team defensive stats...")
    team_defense = feature_engineer.calculate_team_defensive_stats(df)
    print(f"  Team defense records: {len(team_defense)}")
    
    # Calculate team pace
    print("\nCalculating team pace...")
    team_pace = feature_engineer.calculate_team_pace(df)
    print(f"  Team pace records: {len(team_pace)}")
    
    # Engineer all features
    print("\nEngineering player features...")
    df_features = feature_engineer.engineer_features(df, team_defense, team_pace)
    print(f"  Final DataFrame shape: {df_features.shape}")
    
    # Get feature columns
    feature_cols = feature_engineer.get_feature_columns()
    
    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in df_features.columns:
            df_features[col] = 0
    
    print(f"  Feature columns: {len(feature_cols)}")
    
    # =================================================================
    # STEP 3: TRAIN MODEL
    # =================================================================
    print("\n" + "="*50)
    print("STEP 3: TRAINING MODEL")
    print("="*50)
    
    model = WorldClassEnsemble()
    results = model.train(df_features, feature_cols)
    
    # Save model
    model.save('trained_model.pkl')
    
    # =================================================================
    # STEP 4: CALCULATE SGP CORRELATIONS
    # =================================================================
    print("\n" + "="*50)
    print("STEP 4: CALCULATING SGP CORRELATIONS")
    print("="*50)
    
    sgp_calculator = SGPCorrelationCalculator()
    sgp_calculator.calculate_correlations(df_features)
    sgp_calculator.save('SGP_COVARIANCE_MATRIX.pkl')
    
    # =================================================================
    # STEP 5: SAVE TRAINING DATA
    # =================================================================
    print("\n" + "="*50)
    print("STEP 5: SAVING ARTIFACTS")
    print("="*50)
    
    # Save processed data for reference
    training_data = {
        'df_features': df_features,
        'feature_cols': feature_cols,
        'training_cutoff': training_cutoff,
        'seasons': seasons,
        'team_defense': team_defense,
        'team_pace': team_pace
    }
    
    import pickle
    with open('training_data.pkl', 'wb') as f:
        pickle.dump(training_data, f, protocol=4)
    print("✓ Training data saved")
    
    # Save summary
    summary = {
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'training_cutoff': training_cutoff,
        'total_records': len(df),
        'seasons': seasons,
        'model_results': results
    }
    
    import json
    with open('training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print("✓ Training summary saved")
    
    # =================================================================
    # SUMMARY
    # =================================================================
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nModel performance summary:")
    for stat, metrics in results.items():
        print(f"  {stat.upper()}: MAE={metrics['val_mae']:.3f}, RMSE={metrics['val_rmse']:.3f}, "
              f"Within 2={metrics['within_2']:.1f}%")
    
    print("\nArtifacts saved:")
    print("  - trained_model.pkl")
    print("  - SGP_COVARIANCE_MATRIX.pkl")
    print("  - training_data.pkl")
    print("  - training_summary.json")
    
    return True


if __name__ == "__main__":
    success = run_daily_training()
    sys.exit(0 if success else 1)
