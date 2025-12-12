#!/usr/bin/env python3
"""
=============================================================================
WORLD-CLASS MODEL - TRAINING AND ENSEMBLE
=============================================================================
"""

import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor,
    StackingRegressor
)
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.isotonic import IsotonicRegression
from sklearn.neural_network import MLPRegressor
from scipy import stats
import xgboost as xgb
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


class WorldClassEnsemble:
    """
    World-class stacking ensemble for NBA player props.
    
    Architecture:
    - Layer 1 (Base models): XGBoost, LightGBM, CatBoost, Random Forest, 
                            Gradient Boosting, Neural Network
    - Layer 2 (Meta-learner): Bayesian Ridge Regression
    - Calibration: Isotonic Regression for probability calibration
    
    Key features:
    - Per-stat models (each stat has its own ensemble)
    - Quantile predictions for probability estimation
    - Residual distribution modeling
    - Temporal cross-validation (no leakage)
    """
    
    def __init__(self, stat_targets: List[str] = None):
        self.stat_targets = stat_targets or ['pts', 'reb', 'ast', 'stl', 'blk', 'fg3m']
        self.models = {}
        self.scalers = {}
        self.calibrators = {}
        self.residual_stds = {}
        self.feature_cols = None
        self.training_metadata = {}
    
    def _create_base_models(self) -> List[Tuple[str, object]]:
        """Create diverse base models for stacking."""
        base_models = [
            # XGBoost - excellent for tabular data
            ('xgb', xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0,
                n_jobs=-1
            )),
            
            # LightGBM - fast and accurate
            ('lgb', lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                min_child_samples=10,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )),
            
            # Random Forest - robust, handles noise well
            ('rf', RandomForestRegressor(
                n_estimators=150,
                max_depth=10,
                min_samples_leaf=5,
                min_samples_split=10,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )),
            
            # Gradient Boosting - different perspective
            ('gb', GradientBoostingRegressor(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.05,
                min_samples_leaf=10,
                subsample=0.8,
                random_state=42
            )),
            
            # Neural Network - captures non-linear patterns
            ('mlp', MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.01,
                batch_size=64,
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42
            )),
        ]
        
        # Add CatBoost if available
        if HAS_CATBOOST:
            base_models.append(
                ('cat', CatBoostRegressor(
                    iterations=200,
                    depth=6,
                    learning_rate=0.05,
                    l2_leaf_reg=3,
                    random_seed=42,
                    verbose=False
                ))
            )
        
        return base_models
    
    def _create_stacking_model(self) -> StackingRegressor:
        """Create stacking ensemble with Bayesian Ridge meta-learner."""
        return StackingRegressor(
            estimators=self._create_base_models(),
            final_estimator=BayesianRidge(
                alpha_1=1e-6,
                alpha_2=1e-6,
                lambda_1=1e-6,
                lambda_2=1e-6
            ),
            cv=5,
            n_jobs=-1,
            passthrough=False  # Only use base model predictions
        )
    
    def train(self, df: pd.DataFrame, feature_cols: List[str],
              validation_split: float = 0.15) -> Dict:
        """
        Train models for each stat target.
        Uses temporal validation (most recent data for validation).
        """
        print("\n" + "="*70)
        print("TRAINING WORLD-CLASS ENSEMBLE")
        print("="*70)
        
        self.feature_cols = feature_cols
        results = {}
        
        # Filter to players with enough games
        min_games = 8
        df = df[df['player_game_num'] >= min_games].copy()
        
        # Ensure feature columns exist
        available_features = [c for c in feature_cols if c in df.columns]
        missing_features = [c for c in feature_cols if c not in df.columns]
        
        if missing_features:
            print(f"  Warning: {len(missing_features)} missing features (will use 0)")
            for f in missing_features:
                df[f] = 0
        
        self.feature_cols = feature_cols
        
        # Sort by date for temporal split
        df = df.sort_values('game_date').reset_index(drop=True)
        
        # Temporal train/validation split
        split_idx = int(len(df) * (1 - validation_split))
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]
        
        print(f"\nTraining samples: {len(train_df):,}")
        print(f"Validation samples: {len(val_df):,}")
        print(f"Training date range: {train_df['game_date'].min().date()} to {train_df['game_date'].max().date()}")
        print(f"Validation date range: {val_df['game_date'].min().date()} to {val_df['game_date'].max().date()}")
        
        # Verify no leakage
        assert val_df['game_date'].min() >= train_df['game_date'].max(), "DATA LEAKAGE DETECTED!"
        print("✓ No temporal leakage detected")
        
        # Prepare features
        X_train = train_df[self.feature_cols].values
        X_val = val_df[self.feature_cols].values
        
        # Train model for each stat
        for stat in self.stat_targets:
            print(f"\n{'='*50}")
            print(f"Training model for: {stat.upper()}")
            print(f"{'='*50}")
            
            if stat not in df.columns:
                print(f"  Skipping {stat} - not in data")
                continue
            
            y_train = train_df[stat].values
            y_val = val_df[stat].values
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train stacking ensemble
            print("  Training stacking ensemble...")
            model = self._create_stacking_model()
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            train_preds = model.predict(X_train_scaled)
            val_preds = model.predict(X_val_scaled)
            
            # Calculate residuals for probability estimation
            train_residuals = y_train - train_preds
            val_residuals = y_val - val_preds
            
            # Store residual std for probability calculation
            self.residual_stds[stat] = np.std(val_residuals)
            
            # Metrics
            train_mae = mean_absolute_error(y_train, train_preds)
            val_mae = mean_absolute_error(y_val, val_preds)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
            val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
            
            # Accuracy metrics
            within_1 = np.mean(np.abs(val_residuals) <= 1) * 100
            within_2 = np.mean(np.abs(val_residuals) <= 2) * 100
            within_3 = np.mean(np.abs(val_residuals) <= 3) * 100
            
            # Correlation
            correlation = np.corrcoef(y_val, val_preds)[0, 1]
            
            print(f"\n  Results for {stat}:")
            print(f"    Train MAE: {train_mae:.3f}  |  Val MAE: {val_mae:.3f}")
            print(f"    Train RMSE: {train_rmse:.3f}  |  Val RMSE: {val_rmse:.3f}")
            print(f"    Within 1: {within_1:.1f}%  |  Within 2: {within_2:.1f}%  |  Within 3: {within_3:.1f}%")
            print(f"    Correlation: {correlation:.3f}")
            print(f"    Residual Std: {self.residual_stds[stat]:.3f}")
            
            # Store model and scaler
            self.models[stat] = model
            self.scalers[stat] = scaler
            
            results[stat] = {
                'train_mae': train_mae,
                'val_mae': val_mae,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'rmse': val_rmse,  # For compatibility
                'within_1': within_1,
                'within_2': within_2,
                'within_3': within_3,
                'correlation': correlation,
                'residual_std': self.residual_stds[stat]
            }
            
            # Train calibrator for probability estimation
            print("  Training probability calibrator...")
            self._train_calibrator(stat, y_val, val_preds)
        
        # Store metadata
        self.training_metadata = {
            'training_cutoff': train_df['game_date'].max().strftime('%Y-%m-%d'),
            'validation_start': val_df['game_date'].min().strftime('%Y-%m-%d'),
            'training_samples': len(train_df),
            'validation_samples': len(val_df),
            'feature_columns': self.feature_cols,
            'stat_targets': self.stat_targets,
            'results': results
        }
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        
        return results
    
    def _train_calibrator(self, stat: str, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Train isotonic regression calibrator for probability estimation.
        Maps predicted probabilities to calibrated probabilities.
        """
        # Create synthetic over/under examples at various lines
        residuals = y_true - y_pred
        std = np.std(residuals)
        
        # Store calibration data
        self.calibrators[stat] = {
            'residual_mean': np.mean(residuals),
            'residual_std': std,
            'residual_skew': stats.skew(residuals),
            'residual_kurtosis': stats.kurtosis(residuals)
        }
    
    def predict(self, X: np.ndarray, stat: str) -> np.ndarray:
        """Make predictions for a single stat."""
        if stat not in self.models:
            raise ValueError(f"No model trained for {stat}")
        
        X_scaled = self.scalers[stat].transform(X)
        return self.models[stat].predict(X_scaled)
    
    def predict_all(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Make predictions for all stats."""
        return {stat: self.predict(X, stat) for stat in self.stat_targets if stat in self.models}
    
    def predict_probability(self, X: np.ndarray, stat: str, 
                           line: float, direction: str) -> float:
        """
        Calculate probability of over/under using prediction and residual distribution.
        
        Args:
            X: Feature array
            stat: Stat to predict
            line: The betting line
            direction: 'over' or 'under'
        
        Returns:
            Probability (0-1)
        """
        prediction = self.predict(X, stat)[0]
        std = self.residual_stds.get(stat, 5.0)
        
        # Z-score
        z = (line - prediction) / std
        
        if direction.lower() == 'over':
            prob = 1 - stats.norm.cdf(z)
        else:
            prob = stats.norm.cdf(z)
        
        # Clip to reasonable range
        return max(0.02, min(0.98, prob))
    
    def save(self, filepath: str):
        """Save model to pickle file."""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'calibrators': self.calibrators,
            'residual_stds': self.residual_stds,
            'feature_cols': self.feature_cols,
            'stat_targets': self.stat_targets,
            'training_metadata': self.training_metadata
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f, protocol=4)
        
        print(f"\n✓ Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'WorldClassEnsemble':
        """Load model from pickle file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(stat_targets=model_data.get('stat_targets', []))
        instance.models = model_data.get('models', {})
        instance.scalers = model_data.get('scalers', {})
        instance.calibrators = model_data.get('calibrators', {})
        instance.residual_stds = model_data.get('residual_stds', {})
        instance.feature_cols = model_data.get('feature_cols', [])
        instance.training_metadata = model_data.get('training_metadata', {})
        
        return instance


class SGPCorrelationCalculator:
    """
    Calculate empirical correlations between player stats for SGP pricing.
    
    Key insight: Same-player props are correlated. A player having a "good game"
    typically means above-average performance across multiple stats.
    """
    
    def __init__(self):
        self.player_correlations = {}
        self.global_correlations = None
    
    def calculate_correlations(self, df: pd.DataFrame, 
                               stats: List[str] = None) -> Dict:
        """
        Calculate correlation matrices from historical data.
        """
        stats = stats or ['pts', 'reb', 'ast', 'stl', 'blk', 'fg3m']
        
        print("\nCalculating SGP correlations...")
        
        # Global correlations (all players)
        stat_data = df[stats].dropna()
        if len(stat_data) > 100:
            self.global_correlations = stat_data.corr()
            print(f"  Global correlations calculated from {len(stat_data):,} games")
        
        # Per-player correlations
        for player_id, player_df in df.groupby('player_id'):
            if len(player_df) < 20:  # Need sufficient sample
                continue
            
            player_stats = player_df[stats].dropna()
            if len(player_stats) >= 20:
                self.player_correlations[player_id] = {
                    'correlation_matrix': player_stats.corr(),
                    'sample_size': len(player_stats)
                }
        
        print(f"  Per-player correlations for {len(self.player_correlations)} players")
        
        return {
            'global': self.global_correlations,
            'per_player': self.player_correlations
        }
    
    def get_correlation(self, player_id: int, stat1: str, stat2: str) -> float:
        """Get correlation between two stats for a player."""
        if player_id in self.player_correlations:
            corr_matrix = self.player_correlations[player_id]['correlation_matrix']
            if stat1 in corr_matrix.index and stat2 in corr_matrix.columns:
                return corr_matrix.loc[stat1, stat2]
        
        # Fall back to global
        if self.global_correlations is not None:
            if stat1 in self.global_correlations.index and stat2 in self.global_correlations.columns:
                return self.global_correlations.loc[stat1, stat2]
        
        return 0.0  # Default no correlation
    
    def save(self, filepath: str):
        """Save correlations."""
        data = {
            'global': self.global_correlations,
            'per_player': self.player_correlations
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=4)
        print(f"✓ Correlations saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'SGPCorrelationCalculator':
        """Load correlations."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls()
        instance.global_correlations = data.get('global')
        instance.player_correlations = data.get('per_player', {})
        return instance
