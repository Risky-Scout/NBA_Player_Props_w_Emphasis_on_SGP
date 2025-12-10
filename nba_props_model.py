"""
NBA Player Props Prediction Model
================================
Trains through December 6, 2025 for testing on December 7, 2025.
Uses api.balldontlie.io for data ingestion.
Implements strict temporal validation to prevent data leakage.

Author: Portfolio Project
Date: December 2025
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import pickle
import warnings
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

API_BASE_URL = "https://api.balldontlie.io/v1"
API_KEY = os.environ.get("BALLDONTLIE_API_KEY", "1340a2ff-7054-4504-b5b4-96e63281e062")

import os

TRAINING_CUTOFF_DATE = os.environ.get("TRAINING_CUTOFF_DATE", "2025-12-09")  # Train through this date
TEST_DATE = os.environ.get("TEST_DATE", "2025-12-10")  # Test on this date

STAT_TARGETS = ['pts', 'reb', 'ast', 'stl', 'blk']

# Feature engineering parameters
ROLLING_WINDOWS = [3, 5, 10, 15]  # Games for rolling averages
MIN_GAMES_FOR_PREDICTION = 5  # Minimum games before we can predict

# =============================================================================
# DATA INGESTION
# =============================================================================

class BallDontLieAPI:
    """Client for balldontlie.io API with proper rate limiting and error handling."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = API_BASE_URL
        self.headers = {"Authorization": api_key}
        self.request_count = 0
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Ensure we don't exceed rate limits (60 requests/minute for free tier)."""
        current_time = time.time()
        if current_time - self.last_request_time < 1.1:  # ~55 requests/min max
            time.sleep(1.1 - (current_time - self.last_request_time))
        self.last_request_time = time.time()
        self.request_count += 1
        
    def _make_request(self, endpoint: str, params: dict = None) -> dict:
        """Make API request with error handling."""
        self._rate_limit()
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            
            if response.status_code == 429:  # Rate limited
                print("Rate limited, waiting 60 seconds...")
                time.sleep(60)
                return self._make_request(endpoint, params)
            
            if response.status_code != 200:
                print(f"API Error {response.status_code}: {response.text[:200]}")
                return {"data": []}
                
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return {"data": []}
    
    def get_players(self, search: str = None, per_page: int = 100) -> List[dict]:
        """Get player information."""
        params = {"per_page": per_page}
        if search:
            params["search"] = search
        return self._make_request("players", params).get("data", [])
    
    def get_player_stats(self, season: int, player_ids: List[int] = None,
                         start_date: str = None, end_date: str = None,
                         per_page: int = 100) -> List[dict]:
        """Fetch player stats with pagination."""
        all_stats = []
        cursor = None
        
        while True:
            params = {
                "seasons[]": season,
                "per_page": per_page
            }
            if player_ids:
                params["player_ids[]"] = player_ids
            if start_date:
                params["start_date"] = start_date
            if end_date:
                params["end_date"] = end_date
            if cursor:
                params["cursor"] = cursor
                
            response = self._make_request("stats", params)
            data = response.get("data", [])
            
            if not data:
                break
                
            all_stats.extend(data)
            
            # Check for next page
            meta = response.get("meta", {})
            cursor = meta.get("next_cursor")
            
            if not cursor:
                break
                
            print(f"  Fetched {len(all_stats)} stats so far...")
            
        return all_stats
    
    def get_games(self, season: int = None, start_date: str = None, 
                  end_date: str = None, per_page: int = 100) -> List[dict]:
        """Get game information."""
        all_games = []
        cursor = None
        
        while True:
            params = {"per_page": per_page}
            if season:
                params["seasons[]"] = season
            if start_date:
                params["start_date"] = start_date
            if end_date:
                params["end_date"] = end_date
            if cursor:
                params["cursor"] = cursor
                
            response = self._make_request("games", params)
            data = response.get("data", [])
            
            if not data:
                break
                
            all_games.extend(data)
            
            meta = response.get("meta", {})
            cursor = meta.get("next_cursor")
            
            if not cursor:
                break
                
        return all_games

    def get_teams(self) -> List[dict]:
        """Get all teams."""
        return self._make_request("teams").get("data", [])


class DataIngestionEngine:
    """Handles all data ingestion and initial processing."""
    
    def __init__(self, api_key: str):
        self.api = BallDontLieAPI(api_key)
        
    def fetch_season_stats(self, season: int, end_date: str = None) -> pd.DataFrame:
        """Fetch all stats for a season up to end_date."""
        print(f"Fetching {season} season stats through {end_date}...")
        
        stats = self.api.get_player_stats(
            season=season,
            end_date=end_date
        )
        
        if not stats:
            print(f"  No stats found for {season} season")
            return pd.DataFrame()
            
        print(f"  Retrieved {len(stats)} stat lines")
        
        # Convert to DataFrame
        records = []
        for stat in stats:
            game = stat.get("game", {})
            player = stat.get("player", {})
            team = stat.get("team", {})
            
            record = {
                "player_id": player.get("id"),
                "player_name": f"{player.get('first_name', '')} {player.get('last_name', '')}".strip(),
                "team_id": team.get("id"),
                "team_abbrev": team.get("abbreviation"),
                "game_id": game.get("id"),
                "game_date": game.get("date", "")[:10],  # YYYY-MM-DD
                "season": game.get("season"),
                "home_team_id": game.get("home_team_id"),
                "visitor_team_id": game.get("visitor_team_id"),
                "home_team_score": game.get("home_team_score"),
                "visitor_team_score": game.get("visitor_team_score"),
                "min": stat.get("min", "0:00"),
                "pts": stat.get("pts", 0),
                "reb": stat.get("reb", 0),
                "ast": stat.get("ast", 0),
                "stl": stat.get("stl", 0),
                "blk": stat.get("blk", 0),
                "turnover": stat.get("turnover", 0),
                "pf": stat.get("pf", 0),
                "fgm": stat.get("fgm", 0),
                "fga": stat.get("fga", 0),
                "fg3m": stat.get("fg3m", 0),
                "fg3a": stat.get("fg3a", 0),
                "ftm": stat.get("ftm", 0),
                "fta": stat.get("fta", 0),
                "oreb": stat.get("oreb", 0),
                "dreb": stat.get("dreb", 0),
            }
            records.append(record)
            
        df = pd.DataFrame(records)
        
        if len(df) > 0:
            df['game_date'] = pd.to_datetime(df['game_date'])
            df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)
            
        return df
    
    def fetch_multi_season(self, seasons: List[int], end_date: str = None) -> pd.DataFrame:
        """Fetch stats from multiple seasons."""
        all_dfs = []
        
        for season in seasons:
            season_end = end_date if season == max(seasons) else None
            df = self.fetch_season_stats(season, season_end)
            if len(df) > 0:
                all_dfs.append(df)
                
        if not all_dfs:
            return pd.DataFrame()
            
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=['player_id', 'game_date', 'game_id'])
        combined = combined.sort_values(['player_id', 'game_date']).reset_index(drop=True)
        
        return combined


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

class FeatureEngineer:
    """
    Creates features for player props prediction.
    CRITICAL: All features use only past data to prevent leakage.
    """
    
    def __init__(self, rolling_windows: List[int] = None):
        self.rolling_windows = rolling_windows or ROLLING_WINDOWS
        self.stat_cols = ['pts', 'reb', 'ast', 'stl', 'blk', 'turnover', 
                         'fgm', 'fga', 'fg3m', 'fg3a', 'ftm', 'fta', 'min_numeric']
        
    def _parse_minutes(self, min_str) -> float:
        """Convert MM:SS string to float minutes."""
        if pd.isna(min_str) or min_str == '':
            return 0.0
        try:
            if isinstance(min_str, (int, float)):
                return float(min_str)
            if ':' in str(min_str):
                parts = str(min_str).split(':')
                return float(parts[0]) + float(parts[1]) / 60
            return float(min_str)
        except:
            return 0.0
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features. Uses shift() to ensure no data leakage.
        Each feature is calculated BEFORE the game it's attached to.
        """
        print("Engineering features...")
        df = df.copy()
        
        # Parse minutes to numeric
        df['min_numeric'] = df['min'].apply(self._parse_minutes)
        
        # Sort by player and date
        df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)
        
        # Calculate per-player features
        feature_dfs = []
        
        for player_id, player_df in df.groupby('player_id'):
            player_df = player_df.copy()
            
            # Game number for this player (used for filtering)
            player_df['player_game_num'] = range(1, len(player_df) + 1)
            
            # Rolling averages (shifted to prevent leakage)
            for window in self.rolling_windows:
                for stat in self.stat_cols:
                    col_name = f'{stat}_avg_{window}'
                    # shift(1) ensures we only use data BEFORE current game
                    player_df[col_name] = player_df[stat].shift(1).rolling(
                        window=window, min_periods=1
                    ).mean()
                    
                    # Standard deviation for variance
                    col_name_std = f'{stat}_std_{window}'
                    player_df[col_name_std] = player_df[stat].shift(1).rolling(
                        window=window, min_periods=2
                    ).std()
            
            # Days rest (shifted - rest BEFORE this game)
            player_df['prev_game_date'] = player_df['game_date'].shift(1)
            player_df['days_rest'] = (
                player_df['game_date'] - player_df['prev_game_date']
            ).dt.days.fillna(7)  # Default to 7 days if first game
            player_df['days_rest'] = player_df['days_rest'].clip(0, 14)  # Cap at 14
            
            # Back-to-back indicator
            player_df['is_back_to_back'] = (player_df['days_rest'] == 1).astype(int)
            
            # Season averages (up to but not including current game)
            for stat in self.stat_cols:
                player_df[f'{stat}_season_avg'] = player_df[stat].shift(1).expanding().mean()
            
            # Trend features (recent vs season average)
            for stat in ['pts', 'reb', 'ast']:
                recent = player_df[f'{stat}_avg_5']
                season = player_df[f'{stat}_season_avg']
                player_df[f'{stat}_trend'] = (recent - season) / (season + 1)
            
            feature_dfs.append(player_df)
        
        result = pd.concat(feature_dfs, ignore_index=True)
        
        # Home/Away indicator
        result['is_home'] = (result['team_id'] == result['home_team_id']).astype(int)
        
        # Opponent ID
        result['opponent_id'] = np.where(
            result['team_id'] == result['home_team_id'],
            result['visitor_team_id'],
            result['home_team_id']
        )
        
        # Day of week (some players perform differently on certain days)
        result['day_of_week'] = result['game_date'].dt.dayofweek
        
        # Fill NaN values
        result = result.fillna(0)
        
        print(f"  Created {len(result.columns)} columns")
        
        return result
    
    def get_feature_columns(self) -> List[str]:
        """Return list of feature column names for model input."""
        feature_cols = []
        
        # Rolling averages and stds
        for window in self.rolling_windows:
            for stat in self.stat_cols:
                feature_cols.append(f'{stat}_avg_{window}')
                feature_cols.append(f'{stat}_std_{window}')
        
        # Season averages
        for stat in self.stat_cols:
            feature_cols.append(f'{stat}_season_avg')
        
        # Trend features
        for stat in ['pts', 'reb', 'ast']:
            feature_cols.append(f'{stat}_trend')
        
        # Other features
        feature_cols.extend([
            'days_rest', 'is_back_to_back', 'is_home', 'day_of_week',
            'player_game_num'
        ])
        
        return feature_cols


# =============================================================================
# MODEL TRAINING
# =============================================================================

class NBAPropsModel:
    """
    Ensemble model for NBA player props prediction.
    Uses stacking with multiple base models for robust predictions.
    """
    
    def __init__(self, stat_targets: List[str] = None):
        self.stat_targets = stat_targets or STAT_TARGETS
        self.models = {}
        self.scalers = {}
        self.feature_cols = None
        self.training_metadata = {}
        
    def _create_base_models(self) -> List:
        """Create base models for stacking ensemble."""
        base_models = [
            ('rf', RandomForestRegressor(
                n_estimators=100, max_depth=10, min_samples_leaf=5,
                random_state=42, n_jobs=-1
            )),
            ('gb', GradientBoostingRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                min_samples_leaf=5, random_state=42
            )),
            ('xgb', xgb.XGBRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                min_child_weight=5, random_state=42, verbosity=0
            )),
            ('lgb', lgb.LGBMRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                min_child_samples=5, random_state=42, verbose=-1
            )),
        ]
        return base_models
    
    def _create_stacking_model(self) -> StackingRegressor:
        """Create stacking ensemble with meta-learner."""
        return StackingRegressor(
            estimators=self._create_base_models(),
            final_estimator=Ridge(alpha=1.0),
            cv=5,
            n_jobs=-1
        )
    
    def train(self, df: pd.DataFrame, feature_cols: List[str], 
              validation_split: float = 0.15) -> Dict:
        """
        Train models for each stat target.
        Uses temporal validation (most recent data for validation).
        """
        print("\n" + "="*60)
        print("TRAINING NBA PROPS MODELS")
        print("="*60)
        
        self.feature_cols = feature_cols
        results = {}
        
        # Filter to players with enough games
        df = df[df['player_game_num'] >= MIN_GAMES_FOR_PREDICTION].copy()
        
        # Sort by date for temporal split
        df = df.sort_values('game_date').reset_index(drop=True)
        
        # Temporal train/validation split
        split_idx = int(len(df) * (1 - validation_split))
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]
        
        print(f"\nTraining samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        print(f"Training date range: {train_df['game_date'].min().date()} to {train_df['game_date'].max().date()}")
        print(f"Validation date range: {val_df['game_date'].min().date()} to {val_df['game_date'].max().date()}")
        
        # Verify no leakage: validation dates should all be AFTER training dates
        assert val_df['game_date'].min() >= train_df['game_date'].max(), "DATA LEAKAGE DETECTED!"
        print("✓ No temporal data leakage detected")
        
        X_train = train_df[feature_cols].values
        X_val = val_df[feature_cols].values
        
        for stat in self.stat_targets:
            print(f"\n--- Training {stat.upper()} model ---")
            
            y_train = train_df[stat].values
            y_val = val_df[stat].values
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train stacking model
            model = self._create_stacking_model()
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            val_preds = model.predict(X_val_scaled)
            
            mae = mean_absolute_error(y_val, val_preds)
            rmse = np.sqrt(mean_squared_error(y_val, val_preds))
            within_2 = np.mean(np.abs(y_val - val_preds) <= 2) * 100
            within_3 = np.mean(np.abs(y_val - val_preds) <= 3) * 100
            
            results[stat] = {
                'mae': mae,
                'rmse': rmse,
                'within_2': within_2,
                'within_3': within_3
            }
            
            print(f"  MAE: {mae:.3f}")
            print(f"  RMSE: {rmse:.3f}")
            print(f"  Within 2: {within_2:.1f}%")
            print(f"  Within 3: {within_3:.1f}%")
            
            self.models[stat] = model
            self.scalers[stat] = scaler
        
        # Store metadata
        self.training_metadata = {
            'training_cutoff': train_df['game_date'].max().strftime('%Y-%m-%d'),
            'validation_start': val_df['game_date'].min().strftime('%Y-%m-%d'),
            'training_samples': len(train_df),
            'validation_samples': len(val_df),
            'feature_columns': feature_cols,
            'stat_targets': self.stat_targets,
            'results': results
        }
        
        return results
    
    def predict(self, X: np.ndarray, stat: str) -> np.ndarray:
        """Make predictions for a single stat."""
        if stat not in self.models:
            raise ValueError(f"No model trained for {stat}")
            
        X_scaled = self.scalers[stat].transform(X)
        return self.models[stat].predict(X_scaled)
    
    def predict_all(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Make predictions for all stats."""
        return {stat: self.predict(X, stat) for stat in self.stat_targets}
    
    def save(self, filepath: str):
        """Save model to pickle file."""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_cols': self.feature_cols,
            'stat_targets': self.stat_targets,
            'training_metadata': self.training_metadata
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f, protocol=4)
            
        print(f"\n✓ Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'NBAPropsModel':
        """Load model from pickle file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        instance = cls(stat_targets=model_data['stat_targets'])
        instance.models = model_data['models']
        instance.scalers = model_data['scalers']
        instance.feature_cols = model_data['feature_cols']
        instance.training_metadata = model_data['training_metadata']
        
        return instance


# =============================================================================
# COVARIANCE MATRIX FOR SGP (Same Game Parlays)
# =============================================================================

class SGPCovarianceCalculator:
    """
    Calculate covariance matrix between player stat predictions.
    Essential for proper SGP pricing.
    """
    
    def __init__(self):
        self.covariance_matrices = {}
        
    def calculate_player_covariance(self, df: pd.DataFrame, 
                                    player_id: int,
                                    stats: List[str]) -> np.ndarray:
        """Calculate covariance matrix for a player's stats."""
        player_df = df[df['player_id'] == player_id]
        
        if len(player_df) < 10:
            return np.eye(len(stats))  # Return identity if insufficient data
            
        stat_data = player_df[stats].values
        return np.cov(stat_data.T)
    
    def calculate_all_covariances(self, df: pd.DataFrame,
                                  stats: List[str] = None) -> Dict:
        """Calculate covariance matrices for all players."""
        stats = stats or STAT_TARGETS
        
        for player_id in df['player_id'].unique():
            cov_matrix = self.calculate_player_covariance(df, player_id, stats)
            self.covariance_matrices[player_id] = {
                'stats': stats,
                'matrix': cov_matrix
            }
            
        return self.covariance_matrices
    
    def save(self, filepath: str):
        """Save covariance matrices."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.covariance_matrices, f, protocol=4)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_full_pipeline(api_key: str = API_KEY, 
                      training_cutoff: str = TRAINING_CUTOFF_DATE,
                      output_dir: str = ".") -> Dict:
    """
    Run complete training pipeline.
    """
    print("="*60)
    print("NBA PLAYER PROPS MODEL - TRAINING PIPELINE")
    print("="*60)
    print(f"Training cutoff: {training_cutoff}")
    print(f"API endpoint: {API_BASE_URL}")
    print(f"API key: {api_key[:10]}...")
    print()
    
    # Determine seasons to fetch
    cutoff_date = datetime.strptime(training_cutoff, '%Y-%m-%d')
    current_season = cutoff_date.year if cutoff_date.month >= 10 else cutoff_date.year - 1
    seasons = [current_season - 1, current_season]  # Last 2 seasons
    
    print(f"Fetching seasons: {seasons}")
    
    # Step 1: Data Ingestion
    print("\n" + "="*60)
    print("STEP 1: DATA INGESTION")
    print("="*60)
    
    engine = DataIngestionEngine(api_key)
    raw_data = engine.fetch_multi_season(seasons, end_date=training_cutoff)
    
    if len(raw_data) == 0:
        raise ValueError("No data retrieved from API!")
        
    print(f"\nTotal records: {len(raw_data)}")
    print(f"Date range: {raw_data['game_date'].min().date()} to {raw_data['game_date'].max().date()}")
    print(f"Unique players: {raw_data['player_id'].nunique()}")
    
    # Step 2: Feature Engineering
    print("\n" + "="*60)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*60)
    
    fe = FeatureEngineer()
    featured_data = fe.engineer_features(raw_data)
    feature_cols = fe.get_feature_columns()
    
    print(f"Feature columns: {len(feature_cols)}")
    
    # Step 3: Model Training
    print("\n" + "="*60)
    print("STEP 3: MODEL TRAINING")
    print("="*60)
    
    model = NBAPropsModel()
    results = model.train(featured_data, feature_cols)
    
    # Step 4: SGP Covariance Matrix
    print("\n" + "="*60)
    print("STEP 4: SGP COVARIANCE CALCULATION")
    print("="*60)
    
    sgp_calc = SGPCovarianceCalculator()
    covariances = sgp_calc.calculate_all_covariances(featured_data)
    print(f"Calculated covariances for {len(covariances)} players")
    
    # Step 5: Save Models
    print("\n" + "="*60)
    print("STEP 5: SAVING MODELS")
    print("="*60)
    
    model.save(f"{output_dir}/trained_model.pkl")
    sgp_calc.save(f"{output_dir}/SGP_COVARIANCE_MATRIX.pkl")
    
    # Save previous results
    prev_results = {
        'model_path': f"{output_dir}/trained_model.pkl",
        'results': results,
        'train_size': model.training_metadata['training_samples'],
        'val_size': model.training_metadata['validation_samples']
    }
    with open(f"{output_dir}/previous_results.pkl", 'wb') as f:
        pickle.dump(prev_results, f, protocol=4)
    
    # Save raw data for reference
    featured_data.to_pickle(f"{output_dir}/training_data.pkl")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"\nFiles saved to {output_dir}:")
    print("  - trained_model.pkl")
    print("  - SGP_COVARIANCE_MATRIX.pkl")
    print("  - previous_results.pkl")
    print("  - training_data.pkl")
    
    return {
        'model': model,
        'data': featured_data,
        'results': results,
        'covariances': covariances
    }


if __name__ == "__main__":
    run_full_pipeline()
