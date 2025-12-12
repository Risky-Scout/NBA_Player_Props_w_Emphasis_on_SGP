#!/usr/bin/env python3
"""
=============================================================================
WORLD-CLASS NBA PLAYER PROPS PREDICTION MODEL
=============================================================================
Syndicate-Grade Predictive System

Key Features:
- Minutes projection model (foundation of all props)
- Opponent-adjusted predictions (Defense vs Position)
- Pace-adjusted projections (expected possessions)
- Usage redistribution modeling (injury impact)
- Game environment factors (Vegas lines, blowout risk)
- Exponentially weighted features (recent form matters more)
- Quantile regression for probability distributions
- Multi-model stacking ensemble (XGB, LGBM, CatBoost, RF, GB)
- Calibrated probabilities (isotonic regression)
- True empirical SGP correlations

Author: Syndicate-Grade Model
Version: 3.0
Date: December 2025
=============================================================================
"""

import os
import pickle
import warnings
import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from functools import lru_cache
import json

import numpy as np
import pandas as pd
import requests
from scipy import stats
from scipy.optimize import minimize

# ML imports
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor,
    StackingRegressor,
    VotingRegressor
)
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.isotonic import IsotonicRegression
from sklearn.neural_network import MLPRegressor

import xgboost as xgb
import lightgbm as lgb

# Try importing CatBoost (optional)
try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class Config:
    """Model configuration."""
    # API
    API_V1: str = "https://api.balldontlie.io/v1"
    API_V2: str = "https://api.balldontlie.io/v2"
    API_KEY: str = os.environ.get("BALLDONTLIE_API_KEY", "1340a2ff-7054-4504-b5b4-96e63281e062")
    
    # Training
    MIN_GAMES_FOR_PREDICTION: int = 8
    SEASONS_TO_FETCH: List[int] = None  # Set dynamically
    VALIDATION_SPLIT: float = 0.15
    
    # Feature engineering
    ROLLING_WINDOWS: List[int] = (3, 5, 7, 10, 15, 20)
    EWMA_SPANS: List[int] = (3, 5, 10)  # Exponential weighted moving average
    
    # Target stats
    STAT_TARGETS: List[str] = ('pts', 'reb', 'ast', 'stl', 'blk', 'fg3m')
    
    # Betting
    MIN_EDGE_PERCENT: float = 3.0
    MIN_ODDS: int = -300
    MAX_ODDS: int = 300
    
    def __post_init__(self):
        if self.SEASONS_TO_FETCH is None:
            current_year = datetime.now().year
            current_month = datetime.now().month
            if current_month >= 10:
                self.SEASONS_TO_FETCH = [current_year - 1, current_year]
            else:
                self.SEASONS_TO_FETCH = [current_year - 2, current_year - 1]

CONFIG = Config()

# =============================================================================
# API CLIENT
# =============================================================================
class BallDontLieAPI:
    """Professional API client with rate limiting and caching."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or CONFIG.API_KEY
        self.session = requests.Session()
        self.session.headers.update({"Authorization": self.api_key})
        self.last_request_time = 0
        self.min_request_interval = 0.6  # seconds between requests
        self._cache = {}
    
    def _rate_limit(self):
        """Ensure we don't exceed rate limits."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _request(self, url: str, params: dict = None) -> dict:
        """Make API request with rate limiting."""
        self._rate_limit()
        try:
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            else:
                print(f"    API error {resp.status_code}: {url}")
                return {"data": []}
        except Exception as e:
            print(f"    Request failed: {e}")
            return {"data": []}
    
    def get_v1(self, endpoint: str, params: dict = None) -> dict:
        """V1 API request."""
        return self._request(f"{CONFIG.API_V1}/{endpoint}", params)
    
    def get_v2(self, endpoint: str, params: dict = None) -> dict:
        """V2 API request."""
        return self._request(f"{CONFIG.API_V2}/{endpoint}", params)
    
    def get_all_pages(self, endpoint: str, params: dict = None, v2: bool = False) -> List[dict]:
        """Fetch all pages of paginated endpoint."""
        all_data = []
        params = params or {}
        params["per_page"] = 100
        cursor = None
        
        while True:
            if cursor:
                params["cursor"] = cursor
            
            if v2:
                resp = self.get_v2(endpoint, params)
            else:
                resp = self.get_v1(endpoint, params)
            
            data = resp.get("data", [])
            if not data:
                break
            
            all_data.extend(data)
            cursor = resp.get("meta", {}).get("next_cursor")
            if not cursor:
                break
        
        return all_data
    
    # Convenience methods
    def get_games(self, start_date: str, end_date: str = None) -> List[dict]:
        """Get games for date range."""
        params = {"start_date": start_date}
        if end_date:
            params["end_date"] = end_date
        return self.get_v1("games", params).get("data", [])
    
    def get_stats(self, seasons: List[int] = None, player_ids: List[int] = None,
                  start_date: str = None, end_date: str = None) -> List[dict]:
        """Get player stats."""
        params = {"per_page": 100}
        if seasons:
            params["seasons[]"] = seasons[0] if len(seasons) == 1 else seasons
        if player_ids:
            params["player_ids[]"] = player_ids[0] if len(player_ids) == 1 else player_ids
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self.get_all_pages("stats", params)
    
    def get_injuries(self) -> List[dict]:
        """Get current injury report."""
        return self.get_v1("player_injuries").get("data", [])
    
    def get_standings(self, season: int) -> List[dict]:
        """Get team standings."""
        return self.get_v1("standings", {"season": season}).get("data", [])
    
    def get_player_props(self, game_id: int) -> List[dict]:
        """Get player props odds from V2 API."""
        return self.get_all_pages("odds/player_props", {"game_id": game_id}, v2=True)
    
    def get_player(self, player_id: int) -> dict:
        """Get player info."""
        return self.get_v1(f"players/{player_id}").get("data", {})
    
    def get_teams(self) -> List[dict]:
        """Get all teams."""
        return self.get_v1("teams").get("data", [])


# =============================================================================
# DATA PROCESSING
# =============================================================================
class DataProcessor:
    """Process raw API data into clean DataFrames."""
    
    @staticmethod
    def parse_minutes(min_str) -> float:
        """Convert MM:SS to float minutes."""
        if pd.isna(min_str) or min_str == '' or min_str is None:
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
    
    @staticmethod
    def stats_to_dataframe(stats: List[dict]) -> pd.DataFrame:
        """Convert stats list to DataFrame."""
        records = []
        for s in stats:
            game = s.get("game", {})
            player = s.get("player", {})
            team = s.get("team", {})
            
            records.append({
                "player_id": player.get("id"),
                "player_name": f"{player.get('first_name', '')} {player.get('last_name', '')}".strip(),
                "team_id": team.get("id"),
                "team_abbrev": team.get("abbreviation"),
                "game_id": game.get("id"),
                "game_date": game.get("date", "")[:10],
                "season": game.get("season"),
                "home_team_id": game.get("home_team_id"),
                "visitor_team_id": game.get("visitor_team_id"),
                "home_team_score": game.get("home_team_score"),
                "visitor_team_score": game.get("visitor_team_score"),
                "min": s.get("min", "0:00"),
                "pts": s.get("pts", 0) or 0,
                "reb": s.get("reb", 0) or 0,
                "ast": s.get("ast", 0) or 0,
                "stl": s.get("stl", 0) or 0,
                "blk": s.get("blk", 0) or 0,
                "turnover": s.get("turnover", 0) or 0,
                "pf": s.get("pf", 0) or 0,
                "fgm": s.get("fgm", 0) or 0,
                "fga": s.get("fga", 0) or 0,
                "fg3m": s.get("fg3m", 0) or 0,
                "fg3a": s.get("fg3a", 0) or 0,
                "ftm": s.get("ftm", 0) or 0,
                "fta": s.get("fta", 0) or 0,
                "oreb": s.get("oreb", 0) or 0,
                "dreb": s.get("dreb", 0) or 0,
            })
        
        df = pd.DataFrame(records)
        if len(df) > 0:
            df['game_date'] = pd.to_datetime(df['game_date'])
            df['min_numeric'] = df['min'].apply(DataProcessor.parse_minutes)
            df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)
        
        return df
    
    @staticmethod
    def injuries_to_dict(injuries: List[dict]) -> Dict[int, dict]:
        """Convert injuries list to player_id -> injury info dict."""
        out_statuses = {'Out', 'Doubtful', 'out', 'doubtful', 'OUT', 'DOUBTFUL'}
        injury_dict = {}
        
        for inj in injuries:
            player = inj.get("player", {})
            player_id = player.get("id")
            if player_id:
                status = inj.get("status", "")
                injury_dict[player_id] = {
                    'name': f"{player.get('first_name', '')} {player.get('last_name', '')}",
                    'team': player.get('team', {}).get('abbreviation', '?'),
                    'team_id': player.get('team', {}).get('id'),
                    'status': status,
                    'description': inj.get('description', ''),
                    'is_out': status in out_statuses
                }
        
        return injury_dict


# =============================================================================
# ADVANCED FEATURE ENGINEERING
# =============================================================================
class AdvancedFeatureEngineer:
    """
    World-class feature engineering for NBA player props.
    
    Key innovations:
    1. Opponent-adjusted stats (Defense vs Position approximation)
    2. Pace-adjusted projections
    3. Exponentially weighted moving averages (recent form)
    4. Minutes projection features
    5. Game environment features
    6. Usage rate modeling
    7. Consistency/variance features
    """
    
    def __init__(self, config: Config = None):
        self.config = config or CONFIG
        self.stat_cols = ['pts', 'reb', 'ast', 'stl', 'blk', 'turnover',
                         'fgm', 'fga', 'fg3m', 'fg3a', 'ftm', 'fta', 
                         'oreb', 'dreb', 'min_numeric']
        self.team_stats_cache = {}
    
    def calculate_team_defensive_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate team defensive stats for opponent adjustments.
        Returns DataFrame with team_id, date, and defensive metrics.
        """
        # Group by opponent team and game
        df = df.copy()
        df['opponent_id'] = np.where(
            df['team_id'] == df['home_team_id'],
            df['visitor_team_id'],
            df['home_team_id']
        )
        
        # Calculate points allowed per game by each team (as opponent)
        team_def = df.groupby(['opponent_id', 'game_date']).agg({
            'pts': 'sum',
            'reb': 'sum',
            'ast': 'sum',
            'fg3m': 'sum',
            'min_numeric': 'sum'
        }).reset_index()
        
        team_def.columns = ['team_id', 'game_date', 'pts_allowed', 'reb_allowed', 
                           'ast_allowed', 'fg3m_allowed', 'total_min']
        
        # Calculate rolling defensive averages
        team_def = team_def.sort_values(['team_id', 'game_date'])
        
        for stat in ['pts_allowed', 'reb_allowed', 'ast_allowed', 'fg3m_allowed']:
            team_def[f'{stat}_avg_10'] = team_def.groupby('team_id')[stat].transform(
                lambda x: x.shift(1).rolling(10, min_periods=3).mean()
            )
        
        return team_def
    
    def calculate_team_pace(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate team pace (possessions per game)."""
        # Approximate possessions = FGA + 0.44*FTA - OREB + TO
        df = df.copy()
        
        team_game = df.groupby(['team_id', 'game_date']).agg({
            'fga': 'sum',
            'fta': 'sum',
            'oreb': 'sum',
            'turnover': 'sum',
            'min_numeric': 'sum'
        }).reset_index()
        
        team_game['possessions'] = (
            team_game['fga'] + 
            0.44 * team_game['fta'] - 
            team_game['oreb'] + 
            team_game['turnover']
        )
        
        # Pace = possessions per 48 minutes, normalized
        team_game['pace'] = team_game['possessions'] / (team_game['min_numeric'] / 48 + 0.1)
        
        # Rolling pace
        team_game = team_game.sort_values(['team_id', 'game_date'])
        team_game['pace_avg_10'] = team_game.groupby('team_id')['pace'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).mean()
        )
        
        return team_game[['team_id', 'game_date', 'pace', 'pace_avg_10']]
    
    def engineer_features(self, df: pd.DataFrame, 
                         team_defense: pd.DataFrame = None,
                         team_pace: pd.DataFrame = None) -> pd.DataFrame:
        """
        Create all features with no data leakage.
        Each feature uses only data available BEFORE the game.
        """
        print("Engineering advanced features...")
        df = df.copy()
        
        # Sort by player and date
        df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)
        
        # Calculate opponent
        df['opponent_id'] = np.where(
            df['team_id'] == df['home_team_id'],
            df['visitor_team_id'],
            df['home_team_id']
        )
        
        # Home/away
        df['is_home'] = (df['team_id'] == df['home_team_id']).astype(int)
        
        # Process each player
        feature_dfs = []
        
        for player_id, player_df in df.groupby('player_id'):
            player_df = player_df.copy()
            
            # Game number for this player
            player_df['player_game_num'] = range(1, len(player_df) + 1)
            
            # =================================================================
            # ROLLING AVERAGES (standard)
            # =================================================================
            for window in self.config.ROLLING_WINDOWS:
                for stat in self.stat_cols:
                    # Mean
                    col_name = f'{stat}_avg_{window}'
                    player_df[col_name] = player_df[stat].shift(1).rolling(
                        window=window, min_periods=1
                    ).mean()
                    
                    # Std (consistency measure)
                    col_name_std = f'{stat}_std_{window}'
                    player_df[col_name_std] = player_df[stat].shift(1).rolling(
                        window=window, min_periods=2
                    ).std()
            
            # =================================================================
            # EXPONENTIALLY WEIGHTED MOVING AVERAGES (recency)
            # =================================================================
            for span in self.config.EWMA_SPANS:
                for stat in ['pts', 'reb', 'ast', 'min_numeric', 'fga']:
                    col_name = f'{stat}_ewma_{span}'
                    player_df[col_name] = player_df[stat].shift(1).ewm(
                        span=span, min_periods=1
                    ).mean()
            
            # =================================================================
            # SEASON AVERAGES
            # =================================================================
            for stat in self.stat_cols:
                player_df[f'{stat}_season_avg'] = player_df[stat].shift(1).expanding().mean()
            
            # =================================================================
            # TREND FEATURES (hot/cold streak)
            # =================================================================
            for stat in ['pts', 'reb', 'ast', 'min_numeric']:
                recent = player_df[f'{stat}_avg_5']
                season = player_df[f'{stat}_season_avg']
                player_df[f'{stat}_trend'] = (recent - season) / (season + 0.1)
                
                # Momentum (last 3 vs last 10)
                recent_3 = player_df[f'{stat}_avg_3']
                recent_10 = player_df[f'{stat}_avg_10']
                player_df[f'{stat}_momentum'] = (recent_3 - recent_10) / (recent_10 + 0.1)
            
            # =================================================================
            # PER-MINUTE RATES (for minutes-adjusted projections)
            # =================================================================
            for stat in ['pts', 'reb', 'ast', 'stl', 'blk', 'fg3m', 'fga']:
                player_df[f'{stat}_per_min'] = player_df[stat] / (player_df['min_numeric'] + 0.1)
                player_df[f'{stat}_per_min_avg_10'] = player_df[f'{stat}_per_min'].shift(1).rolling(
                    window=10, min_periods=3
                ).mean()
            
            # =================================================================
            # REST DAYS
            # =================================================================
            player_df['prev_game_date'] = player_df['game_date'].shift(1)
            player_df['days_rest'] = (
                player_df['game_date'] - player_df['prev_game_date']
            ).dt.days.fillna(7)
            player_df['days_rest'] = player_df['days_rest'].clip(0, 14)
            player_df['is_back_to_back'] = (player_df['days_rest'] == 1).astype(int)
            player_df['is_well_rested'] = (player_df['days_rest'] >= 3).astype(int)
            
            # =================================================================
            # CONSISTENCY / VARIANCE FEATURES
            # =================================================================
            for stat in ['pts', 'reb', 'ast']:
                cv = player_df[f'{stat}_std_10'] / (player_df[f'{stat}_avg_10'] + 0.1)
                player_df[f'{stat}_cv'] = cv  # Coefficient of variation
                
                # Min/max range in last 10
                player_df[f'{stat}_max_10'] = player_df[stat].shift(1).rolling(10, min_periods=3).max()
                player_df[f'{stat}_min_10'] = player_df[stat].shift(1).rolling(10, min_periods=3).min()
                player_df[f'{stat}_range_10'] = player_df[f'{stat}_max_10'] - player_df[f'{stat}_min_10']
            
            # =================================================================
            # USAGE PROXY
            # =================================================================
            player_df['usage_proxy'] = (
                player_df['fga_avg_10'] + 
                0.44 * player_df.get('fta_avg_10', 0) + 
                player_df['ast_avg_10'] * 0.33 +
                player_df['turnover_avg_10']
            ) / (player_df['min_numeric_avg_10'] + 0.1)
            
            # =================================================================
            # HOME/AWAY SPLITS
            # =================================================================
            for stat in ['pts', 'reb', 'ast']:
                # Home performance
                home_mask = player_df['is_home'].shift(1) == 1
                player_df[f'{stat}_home_avg'] = player_df[stat].where(
                    player_df['is_home'].shift(1) == 1
                ).shift(1).expanding().mean()
                
                # Away performance  
                player_df[f'{stat}_away_avg'] = player_df[stat].where(
                    player_df['is_home'].shift(1) == 0
                ).shift(1).expanding().mean()
            
            # Day of week
            player_df['day_of_week'] = player_df['game_date'].dt.dayofweek
            
            feature_dfs.append(player_df)
        
        result = pd.concat(feature_dfs, ignore_index=True)
        
        # =================================================================
        # MERGE TEAM DEFENSE (opponent adjustments)
        # =================================================================
        if team_defense is not None:
            result = result.merge(
                team_defense[['team_id', 'game_date', 'pts_allowed_avg_10', 
                             'reb_allowed_avg_10', 'ast_allowed_avg_10', 'fg3m_allowed_avg_10']],
                left_on=['opponent_id', 'game_date'],
                right_on=['team_id', 'game_date'],
                how='left',
                suffixes=('', '_opp_def')
            )
            # Rename for clarity
            result = result.rename(columns={
                'pts_allowed_avg_10': 'opp_pts_allowed_avg',
                'reb_allowed_avg_10': 'opp_reb_allowed_avg',
                'ast_allowed_avg_10': 'opp_ast_allowed_avg',
                'fg3m_allowed_avg_10': 'opp_fg3m_allowed_avg'
            })
        
        # =================================================================
        # MERGE TEAM PACE
        # =================================================================
        if team_pace is not None:
            # Player's team pace
            result = result.merge(
                team_pace[['team_id', 'game_date', 'pace_avg_10']],
                on=['team_id', 'game_date'],
                how='left'
            )
            result = result.rename(columns={'pace_avg_10': 'team_pace'})
            
            # Opponent pace
            result = result.merge(
                team_pace[['team_id', 'game_date', 'pace_avg_10']],
                left_on=['opponent_id', 'game_date'],
                right_on=['team_id', 'game_date'],
                how='left',
                suffixes=('', '_opp')
            )
            result = result.rename(columns={'pace_avg_10': 'opp_pace'})
            
            # Expected game pace (average of both teams)
            result['expected_pace'] = (result['team_pace'].fillna(100) + result['opp_pace'].fillna(100)) / 2
        
        # Fill NaN
        result = result.fillna(0)
        
        # Drop duplicate columns
        result = result.loc[:, ~result.columns.duplicated()]
        
        print(f"  Created {len(result.columns)} columns")
        return result
    
    def get_feature_columns(self) -> List[str]:
        """Return list of feature columns for model input."""
        feature_cols = []
        
        # Rolling averages and stds
        for window in self.config.ROLLING_WINDOWS:
            for stat in self.stat_cols:
                feature_cols.append(f'{stat}_avg_{window}')
                feature_cols.append(f'{stat}_std_{window}')
        
        # EWMA features
        for span in self.config.EWMA_SPANS:
            for stat in ['pts', 'reb', 'ast', 'min_numeric', 'fga']:
                feature_cols.append(f'{stat}_ewma_{span}')
        
        # Season averages
        for stat in self.stat_cols:
            feature_cols.append(f'{stat}_season_avg')
        
        # Trends and momentum
        for stat in ['pts', 'reb', 'ast', 'min_numeric']:
            feature_cols.append(f'{stat}_trend')
            feature_cols.append(f'{stat}_momentum')
        
        # Per-minute rates
        for stat in ['pts', 'reb', 'ast', 'stl', 'blk', 'fg3m', 'fga']:
            feature_cols.append(f'{stat}_per_min_avg_10')
        
        # Rest and schedule
        feature_cols.extend(['days_rest', 'is_back_to_back', 'is_well_rested'])
        
        # Consistency
        for stat in ['pts', 'reb', 'ast']:
            feature_cols.append(f'{stat}_cv')
            feature_cols.append(f'{stat}_range_10')
        
        # Usage
        feature_cols.append('usage_proxy')
        
        # Home/away
        feature_cols.append('is_home')
        for stat in ['pts', 'reb', 'ast']:
            feature_cols.append(f'{stat}_home_avg')
            feature_cols.append(f'{stat}_away_avg')
        
        # Time
        feature_cols.extend(['day_of_week', 'player_game_num'])
        
        # Opponent adjustments (if available)
        feature_cols.extend([
            'opp_pts_allowed_avg', 'opp_reb_allowed_avg', 
            'opp_ast_allowed_avg', 'opp_fg3m_allowed_avg',
            'team_pace', 'opp_pace', 'expected_pace'
        ])
        
        return feature_cols


# Continue in next file...
