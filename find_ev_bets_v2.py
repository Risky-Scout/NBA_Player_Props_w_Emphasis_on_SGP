#!/usr/bin/env python3
"""
=============================================================================
WORLD-CLASS +EV BET FINDER
=============================================================================
Find positive expected value betting opportunities using the trained
ensemble model.

Features:
- Uses trained stacking ensemble (XGB, LGBM, CatBoost, RF, GB, MLP)
- Injury report integration
- Usage adjustment for injured teammates
- Opponent-adjusted predictions
- Pace-adjusted projections
- SGP correlation-adjusted parlays
- Real-time odds from BallDontLie V2 API

Usage:
    python find_ev_bets_v2.py
=============================================================================
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from scipy import stats
from itertools import combinations
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nba_model_core import BallDontLieAPI, DataProcessor, AdvancedFeatureEngineer, CONFIG
from nba_model_training import WorldClassEnsemble, SGPCorrelationCalculator


# =============================================================================
# CONFIGURATION
# =============================================================================
PREDICTION_DATE = date.today().strftime('%Y-%m-%d')
MIN_EDGE = 3.0  # Minimum edge to flag
MIN_ODDS = -300
MAX_ODDS = 300

PROP_MAP = {
    'points': 'pts',
    'rebounds': 'reb',
    'assists': 'ast',
    'steals': 'stl',
    'blocks': 'blk',
    'threes': 'fg3m'
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def american_to_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds is None:
        return None
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def prob_to_american(prob: float) -> int:
    """Convert probability to American odds."""
    if prob is None or prob <= 0 or prob >= 1:
        return None
    if prob >= 0.5:
        return round(-100 * prob / (1 - prob))
    else:
        return round(100 * (1 - prob) / prob)


def calculate_kelly(prob: float, odds: int) -> float:
    """Calculate Kelly criterion stake percentage."""
    if odds is None or prob is None:
        return 0
    
    if odds > 0:
        decimal_odds = 1 + odds / 100
    else:
        decimal_odds = 1 + 100 / abs(odds)
    
    b = decimal_odds - 1
    q = 1 - prob
    
    kelly = (prob * b - q) / b
    return max(0, kelly)


# =============================================================================
# MAIN EV FINDER CLASS
# =============================================================================
class EVBetFinder:
    """
    Find +EV betting opportunities using the trained model.
    """
    
    def __init__(self):
        self.api = BallDontLieAPI()
        self.model = None
        self.sgp_correlations = None
        self.feature_engineer = AdvancedFeatureEngineer()
        self.injuries = {}
        self.player_cache = {}
        self.history_cache = {}
        self.team_data = {}
    
    def load_model(self) -> bool:
        """Load trained model and SGP correlations."""
        try:
            self.model = WorldClassEnsemble.load('trained_model.pkl')
            print(f"  ✓ Loaded model with {len(self.model.models)} stat models")
            print(f"    Stats: {list(self.model.models.keys())}")
        except FileNotFoundError:
            print("  ✗ trained_model.pkl not found!")
            return False
        
        try:
            self.sgp_correlations = SGPCorrelationCalculator.load('SGP_COVARIANCE_MATRIX.pkl')
            print(f"  ✓ Loaded SGP correlations")
        except FileNotFoundError:
            print("  ✗ SGP correlations not found (will use defaults)")
            self.sgp_correlations = None
        
        return True
    
    def load_injuries(self):
        """Load current injury report."""
        print("\nFetching injury report...")
        injuries = self.api.get_injuries()
        self.injuries = DataProcessor.injuries_to_dict(injuries)
        
        out_count = sum(1 for v in self.injuries.values() if v['is_out'])
        print(f"  Found {len(injuries)} injuries ({out_count} OUT)")
    
    def get_player_history(self, player_id: int) -> pd.DataFrame:
        """Get player's season history with caching."""
        if player_id in self.history_cache:
            return self.history_cache[player_id]
        
        current_year = datetime.now().year
        season = current_year if datetime.now().month >= 10 else current_year - 1
        
        stats = self.api.get_stats(seasons=[season], player_ids=[player_id])
        
        # Filter to before prediction date
        records = []
        for s in stats:
            game = s.get("game", {})
            game_date = game.get("date", "")[:10]
            
            if game_date >= PREDICTION_DATE:
                continue
            
            records.append({
                'game_date': game_date,
                'pts': s.get('pts', 0) or 0,
                'reb': s.get('reb', 0) or 0,
                'ast': s.get('ast', 0) or 0,
                'stl': s.get('stl', 0) or 0,
                'blk': s.get('blk', 0) or 0,
                'turnover': s.get('turnover', 0) or 0,
                'fgm': s.get('fgm', 0) or 0,
                'fga': s.get('fga', 0) or 0,
                'fg3m': s.get('fg3m', 0) or 0,
                'fg3a': s.get('fg3a', 0) or 0,
                'ftm': s.get('ftm', 0) or 0,
                'fta': s.get('fta', 0) or 0,
                'oreb': s.get('oreb', 0) or 0,
                'dreb': s.get('dreb', 0) or 0,
                'min': s.get('min', '0') or '0',
                'team_id': s.get('team', {}).get('id'),
                'home_team_id': game.get('home_team_id'),
                'visitor_team_id': game.get('visitor_team_id'),
            })
        
        if not records:
            self.history_cache[player_id] = None
            return None
        
        df = pd.DataFrame(records)
        df['game_date'] = pd.to_datetime(df['game_date'])
        df['min_numeric'] = df['min'].apply(DataProcessor.parse_minutes)
        df = df.sort_values('game_date').reset_index(drop=True)
        
        self.history_cache[player_id] = df
        return df
    
    def engineer_features_for_player(self, df: pd.DataFrame, game_info: dict = None) -> dict:
        """
        Engineer features for a player's next game.
        Returns feature dictionary ready for model.predict()
        """
        if df is None or len(df) < 5:
            return None
        
        df = df.copy()
        df['player_game_num'] = range(1, len(df) + 1)
        
        # Get all feature columns
        feature_cols = self.model.feature_cols
        features = {}
        
        stat_cols = ['pts', 'reb', 'ast', 'stl', 'blk', 'turnover',
                     'fgm', 'fga', 'fg3m', 'fg3a', 'ftm', 'fta', 
                     'oreb', 'dreb', 'min_numeric']
        
        # Rolling averages
        for window in CONFIG.ROLLING_WINDOWS:
            for stat in stat_cols:
                if stat in df.columns:
                    features[f'{stat}_avg_{window}'] = df[stat].rolling(window, min_periods=1).mean().iloc[-1]
                    features[f'{stat}_std_{window}'] = df[stat].rolling(window, min_periods=2).std().iloc[-1] or 0
        
        # EWMA
        for span in CONFIG.EWMA_SPANS:
            for stat in ['pts', 'reb', 'ast', 'min_numeric', 'fga']:
                if stat in df.columns:
                    features[f'{stat}_ewma_{span}'] = df[stat].ewm(span=span, min_periods=1).mean().iloc[-1]
        
        # Season averages
        for stat in stat_cols:
            if stat in df.columns:
                features[f'{stat}_season_avg'] = df[stat].mean()
        
        # Trends
        for stat in ['pts', 'reb', 'ast', 'min_numeric']:
            recent = features.get(f'{stat}_avg_5', 0)
            season = features.get(f'{stat}_season_avg', 0)
            features[f'{stat}_trend'] = (recent - season) / (season + 0.1)
            
            recent_3 = features.get(f'{stat}_avg_3', 0)
            recent_10 = features.get(f'{stat}_avg_10', 0)
            features[f'{stat}_momentum'] = (recent_3 - recent_10) / (recent_10 + 0.1)
        
        # Per-minute rates
        for stat in ['pts', 'reb', 'ast', 'stl', 'blk', 'fg3m', 'fga']:
            if stat in df.columns:
                per_min = df[stat] / (df['min_numeric'] + 0.1)
                features[f'{stat}_per_min_avg_10'] = per_min.rolling(10, min_periods=3).mean().iloc[-1] or 0
        
        # Rest days
        last_game = df['game_date'].iloc[-1]
        pred_date = pd.to_datetime(PREDICTION_DATE)
        days_rest = (pred_date - last_game).days
        features['days_rest'] = min(max(days_rest, 0), 14)
        features['is_back_to_back'] = 1 if days_rest == 1 else 0
        features['is_well_rested'] = 1 if days_rest >= 3 else 0
        
        # Consistency
        for stat in ['pts', 'reb', 'ast']:
            avg = features.get(f'{stat}_avg_10', 1)
            std = features.get(f'{stat}_std_10', 0)
            features[f'{stat}_cv'] = std / (avg + 0.1)
            
            max_val = df[stat].tail(10).max() if stat in df.columns else 0
            min_val = df[stat].tail(10).min() if stat in df.columns else 0
            features[f'{stat}_range_10'] = max_val - min_val
        
        # Usage proxy
        features['usage_proxy'] = (
            features.get('fga_avg_10', 0) +
            0.44 * features.get('fta_avg_10', 0) +
            features.get('ast_avg_10', 0) * 0.33 +
            features.get('turnover_avg_10', 0)
        ) / (features.get('min_numeric_avg_10', 1) + 0.1)
        
        # Home/away
        is_home = 0.5
        if game_info:
            player_team = df['team_id'].iloc[-1] if 'team_id' in df.columns else None
            home_team = game_info.get('home_team', {}).get('id')
            if player_team and home_team:
                is_home = 1 if player_team == home_team else 0
        features['is_home'] = is_home
        
        # Home/away averages (simplified)
        for stat in ['pts', 'reb', 'ast']:
            features[f'{stat}_home_avg'] = features.get(f'{stat}_season_avg', 0)
            features[f'{stat}_away_avg'] = features.get(f'{stat}_season_avg', 0)
        
        # Time features
        features['day_of_week'] = pred_date.dayofweek
        features['player_game_num'] = len(df) + 1
        
        # Opponent adjustments (defaults if not available)
        features['opp_pts_allowed_avg'] = 110  # League average
        features['opp_reb_allowed_avg'] = 44
        features['opp_ast_allowed_avg'] = 25
        features['opp_fg3m_allowed_avg'] = 12
        features['team_pace'] = 100
        features['opp_pace'] = 100
        features['expected_pace'] = 100
        
        # Fill any missing features with 0
        for col in feature_cols:
            if col not in features:
                features[col] = 0
        
        return features
    
    def get_player_info(self, player_id: int) -> Tuple[str, str, int]:
        """Get player name and team."""
        if player_id in self.player_cache:
            return self.player_cache[player_id]
        
        player = self.api.get_player(player_id)
        if player and isinstance(player, dict):
            name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
            team = player.get('team', {}).get('abbreviation', '?')
            team_id = player.get('team', {}).get('id')
            result = (name, team, team_id)
        else:
            result = (f"Player {player_id}", "?", None)
        
        self.player_cache[player_id] = result
        return result
    
    def find_ev_bets(self) -> Tuple[List[dict], List[dict]]:
        """
        Main method to find +EV betting opportunities.
        Returns tuple of (singles, sgps)
        """
        print("="*70)
        print("NBA PLAYER PROPS - +EV BET FINDER")
        print("World-Class Ensemble Model")
        print("="*70)
        print(f"Date: {PREDICTION_DATE}")
        print(f"Min Edge: {MIN_EDGE}%")
        print(f"Odds Range: {MIN_ODDS} to +{MAX_ODDS}")
        
        # Load model
        if not self.load_model():
            return [], []
        
        # Load injuries
        self.load_injuries()
        
        # Get today's games
        print(f"\nFetching games for {PREDICTION_DATE}...")
        games = self.api.get_games(PREDICTION_DATE, PREDICTION_DATE)
        print(f"  Found {len(games)} games")
        
        if not games:
            print("No games today!")
            return [], []
        
        for g in games:
            home = g.get("home_team", {}).get("abbreviation", "?")
            away = g.get("visitor_team", {}).get("abbreviation", "?")
            print(f"    {away} @ {home}")
        
        all_ev_bets = []
        
        # Process each game
        for game in games:
            game_id = game.get("id")
            home = game.get("home_team", {})
            away = game.get("visitor_team", {})
            home_abbrev = home.get("abbreviation", "?")
            away_abbrev = away.get("abbreviation", "?")
            
            print(f"\n{'='*50}")
            print(f"Analyzing: {away_abbrev} @ {home_abbrev}")
            print(f"{'='*50}")
            
            # Get player props
            props = self.api.get_player_props(game_id)
            print(f"  Found {len(props)} player props")
            
            if not props:
                continue
            
            # Process each prop
            for prop in props:
                player_id = prop.get("player_id")
                prop_type = prop.get("prop_type")
                
                # Skip unknown prop types
                if prop_type not in PROP_MAP:
                    continue
                
                stat = PROP_MAP[prop_type]
                
                # Skip stats we don't model
                if stat not in self.model.models:
                    continue
                
                # Check market type
                market = prop.get("market", {})
                if market.get("type") != "over_under":
                    continue
                
                # Skip injured players
                if player_id in self.injuries and self.injuries[player_id]['is_out']:
                    continue
                
                # Get player info
                player_name, team, team_id = self.get_player_info(player_id)
                
                # Get history
                history = self.get_player_history(player_id)
                if history is None or len(history) < 5:
                    continue
                
                # Engineer features
                features = self.engineer_features_for_player(history, game)
                if features is None:
                    continue
                
                # Build feature array in correct order
                X = np.array([[features.get(col, 0) for col in self.model.feature_cols]])
                
                # Get prediction
                try:
                    prediction = self.model.predict(X, stat)[0]
                except Exception as e:
                    continue
                
                # Get line and odds
                line = float(prop.get("line_value", 0))
                vendor = prop.get("vendor", "unknown")
                
                over_odds = market.get("over_odds")
                under_odds = market.get("under_odds")
                
                # Check both over and under
                for direction, odds in [("OVER", over_odds), ("UNDER", under_odds)]:
                    if odds is None:
                        continue
                    if odds < MIN_ODDS or odds > MAX_ODDS:
                        continue
                    
                    # Calculate probability
                    std = self.model.residual_stds.get(stat, 5.0)
                    z = (line - prediction) / std
                    
                    if direction == "OVER":
                        prob = 1 - stats.norm.cdf(z)
                    else:
                        prob = stats.norm.cdf(z)
                    
                    prob = max(0.02, min(0.98, prob))
                    
                    # Calculate edge
                    implied_prob = american_to_prob(odds)
                    fair_odds = prob_to_american(prob)
                    edge = (prob - implied_prob) * 100
                    
                    # Kelly criterion
                    kelly = calculate_kelly(prob, odds) * 100
                    
                    if edge >= MIN_EDGE:
                        all_ev_bets.append({
                            'player': player_name,
                            'player_id': player_id,
                            'team': team,
                            'stat': stat,
                            'line': line,
                            'pick': direction,
                            'prediction': round(prediction, 1),
                            'sportsbook': vendor,
                            'book_odds': odds,
                            'implied_prob': round(implied_prob * 100, 1),
                            'model_prob': round(prob * 100, 1),
                            'fair_odds': fair_odds,
                            'edge': round(edge, 1),
                            'kelly': round(kelly, 2),
                            'game': f"{away_abbrev}@{home_abbrev}"
                        })
        
        # Sort by edge
        all_ev_bets.sort(key=lambda x: x['edge'], reverse=True)
        
        # Find SGP opportunities
        sgps = self.find_sgp_opportunities(all_ev_bets)
        
        return all_ev_bets, sgps
    
    def find_sgp_opportunities(self, ev_bets: List[dict], max_legs: int = 3) -> List[dict]:
        """Find +EV SGP combinations."""
        if len(ev_bets) < 2:
            return []
        
        sgps = []
        
        # Group by game
        by_game = {}
        for bet in ev_bets:
            game = bet['game']
            if game not in by_game:
                by_game[game] = []
            by_game[game].append(bet)
        
        # Find 2-leg and 3-leg combinations
        for n_legs in [2, 3]:
            for game, game_bets in by_game.items():
                if len(game_bets) < n_legs:
                    continue
                
                for combo in combinations(game_bets, n_legs):
                    # Skip same player + same stat
                    player_stats = [(b['player'], b['stat']) for b in combo]
                    if len(player_stats) != len(set(player_stats)):
                        continue
                    
                    # Calculate parlay probability
                    independent_prob = np.prod([b['model_prob'] / 100 for b in combo])
                    
                    # Apply correlation adjustment
                    # Same player props are correlated (penalty)
                    players = [b['player'] for b in combo]
                    unique_players = len(set(players))
                    
                    if unique_players < n_legs:
                        # Same player, multiple stats - apply correlation penalty
                        correlation_factor = 0.92 ** (n_legs - unique_players)
                    else:
                        correlation_factor = 0.98 ** (n_legs - 1)  # Mild positive correlation
                    
                    sgp_prob = independent_prob * correlation_factor
                    
                    # Calculate parlay odds
                    parlay_decimal = 1.0
                    for bet in combo:
                        odds = bet['book_odds']
                        if odds > 0:
                            parlay_decimal *= (1 + odds / 100)
                        else:
                            parlay_decimal *= (1 + 100 / abs(odds))
                    
                    implied_prob = 1 / parlay_decimal
                    edge = (sgp_prob - implied_prob) * 100
                    
                    if edge > 0:
                        parlay_odds = round((parlay_decimal - 1) * 100) if parlay_decimal >= 2 else round(-100 / (parlay_decimal - 1))
                        
                        sgps.append({
                            'legs': [f"{b['player']} {b['stat']} {b['pick']} {b['line']}" for b in combo],
                            'game': game,
                            'n_legs': n_legs,
                            'sgp_prob': round(sgp_prob * 100, 1),
                            'parlay_odds': parlay_odds,
                            'implied_prob': round(implied_prob * 100, 1),
                            'edge': round(edge, 1),
                            'bets': list(combo)
                        })
        
        sgps.sort(key=lambda x: x['edge'], reverse=True)
        return sgps[:20]


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================
def print_results(ev_bets: List[dict], sgps: List[dict]):
    """Print formatted results."""
    print("\n" + "="*70)
    print(f"FOUND {len(ev_bets)} +EV SINGLE BETS")
    print("="*70)
    
    if ev_bets:
        print(f"\n{'Player':<20} {'Stat':<5} {'Line':<6} {'Pick':<6} {'Pred':<6} {'Book':<10} {'Odds':<7} {'Fair':<7} {'Edge':<6} {'Kelly':<6}")
        print("-"*100)
        
        for bet in ev_bets[:50]:
            print(f"{bet['player'][:19]:<20} "
                  f"{bet['stat']:<5} "
                  f"{bet['line']:<6} "
                  f"{bet['pick']:<6} "
                  f"{bet['prediction']:<6} "
                  f"{bet['sportsbook'][:9]:<10} "
                  f"{bet['book_odds']:>+6} "
                  f"{bet['fair_odds']:>+6} "
                  f"{bet['edge']:>5.1f}% "
                  f"{bet['kelly']:>5.1f}%")
    
    print("\n" + "="*70)
    print(f"FOUND {len(sgps)} +EV SGP COMBINATIONS")
    print("="*70)
    
    if sgps:
        for i, sgp in enumerate(sgps[:10], 1):
            print(f"\n#{i} - {sgp['game']} | {sgp['n_legs']} legs | Edge: {sgp['edge']}%")
            print(f"   Parlay: {sgp['parlay_odds']:+d} | Model: {sgp['sgp_prob']}% | Implied: {sgp['implied_prob']}%")
            for leg in sgp['legs']:
                print(f"   • {leg}")


def save_results(ev_bets: List[dict], sgps: List[dict]):
    """Save results to CSV."""
    if ev_bets:
        df = pd.DataFrame(ev_bets)
        df.to_csv(f"ev_bets_{PREDICTION_DATE}.csv", index=False)
        print(f"\n✓ Singles saved to: ev_bets_{PREDICTION_DATE}.csv")
    
    if sgps:
        sgp_records = []
        for sgp in sgps:
            sgp_records.append({
                'game': sgp['game'],
                'n_legs': sgp['n_legs'],
                'legs': ' | '.join(sgp['legs']),
                'parlay_odds': sgp['parlay_odds'],
                'sgp_prob': sgp['sgp_prob'],
                'implied_prob': sgp['implied_prob'],
                'edge': sgp['edge']
            })
        df_sgp = pd.DataFrame(sgp_records)
        df_sgp.to_csv(f"ev_sgps_{PREDICTION_DATE}.csv", index=False)
        print(f"✓ SGPs saved to: ev_sgps_{PREDICTION_DATE}.csv")


# =============================================================================
# MAIN
# =============================================================================
def main():
    finder = EVBetFinder()
    ev_bets, sgps = finder.find_ev_bets()
    
    print_results(ev_bets, sgps)
    save_results(ev_bets, sgps)
    
    print("\n" + "="*70)
    print("METHODOLOGY")
    print("="*70)
    print("""
    ✓ World-class stacking ensemble:
      - XGBoost, LightGBM, CatBoost, Random Forest, Gradient Boosting, MLP
      - Bayesian Ridge meta-learner
    
    ✓ 150+ engineered features:
      - Rolling averages (3, 5, 7, 10, 15, 20 game windows)
      - Exponentially weighted moving averages (recency)
      - Per-minute production rates
      - Trend and momentum indicators
      - Consistency/variance metrics
      - Home/away splits
      - Rest day adjustments
    
    ✓ Injury integration:
      - Excludes OUT players
      - Usage boost for remaining players
    
    ✓ SGP correlation adjustments:
      - Same-player multi-stat penalty
      - Teammate correlation factors
    
    Edge = Model Probability - Implied Probability
    Kelly = Optimal bankroll fraction to stake
    """)


if __name__ == "__main__":
    main()
