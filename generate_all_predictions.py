#!/usr/bin/env python3
"""
NBA Player Props - Complete Prediction Generator
=================================================
Generates:
1. Singles picks with probabilities AND fair odds
2. Two-leg parlays (SGP)
3. Three-leg parlays (SGP)

Output files (in predictions/ folder):
- singles_YYYY-MM-DD.csv
- best_singles_YYYY-MM-DD.csv
- fair_odds_YYYY-MM-DD.csv
- parlays_2leg_YYYY-MM-DD.csv
- parlays_3leg_YYYY-MM-DD.csv

Usage:
    python3 generate_all_predictions.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import requests
import time
from scipy import stats
from itertools import combinations
import warnings
import os

warnings.filterwarnings('ignore')

# Configuration
API_BASE_URL = "https://api.balldontlie.io/v1"
API_KEY = os.environ.get("BALLDONTLIE_API_KEY", "1340a2ff-7054-4504-b5b4-96e63281e062")
PREDICTION_DATE = os.environ.get("PREDICTION_DATE", date.today().strftime('%Y-%m-%d'))

STAT_TARGETS = ['pts', 'reb', 'ast', 'stl', 'blk']

STANDARD_LINES = {
    'pts': [9.5, 12.5, 14.5, 17.5, 19.5, 21.5, 24.5, 27.5, 29.5, 32.5, 34.5],
    'reb': [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5],
    'ast': [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
    'stl': [0.5, 1.5, 2.5],
    'blk': [0.5, 1.5, 2.5, 3.5],
}


def api_request(endpoint, params=None):
    """Make API request with rate limiting."""
    time.sleep(1.2)
    headers = {"Authorization": API_KEY}
    try:
        resp = requests.get(f"{API_BASE_URL}/{endpoint}", headers=headers, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"    Request failed: {e}")
    return {"data": []}


def probability_to_american_odds(prob):
    """
    Convert probability to American odds.
    
    For odds makers:
    - Negative odds (favorites): How much to bet to win $100
    - Positive odds (underdogs): How much you win on $100 bet
    
    Formula:
    - If prob > 0.5: odds = -(prob / (1 - prob)) * 100
    - If prob < 0.5: odds = ((1 - prob) / prob) * 100
    """
    if prob >= 0.99:
        return -10000
    if prob <= 0.01:
        return 10000
    
    if prob > 0.5:
        odds = -(prob / (1 - prob)) * 100
        return int(round(odds))
    elif prob < 0.5:
        odds = ((1 - prob) / prob) * 100
        return int(round(odds))
    else:
        return -100


def probability_to_decimal_odds(prob):
    """Convert probability to decimal odds."""
    if prob <= 0:
        return 999.99
    return round(1 / prob, 2)


def american_odds_to_string(odds):
    """Format American odds with + or - sign."""
    if odds > 0:
        return f"+{odds}"
    else:
        return str(odds)


def calculate_parlay_probability(probs):
    """Calculate combined probability for a parlay."""
    result = 1.0
    for p in probs:
        result *= p
    return result


def get_todays_games():
    """Get today's games."""
    print(f"\nFetching games for {PREDICTION_DATE}...")
    games = api_request("games", {"start_date": PREDICTION_DATE, "end_date": PREDICTION_DATE}).get("data", [])
    print(f"  Found {len(games)} games")
    return games


def get_recent_stats_for_season():
    """Get all recent stats to find active players."""
    print("\nFetching recent player stats...")
    
    current_year = datetime.now().year
    season = current_year if datetime.now().month >= 10 else current_year - 1
    
    pred_date = datetime.strptime(PREDICTION_DATE, '%Y-%m-%d').date()
    end_date = (pred_date - timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = (pred_date - timedelta(days=14)).strftime('%Y-%m-%d')
    
    all_stats = []
    cursor = None
    
    while True:
        params = {
            "seasons[]": season,
            "start_date": start_date,
            "end_date": end_date,
            "per_page": 100
        }
        if cursor:
            params["cursor"] = cursor
        
        resp = api_request("stats", params)
        data = resp.get("data", [])
        
        if not data:
            break
        
        all_stats.extend(data)
        print(f"    Fetched {len(all_stats)} stat lines...")
        
        cursor = resp.get("meta", {}).get("next_cursor")
        if not cursor:
            break
    
    print(f"  Total recent stats: {len(all_stats)}")
    return all_stats


def get_player_full_history(player_id):
    """Get full season history for a player."""
    current_year = datetime.now().year
    season = current_year if datetime.now().month >= 10 else current_year - 1
    
    stats = api_request("stats", {
        "seasons[]": season,
        "player_ids[]": player_id,
        "per_page": 100
    }).get("data", [])
    
    records = []
    for s in stats:
        game_date = s.get("game", {}).get("date", "")[:10]
        if game_date >= PREDICTION_DATE:
            continue
        
        mins = s.get("min", "0:00")
        try:
            if ':' in str(mins):
                min_val = float(str(mins).split(':')[0]) + float(str(mins).split(':')[1])/60
            else:
                min_val = float(mins) if mins else 0
        except:
            min_val = 0
        
        if min_val < 5:
            continue
        
        records.append({
            'game_date': game_date,
            'pts': s.get('pts') or 0,
            'reb': s.get('reb') or 0,
            'ast': s.get('ast') or 0,
            'stl': s.get('stl') or 0,
            'blk': s.get('blk') or 0,
            'min': min_val,
        })
    
    df = pd.DataFrame(records)
    if len(df) > 0:
        df = df.sort_values('game_date')
    return df


def calculate_prediction(player_stats, stat):
    """Calculate prediction with probability."""
    if len(player_stats) < 5:
        return None
    
    values = player_stats[stat].values
    
    weights = np.exp(np.linspace(-1, 0, len(values)))
    weights /= weights.sum()
    point_estimate = np.average(values, weights=weights)
    
    std_dev = max(values.std(), 0.5)
    
    if len(values) >= 5:
        recent = values[-3:].mean()
        overall = values.mean()
        trend = recent - overall
        point_estimate += 0.3 * trend
    
    return {
        'prediction': round(point_estimate, 1),
        'std_dev': round(std_dev, 2),
        'avg_3': round(values[-3:].mean(), 1) if len(values) >= 3 else None,
        'avg_5': round(values[-5:].mean(), 1) if len(values) >= 5 else None,
        'avg_10': round(values[-10:].mean(), 1) if len(values) >= 10 else None,
        'season_avg': round(values.mean(), 1),
        'games': len(values),
    }


def over_probability(pred, std, line):
    """Calculate over probability."""
    z = (line - pred) / std
    return round(1 - stats.norm.cdf(z), 4)


def generate_singles(players_data):
    """Generate all singles predictions with fair odds."""
    print("\nGenerating singles predictions...")
    
    all_singles = []
    
    for player_id, player_info in players_data.items():
        player_name = player_info['name']
        team = player_info['team']
        opponent = player_info['opponent']
        game = player_info['game']
        history = player_info['history']
        
        if len(history) < 5:
            continue
        
        for stat in STAT_TARGETS:
            pred_data = calculate_prediction(history, stat)
            if pred_data is None:
                continue
            
            projection = pred_data['prediction']
            lines = STANDARD_LINES.get(stat, [])
            
            for line in lines:
                over_prob = over_probability(pred_data['prediction'], pred_data['std_dev'], line)
                under_prob = 1 - over_prob
                
                if over_prob < 0.10 or over_prob > 0.90:
                    continue
                
                if over_prob > under_prob:
                    pick = 'OVER'
                    prob = over_prob
                else:
                    pick = 'UNDER'
                    prob = under_prob
                
                edge = prob - 0.524
                
                # Fair odds calculations
                over_odds = probability_to_american_odds(over_prob)
                under_odds = probability_to_american_odds(under_prob)
                pick_odds = over_odds if pick == 'OVER' else under_odds
                
                over_decimal = probability_to_decimal_odds(over_prob)
                under_decimal = probability_to_decimal_odds(under_prob)
                
                all_singles.append({
                    'date': PREDICTION_DATE,
                    'game': game,
                    'player': player_name,
                    'team': team,
                    'opponent': opponent,
                    'stat': stat.upper(),
                    'line': line,
                    'pick': pick,
                    'probability': round(prob, 4),
                    'probability_pct': f"{prob*100:.1f}%",
                    'edge': round(edge, 4),
                    'edge_pct': f"{edge*100:.1f}%",
                    'fair_odds_american': american_odds_to_string(pick_odds),
                    'fair_odds_decimal': probability_to_decimal_odds(prob),
                    'over_prob': round(over_prob, 4),
                    'under_prob': round(under_prob, 4),
                    'over_fair_american': american_odds_to_string(over_odds),
                    'under_fair_american': american_odds_to_string(under_odds),
                    'over_fair_decimal': over_decimal,
                    'under_fair_decimal': under_decimal,
                    'projection': pred_data['prediction'],
                    'avg_5': pred_data['avg_5'],
                    'season_avg': pred_data['season_avg'],
                    'games_sample': pred_data['games'],
                })
    
    df = pd.DataFrame(all_singles)
    if len(df) > 0:
        df = df.sort_values('edge', ascending=False)
    
    print(f"  Generated {len(df)} singles predictions")
    return df


def generate_2leg_parlays(singles_df, min_edge=0.03, max_parlays=100):
    """Generate 2-leg SGP parlays."""
    print("\nGenerating 2-leg parlays (SGP)...")
    
    if len(singles_df) == 0:
        return pd.DataFrame()
    
    good_picks = singles_df[singles_df['edge'] >= min_edge].copy()
    parlays = []
    
    for game in good_picks['game'].unique():
        game_picks = good_picks[good_picks['game'] == game]
        
        if len(game_picks) < 2:
            continue
        
        for (idx1, pick1), (idx2, pick2) in combinations(game_picks.iterrows(), 2):
            if pick1['player'] == pick2['player'] and pick1['stat'] == pick2['stat']:
                continue
            
            probs = [pick1['probability'], pick2['probability']]
            parlay_prob = calculate_parlay_probability(probs)
            parlay_american = probability_to_american_odds(parlay_prob)
            parlay_decimal = probability_to_decimal_odds(parlay_prob)
            
            implied_break_even = 0.524 * 0.524
            parlay_edge = parlay_prob - implied_break_even
            
            leg1 = f"{pick1['player']} {pick1['stat']} {pick1['pick']} {pick1['line']}"
            leg2 = f"{pick2['player']} {pick2['stat']} {pick2['pick']} {pick2['line']}"
            
            parlays.append({
                'date': PREDICTION_DATE,
                'game': game,
                'legs': 2,
                'leg_1': leg1,
                'leg_1_prob': round(pick1['probability'], 3),
                'leg_1_fair_odds': pick1['fair_odds_american'],
                'leg_2': leg2,
                'leg_2_prob': round(pick2['probability'], 3),
                'leg_2_fair_odds': pick2['fair_odds_american'],
                'parlay_probability': round(parlay_prob, 4),
                'parlay_prob_pct': f"{parlay_prob*100:.1f}%",
                'parlay_fair_american': american_odds_to_string(parlay_american),
                'parlay_fair_decimal': parlay_decimal,
                'parlay_edge': round(parlay_edge, 4),
            })
    
    df = pd.DataFrame(parlays)
    if len(df) > 0:
        df = df.sort_values('parlay_probability', ascending=False).head(max_parlays)
    
    print(f"  Generated {len(df)} 2-leg parlays")
    return df


def generate_3leg_parlays(singles_df, min_edge=0.02, max_parlays=100):
    """Generate 3-leg SGP parlays."""
    print("\nGenerating 3-leg parlays (SGP)...")
    
    if len(singles_df) == 0:
        return pd.DataFrame()
    
    good_picks = singles_df[singles_df['edge'] >= min_edge].copy()
    parlays = []
    
    for game in good_picks['game'].unique():
        game_picks = good_picks[good_picks['game'] == game]
        
        if len(game_picks) < 3:
            continue
        
        for (idx1, pick1), (idx2, pick2), (idx3, pick3) in combinations(game_picks.iterrows(), 3):
            players_stats = [
                (pick1['player'], pick1['stat']),
                (pick2['player'], pick2['stat']),
                (pick3['player'], pick3['stat']),
            ]
            if len(set(players_stats)) != 3:
                continue
            
            probs = [pick1['probability'], pick2['probability'], pick3['probability']]
            parlay_prob = calculate_parlay_probability(probs)
            parlay_american = probability_to_american_odds(parlay_prob)
            parlay_decimal = probability_to_decimal_odds(parlay_prob)
            
            implied_break_even = 0.524 ** 3
            parlay_edge = parlay_prob - implied_break_even
            
            leg1 = f"{pick1['player']} {pick1['stat']} {pick1['pick']} {pick1['line']}"
            leg2 = f"{pick2['player']} {pick2['stat']} {pick2['pick']} {pick2['line']}"
            leg3 = f"{pick3['player']} {pick3['stat']} {pick3['pick']} {pick3['line']}"
            
            parlays.append({
                'date': PREDICTION_DATE,
                'game': game,
                'legs': 3,
                'leg_1': leg1,
                'leg_1_prob': round(pick1['probability'], 3),
                'leg_1_fair_odds': pick1['fair_odds_american'],
                'leg_2': leg2,
                'leg_2_prob': round(pick2['probability'], 3),
                'leg_2_fair_odds': pick2['fair_odds_american'],
                'leg_3': leg3,
                'leg_3_prob': round(pick3['probability'], 3),
                'leg_3_fair_odds': pick3['fair_odds_american'],
                'parlay_probability': round(parlay_prob, 4),
                'parlay_prob_pct': f"{parlay_prob*100:.1f}%",
                'parlay_fair_american': american_odds_to_string(parlay_american),
                'parlay_fair_decimal': parlay_decimal,
                'parlay_edge': round(parlay_edge, 4),
            })
    
    df = pd.DataFrame(parlays)
    if len(df) > 0:
        df = df.sort_values('parlay_probability', ascending=False).head(max_parlays)
    
    print(f"  Generated {len(df)} 3-leg parlays")
    return df


def main():
    print("="*70)
    print(f"NBA PLAYER PROPS - PREDICTIONS FOR {PREDICTION_DATE}")
    print("="*70)
    print("\nOutputs:")
    print("  • Singles with probabilities and fair odds")
    print("  • 2-leg SGP parlays")
    print("  • 3-leg SGP parlays")
    print()
    
    games = get_todays_games()
    if not games:
        print("No games today!")
        return
    
    for g in games:
        home = g.get('home_team', {}).get('abbreviation', '?')
        away = g.get('visitor_team', {}).get('abbreviation', '?')
        print(f"  {away} @ {home}")
    
    team_ids = set()
    team_info = {}
    
    for g in games:
        home_id = g.get('home_team', {}).get('id')
        away_id = g.get('visitor_team', {}).get('id')
        home_abbr = g.get('home_team', {}).get('abbreviation')
        away_abbr = g.get('visitor_team', {}).get('abbreviation')
        game_str = f"{away_abbr} @ {home_abbr}"
        
        if home_id:
            team_ids.add(home_id)
            team_info[home_id] = {'abbr': home_abbr, 'opponent': away_abbr, 'game': game_str}
        if away_id:
            team_ids.add(away_id)
            team_info[away_id] = {'abbr': away_abbr, 'opponent': home_abbr, 'game': game_str}
    
    recent_stats = get_recent_stats_for_season()
    
    players = {}
    for s in recent_stats:
        team_id = s.get('team', {}).get('id')
        if team_id not in team_ids:
            continue
        
        player_id = s.get('player', {}).get('id')
        player_name = f"{s.get('player', {}).get('first_name', '')} {s.get('player', {}).get('last_name', '')}".strip()
        
        mins = s.get('min', '0:00')
        try:
            if ':' in str(mins):
                min_val = float(str(mins).split(':')[0])
            else:
                min_val = float(mins) if mins else 0
        except:
            min_val = 0
        
        if min_val < 10:
            continue
        
        if player_id not in players:
            players[player_id] = {'name': player_name, 'team_id': team_id, 'games': 0, 'total_pts': 0}
        
        players[player_id]['games'] += 1
        players[player_id]['total_pts'] += s.get('pts', 0) or 0
    
    active_players = {pid: p for pid, p in players.items() if p['games'] >= 3}
    print(f"\nFound {len(active_players)} active players")
    
    players_data = {}
    sorted_players = sorted(active_players.items(), key=lambda x: x[1]['total_pts'], reverse=True)
    
    for player_id, player_info in sorted_players[:80]:
        team_id = player_info['team_id']
        team_data = team_info.get(team_id, {})
        
        print(f"  {player_info['name']} ({team_data.get('abbr', '?')})...")
        
        history = get_player_full_history(player_id)
        
        if len(history) < 5:
            continue
        
        players_data[player_id] = {
            'name': player_info['name'],
            'team': team_data.get('abbr', '?'),
            'opponent': team_data.get('opponent', '?'),
            'game': team_data.get('game', '?'),
            'history': history,
        }
    
    # Generate all predictions
    singles_df = generate_singles(players_data)
    parlays_2leg_df = generate_2leg_parlays(singles_df)
    parlays_3leg_df = generate_3leg_parlays(singles_df)
    
    # Create output directory
    os.makedirs('predictions', exist_ok=True)
    
    # Save files
    if len(singles_df) > 0:
        singles_df.to_csv(f"predictions/singles_{PREDICTION_DATE}.csv", index=False)
        print(f"\n✓ Saved singles to predictions/singles_{PREDICTION_DATE}.csv")
        
        best_singles = singles_df[singles_df['edge'] >= 0.05]
        if len(best_singles) > 0:
            best_singles.to_csv(f"predictions/best_singles_{PREDICTION_DATE}.csv", index=False)
            print(f"✓ Saved best singles to predictions/best_singles_{PREDICTION_DATE}.csv")
        
        # Fair odds version (for odds maker study)
        fair_odds_cols = ['date', 'game', 'player', 'team', 'stat', 'line', 
                         'over_fair_american', 'under_fair_american', 
                         'over_fair_decimal', 'under_fair_decimal',
                         'over_prob', 'under_prob', 'projection']
        fair_odds_df = singles_df[fair_odds_cols].drop_duplicates()
        fair_odds_df.to_csv(f"predictions/fair_odds_{PREDICTION_DATE}.csv", index=False)
        print(f"✓ Saved fair odds to predictions/fair_odds_{PREDICTION_DATE}.csv")
    
    if len(parlays_2leg_df) > 0:
        parlays_2leg_df.to_csv(f"predictions/parlays_2leg_{PREDICTION_DATE}.csv", index=False)
        print(f"✓ Saved 2-leg parlays to predictions/parlays_2leg_{PREDICTION_DATE}.csv")
    
    if len(parlays_3leg_df) > 0:
        parlays_3leg_df.to_csv(f"predictions/parlays_3leg_{PREDICTION_DATE}.csv", index=False)
        print(f"✓ Saved 3-leg parlays to predictions/parlays_3leg_{PREDICTION_DATE}.csv")
    
    # Print summary
    print("\n" + "="*70)
    print("TOP 10 SINGLES")
    print("="*70)
    
    if len(singles_df) > 0:
        for _, row in singles_df.head(10).iterrows():
            stars = "⭐⭐⭐" if row['edge'] >= 0.08 else "⭐⭐" if row['edge'] >= 0.05 else "⭐"
            print(f"\n{stars} {row['player']} {row['stat']} {row['pick']} {row['line']}")
            print(f"   Prob: {row['probability_pct']} | Fair Odds: {row['fair_odds_american']} ({row['fair_odds_decimal']})")
    
    print("\n" + "="*70)
    print("TOP 5 2-LEG PARLAYS")
    print("="*70)
    
    if len(parlays_2leg_df) > 0:
        for _, row in parlays_2leg_df.head(5).iterrows():
            print(f"\n🎯 {row['game']}")
            print(f"   {row['leg_1']}")
            print(f"   {row['leg_2']}")
            print(f"   Parlay: {row['parlay_prob_pct']} | Fair: {row['parlay_fair_american']}")
    
    print("\n" + "="*70)
    print("TOP 5 3-LEG PARLAYS")
    print("="*70)
    
    if len(parlays_3leg_df) > 0:
        for _, row in parlays_3leg_df.head(5).iterrows():
            print(f"\n🎯 {row['game']}")
            print(f"   {row['leg_1']}")
            print(f"   {row['leg_2']}")
            print(f"   {row['leg_3']}")
            print(f"   Parlay: {row['parlay_prob_pct']} | Fair: {row['parlay_fair_american']}")
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == "__main__":
    main()
