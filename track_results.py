#!/usr/bin/env python3
"""
NBA Player Props - Prediction Tracker & Grader
===============================================
Track model performance over time.

Records:
- Every prediction made
- Actual results
- Win/Loss/Push for each prediction
- Performance by probability bucket
- Overall accuracy and calibration

Usage:
    # After games are complete:
    python3 track_results.py grade

    # View performance:
    python3 track_results.py report

    # Export full history:
    python3 track_results.py export
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import requests
import time
import os
import sys
import json
import warnings

warnings.filterwarnings('ignore')

# Configuration
API_BASE_URL = "https://api.balldontlie.io/v1"
API_KEY = os.environ.get("BALLDONTLIE_API_KEY", "1340a2ff-7054-4504-b5b4-96e63281e062")

TRACKING_FILE = "prediction_history.csv"
RESULTS_FILE = "graded_results.csv"


class PredictionTracker:
    """Track and grade predictions."""
    
    def __init__(self):
        self.headers = {"Authorization": API_KEY}
        self.history = self._load_history()
        self.results = self._load_results()
    
    def _load_history(self) -> pd.DataFrame:
        """Load prediction history."""
        if os.path.exists(TRACKING_FILE):
            return pd.read_csv(TRACKING_FILE)
        return pd.DataFrame()
    
    def _load_results(self) -> pd.DataFrame:
        """Load graded results."""
        if os.path.exists(RESULTS_FILE):
            return pd.read_csv(RESULTS_FILE)
        return pd.DataFrame()
    
    def _save_history(self):
        """Save prediction history."""
        self.history.to_csv(TRACKING_FILE, index=False)
    
    def _save_results(self):
        """Save graded results."""
        self.results.to_csv(RESULTS_FILE, index=False)
    
    def _api_request(self, endpoint: str, params: dict = None) -> dict:
        """Make API request."""
        time.sleep(1.1)
        try:
            resp = requests.get(
                f"{API_BASE_URL}/{endpoint}",
                headers=self.headers,
                params=params,
                timeout=30
            )
            if resp.status_code == 200:
                return resp.json()
        except:
            pass
        return {"data": []}
    
    def add_predictions(self, predictions_file: str):
        """Add predictions to tracking history."""
        print(f"Adding predictions from {predictions_file}...")
        
        new_preds = pd.read_csv(predictions_file)
        
        # Add tracking columns
        new_preds['added_at'] = datetime.now().isoformat()
        new_preds['graded'] = False
        new_preds['actual'] = None
        new_preds['result'] = None  # WIN, LOSS, PUSH
        
        # Append to history
        if len(self.history) > 0:
            self.history = pd.concat([self.history, new_preds], ignore_index=True)
        else:
            self.history = new_preds
        
        # Remove duplicates (same player, date, stat, line)
        self.history = self.history.drop_duplicates(
            subset=['date', 'player_id', 'stat', 'line'],
            keep='last'
        )
        
        self._save_history()
        print(f"  Added {len(new_preds)} predictions")
        print(f"  Total in history: {len(self.history)}")
    
    def get_actual_stats(self, game_date: str, player_id: int) -> dict:
        """Get actual stats for a player on a specific date."""
        current_year = datetime.now().year
        season = current_year if datetime.now().month >= 10 else current_year - 1
        
        stats = self._api_request("stats", {
            "seasons[]": season,
            "player_ids[]": player_id,
            "start_date": game_date,
            "end_date": game_date,
        }).get("data", [])
        
        if not stats:
            return None
        
        s = stats[0]
        return {
            'pts': s.get('pts') or 0,
            'reb': s.get('reb') or 0,
            'ast': s.get('ast') or 0,
            'stl': s.get('stl') or 0,
            'blk': s.get('blk') or 0,
        }
    
    def grade_predictions(self, game_date: str = None):
        """
        Grade predictions for a specific date.
        
        Result logic:
        - If we predicted OVER (prob > 0.5) and actual > line: WIN
        - If we predicted OVER and actual < line: LOSS
        - If actual == line: PUSH
        - Same logic for UNDER
        """
        if game_date is None:
            game_date = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        print(f"\nGrading predictions for {game_date}...")
        
        # Filter ungraded predictions for this date
        mask = (self.history['date'] == game_date) & (self.history['graded'] == False)
        to_grade = self.history[mask]
        
        if len(to_grade) == 0:
            print("  No predictions to grade for this date.")
            return
        
        print(f"  Found {len(to_grade)} predictions to grade")
        
        # Get unique players
        players = to_grade['player_id'].unique()
        
        # Fetch actuals
        actuals = {}
        for pid in players:
            print(f"  Fetching actuals for player {pid}...")
            actual = self.get_actual_stats(game_date, pid)
            if actual:
                actuals[pid] = actual
        
        print(f"  Got actuals for {len(actuals)} players")
        
        # Grade each prediction
        graded_count = 0
        wins = 0
        losses = 0
        pushes = 0
        
        for idx in to_grade.index:
            row = self.history.loc[idx]
            pid = row['player_id']
            stat = row['stat'].lower()
            line = row['line']
            over_prob = row['over_prob']
            
            if pid not in actuals:
                continue
            
            actual_value = actuals[pid].get(stat)
            if actual_value is None:
                continue
            
            # Record actual
            self.history.loc[idx, 'actual'] = actual_value
            self.history.loc[idx, 'graded'] = True
            
            # Determine our pick (OVER if over_prob > 0.5, else UNDER)
            our_pick = 'OVER' if over_prob > 0.5 else 'UNDER'
            
            # Grade
            if actual_value > line:
                # Actual went OVER
                if our_pick == 'OVER':
                    result = 'WIN'
                    wins += 1
                else:
                    result = 'LOSS'
                    losses += 1
            elif actual_value < line:
                # Actual went UNDER
                if our_pick == 'UNDER':
                    result = 'WIN'
                    wins += 1
                else:
                    result = 'LOSS'
                    losses += 1
            else:
                # Push
                result = 'PUSH'
                pushes += 1
            
            self.history.loc[idx, 'result'] = result
            graded_count += 1
        
        self._save_history()
        
        print(f"\n  Graded: {graded_count}")
        print(f"  Record: {wins}/{losses}/{pushes} (W/L/P)")
        
        if wins + losses > 0:
            win_pct = wins / (wins + losses) * 100
            print(f"  Win Rate: {win_pct:.1f}%")
    
    def generate_report(self) -> dict:
        """Generate comprehensive performance report."""
        graded = self.history[self.history['graded'] == True].copy()
        
        if len(graded) == 0:
            print("No graded predictions yet.")
            return {}
        
        report = {}
        
        # Overall record
        wins = (graded['result'] == 'WIN').sum()
        losses = (graded['result'] == 'LOSS').sum()
        pushes = (graded['result'] == 'PUSH').sum()
        total = wins + losses
        
        report['overall'] = {
            'record': f"{wins}/{losses}/{pushes}",
            'win_rate': round(wins / total * 100, 1) if total > 0 else 0,
            'total_graded': len(graded),
        }
        
        # By stat type
        report['by_stat'] = {}
        for stat in graded['stat'].unique():
            stat_df = graded[graded['stat'] == stat]
            w = (stat_df['result'] == 'WIN').sum()
            l = (stat_df['result'] == 'LOSS').sum()
            p = (stat_df['result'] == 'PUSH').sum()
            t = w + l
            
            report['by_stat'][stat] = {
                'record': f"{w}/{l}/{p}",
                'win_rate': round(w / t * 100, 1) if t > 0 else 0,
            }
        
        # By probability bucket (CALIBRATION - most important!)
        report['calibration'] = {}
        graded['prob_bucket'] = pd.cut(
            graded['over_prob'].clip(0.01, 0.99),
            bins=[0, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.0],
            labels=['<35%', '35-45%', '45-55%', '55-65%', '65-75%', '75-85%', '>85%']
        )
        
        for bucket in graded['prob_bucket'].unique():
            if pd.isna(bucket):
                continue
            bucket_df = graded[graded['prob_bucket'] == bucket]
            
            # For OVER predictions in this bucket, what % actually went over?
            over_preds = bucket_df[bucket_df['over_prob'] > 0.5]
            if len(over_preds) > 0:
                actual_over_rate = (over_preds['actual'] > over_preds['line']).mean()
            else:
                actual_over_rate = None
            
            # Win rate in this bucket
            w = (bucket_df['result'] == 'WIN').sum()
            l = (bucket_df['result'] == 'LOSS').sum()
            t = w + l
            
            report['calibration'][str(bucket)] = {
                'predictions': len(bucket_df),
                'win_rate': round(w / t * 100, 1) if t > 0 else 0,
                'expected_win_rate': self._expected_rate_for_bucket(str(bucket)),
            }
        
        # By edge bucket
        report['by_edge'] = {}
        graded['best_edge'] = graded[['edge_over', 'edge_under']].max(axis=1)
        graded['edge_bucket'] = pd.cut(
            graded['best_edge'],
            bins=[-1, 0, 0.03, 0.05, 0.08, 1],
            labels=['Negative', '0-3%', '3-5%', '5-8%', '>8%']
        )
        
        for bucket in ['0-3%', '3-5%', '5-8%', '>8%']:
            bucket_df = graded[graded['edge_bucket'] == bucket]
            if len(bucket_df) == 0:
                continue
            
            w = (bucket_df['result'] == 'WIN').sum()
            l = (bucket_df['result'] == 'LOSS').sum()
            t = w + l
            
            report['by_edge'][bucket] = {
                'predictions': len(bucket_df),
                'win_rate': round(w / t * 100, 1) if t > 0 else 0,
            }
        
        # ROI calculation (assuming -110 odds on all bets)
        # Win pays 0.909 units, Loss costs 1 unit
        report['roi'] = {
            'units_won': round(wins * 0.909, 2),
            'units_lost': round(losses * 1, 2),
            'net_units': round(wins * 0.909 - losses, 2),
            'roi_pct': round((wins * 0.909 - losses) / total * 100, 2) if total > 0 else 0,
        }
        
        return report
    
    def _expected_rate_for_bucket(self, bucket: str) -> float:
        """Expected win rate for a probability bucket."""
        expected = {
            '<35%': 65,  # UNDER picks should win 65%
            '35-45%': 55,
            '45-55%': 50,
            '55-65%': 60,
            '65-75%': 70,
            '75-85%': 80,
            '>85%': 85,
        }
        return expected.get(bucket, 50)
    
    def print_report(self):
        """Print formatted performance report."""
        report = self.generate_report()
        
        if not report:
            return
        
        print("\n" + "="*70)
        print("MODEL PERFORMANCE REPORT")
        print("="*70)
        
        # Overall
        print("\n📊 OVERALL RECORD")
        print("-"*40)
        overall = report['overall']
        print(f"  Record: {overall['record']} (W/L/P)")
        print(f"  Win Rate: {overall['win_rate']}%")
        print(f"  Total Graded: {overall['total_graded']}")
        
        # By Stat
        print("\n📈 BY STAT TYPE")
        print("-"*40)
        for stat, data in report['by_stat'].items():
            print(f"  {stat}: {data['record']} ({data['win_rate']}%)")
        
        # Calibration (MOST IMPORTANT)
        print("\n🎯 CALIBRATION (Probability Accuracy)")
        print("-"*40)
        print("  If model says 70%, it should win ~70% of the time")
        print()
        for bucket, data in report['calibration'].items():
            expected = data['expected_win_rate']
            actual = data['win_rate']
            diff = actual - expected
            status = "✓" if abs(diff) <= 5 else "⚠️" if abs(diff) <= 10 else "❌"
            print(f"  {bucket}: {actual}% actual vs {expected}% expected {status}")
            print(f"           ({data['predictions']} predictions)")
        
        # By Edge
        print("\n💰 BY EDGE (How confident picks perform)")
        print("-"*40)
        for bucket, data in report.get('by_edge', {}).items():
            print(f"  {bucket} edge: {data['win_rate']}% ({data['predictions']} picks)")
        
        # ROI
        print("\n💵 ROI (Assuming -110 odds)")
        print("-"*40)
        roi = report['roi']
        print(f"  Units Won: +{roi['units_won']}")
        print(f"  Units Lost: -{roi['units_lost']}")
        print(f"  Net: {roi['net_units']:+.2f} units")
        print(f"  ROI: {roi['roi_pct']:+.2f}%")
        
        # Required win rate to break even at -110
        print("\n  (Break-even at -110: 52.4%)")
        
        print("\n" + "="*70)
    
    def export_full_history(self, filename: str = None):
        """Export complete prediction history."""
        if filename is None:
            filename = f"prediction_history_export_{date.today()}.csv"
        
        self.history.to_csv(filename, index=False)
        print(f"Exported {len(self.history)} predictions to {filename}")


def main():
    tracker = PredictionTracker()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 track_results.py add <predictions_file.csv>")
        print("  python3 track_results.py grade [YYYY-MM-DD]")
        print("  python3 track_results.py report")
        print("  python3 track_results.py export")
        return
    
    command = sys.argv[1].lower()
    
    if command == "add":
        if len(sys.argv) < 3:
            print("Specify predictions file: python3 track_results.py add predictions.csv")
            return
        tracker.add_predictions(sys.argv[2])
    
    elif command == "grade":
        game_date = sys.argv[2] if len(sys.argv) > 2 else None
        tracker.grade_predictions(game_date)
    
    elif command == "report":
        tracker.print_report()
    
    elif command == "export":
        tracker.export_full_history()
    
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
