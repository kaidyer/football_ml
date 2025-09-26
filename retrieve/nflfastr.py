import nfl_data_py as nfl
import pandas as pd
from pathlib import Path
import os, pickle

BASE_DIR = Path(__file__).parent      # folder2/
PKL_DIR = BASE_DIR.parent / "pickles"   # folder/data

def save_weekly_2024():
    # Pull 2024 weekly player stats
    stats_2024 = nfl.import_weekly_data([2024])
    stats_2024.to_pickle(PKL_DIR / "2024_weekly.pkl")


def get_weekly_data(year = 2024):
    return pd.read_pickle(PKL_DIR / (str(year) + "_weekly.pkl"))


def save_sliding():
    sliding_df = make_sliding_window_dataset()
    sliding_df.to_pickle(PKL_DIR / "2024_sliding.pkl")


def get_sliding(year = 2024):
    return pd.read_pickle(PKL_DIR / (str(year) + "_sliding.pkl"))


def make_sliding_window_dataset(window=3):
    """
    Converts a weekly player stats pickle into a sliding-window dataset.
    Features: last `window` weeks of fantasy points.
    Target: next week's fantasy points.
    """
    # Load the pickle
    df = get_weekly_data()

    # Keep only columns we care about
    df = df[['player_id', 'player_display_name', 'week', 'fantasy_points_ppr']]

    # Sort by player and week
    df = df.sort_values(['player_id', 'week']).reset_index(drop=True)

    # Prepare lists to collect the data
    X_all, y_all, player_all, player_name_all, week_all = [], [], [], [], []

    # Function to create sliding windows for a single player
    def make_windows(player_df):
        fp = player_df['fantasy_points_ppr'].values
        X, y = [], []
        for i in range(window, len(fp)):
            X.append(fp[i-window:i])
            y.append(fp[i])
        return X, y

    # Loop over players
    for pid, group in df.groupby('player_id'):
        X, y = make_windows(group)
        X_all.extend(X)
        y_all.extend(y)
        player_all.extend([pid]*len(y))
        player_name_all.extend([group['player_display_name'].iloc[0]]*len(y))
        week_all.extend(group['week'].values[window:])

    # Create a DataFrame
    X_df = pd.DataFrame(X_all, columns=[f'week_t-{i}' for i in range(window,0,-1)])
    X_df['y'] = y_all
    X_df['player_id'] = player_all
    X_df['player_name'] = player_name_all
    X_df['week'] = week_all

    return X_df

