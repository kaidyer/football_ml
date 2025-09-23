from get_players import get_pickled_players
from weekly_data import get_fantasy_data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import numpy as np

# --- Load Data ---
nfl_players = get_pickled_players()
fantasy_data = get_fantasy_data()
full_data = pd.merge(nfl_players, fantasy_data, left_on='full_name', right_on='Player', how='inner')
full_data.columns = ['player_id', 'full_name', 'position', 'team', 'team_abbr', 'years_exp', 'fantasy_positions', 'status', 'age', 'college', 'Rk', 'Player', 'Tm', 'FantPos', 'Age', 'G', 'GS', 'Cmp', 'PassAtt', 'PassYds', 'PassTD', 'Int', 'RushAtt', 'RushYds', 'RushY/A', 'RushTD', 'RecTgt', 'Rec', 'RecYds', 'RecY/R', 'RecTD', 'Fmb', 'FL', 'TD', '2PM', '2PP', 'FantPt', 'PPR', 'DKPt', 'FDPt', 'VBD', 'PosRank', 'OvRank']

# --- Data Cleaning / Type Conversion ---
cols_to_numeric = ['Age', 'G', 'GS', 'Cmp', 'PassAtt', 'PassYds', 'PassTD', 'Int', 'RushAtt', 'RushYds', 'RushY/A', 'RushTD', 'RecTgt', 'Rec', 'RecYds', 'RecY/R', 'RecTD', 'Fmb', 'FL', 'TD', '2PM', '2PP', 'FantPt', 'PPR', 'DKPt', 'FDPt', 'VBD', 'PosRank', 'OvRank']
for col in cols_to_numeric:
    full_data[col] = pd.to_numeric(full_data[col], errors='coerce')

# --- Feature Engineering ---

# Per game metrics
full_data['pass_yds_per_game'] = full_data['PassYds'] / full_data['G'].replace(0, pd.NA)
full_data['rush_yds_per_game'] = full_data['RushYds'] / full_data['G'].replace(0, pd.NA)
full_data['rec_yds_per_game'] = full_data['RecYds'] / full_data['G'].replace(0, pd.NA)
full_data['rec_per_game'] = full_data['Rec'] / full_data['G'].replace(0, pd.NA)
full_data['rush_att_per_game'] = full_data['RushAtt'] / full_data['G'].replace(0, pd.NA)
full_data['pass_att_per_game'] = full_data['PassAtt'] / full_data['G'].replace(0, pd.NA)
full_data['targets_per_game'] = full_data['RecTgt'] / full_data['G'].replace(0, pd.NA)

# Usage and efficiency metrics
full_data['catch_rate'] = full_data['Rec'] / full_data['RecTgt'].replace(0, pd.NA)
full_data['rush_td_rate'] = full_data['RushTD'] / full_data['RushAtt'].replace(0, pd.NA)
full_data['rec_td_rate'] = full_data['RecTD'] / full_data['Rec'].replace(0, pd.NA)
full_data['pass_td_rate'] = full_data['PassTD'] / full_data['PassAtt'].replace(0, pd.NA)
full_data['int_rate'] = full_data['Int'] / full_data['PassAtt'].replace(0, pd.NA)

# Contextual features
full_data['total_touchdowns'] = full_data['RushTD'] + full_data['RecTD'] + full_data['PassTD']
full_data['total_yards'] = full_data['PassYds'] + full_data['RushYds'] + full_data['RecYds']

# Ranking normalization
full_data['pos_rank_norm'] = full_data['PosRank'] / full_data['PosRank'].max()
full_data['ov_rank_norm'] = full_data['OvRank'] / full_data['OvRank'].max()

# --- Target Creation ---
# Define breakout as top 20% of players in PPR per game per position
full_data['PPR_per_game'] = full_data['PPR'] / full_data['G'].replace(0, pd.NA)

# Compute threshold per position
thresholds = full_data.groupby('position')['PPR_per_game'].transform(lambda x: x.quantile(0.8))

# Assign breakout = 1 if player's PPR_per_game is above their position's threshold
full_data['Breakout'] = (full_data['PPR_per_game'] >= thresholds).astype(int)

y = full_data['Breakout']

print(y.value_counts())

# --- Prepare Feature Matrix ---
# One-hot encoding for position
position_dummies = pd.get_dummies(full_data['position'], prefix='pos')

feature_cols = [
    'pass_yds_per_game', 'rush_yds_per_game', 'rec_yds_per_game', 'rec_per_game',
    'rush_att_per_game', 'pass_att_per_game', 'targets_per_game', 'catch_rate',
    'rush_td_rate', 'rec_td_rate', 'pass_td_rate', 'int_rate', 'total_touchdowns',
    'total_yards', 'pos_rank_norm', 'ov_rank_norm'
]

X = pd.concat([full_data[feature_cols], position_dummies], axis=1)

# --- Handle Missing Values ---
X = X.where(pd.notna(X), np.nan)
imputer = SimpleImputer(strategy="constant", fill_value=0)
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Model Training & Evaluation ---
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- Compute predicted breakout probabilities for all players ---
breakout_probs = clf.predict_proba(X)[:, 1]
full_data['Breakout_Prob'] = breakout_probs

# Sort full_data by Breakout_Prob in descending order
full_data_sorted = full_data.sort_values(by='Breakout_Prob', ascending=False)

# Print top 10 players with full_name, position, and Breakout_Prob
print("\nTop 10 Players by Predicted Breakout Probability:")
print(full_data_sorted[['full_name', 'position', 'Breakout_Prob']].head(10))

full_data_sorted.to_pickle("full_data.pkl")
