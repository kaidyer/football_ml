from get_players import get_pickled_players
from weekly_data import get_fantasy_data
import pandas as pd


nfl_players = get_pickled_players()
fantasy_data = get_fantasy_data()
full_data = pd.merge(nfl_players, fantasy_data, left_on='full_name', right_on='Player', how='inner')
full_data.columns = ['player_id', 'full_name', 'position', 'team', 'team_abbr', 'years_exp', 'fantasy_positions', 'status', 'age', 'college', 'Rk', 'Player', 'Tm', 'FantPos', 'Age', 'G', 'GS', 'Cmp', 'PassAtt', 'PassYds', 'PassTD', 'Int', 'RushAtt', 'RushYds', 'RushY/A', 'RushTD', 'RecTgt', 'Rec', 'RecYds', 'RecY/R', 'RecTD', 'Fmb', 'FL', 'TD', '2PM', '2PP', 'FantPt', 'PPR', 'DKPt', 'FDPt', 'VBD', 'PosRank', 'OvRank']

cols_to_numeric = ['Age', 'G', 'GS', 'Cmp', 'PassAtt', 'PassYds', 'PassTD', 'Int', 'RushAtt', 'RushYds', 'RushY/A', 'RushTD', 'RecTgt', 'Rec', 'RecYds', 'RecY/R', 'RecTD', 'Fmb', 'FL', 'TD', '2PM', '2PP', 'FantPt', 'PPR', 'DKPt', 'FDPt', 'VBD', 'PosRank', 'OvRank']
for col in cols_to_numeric:
    full_data[col] = pd.to_numeric(full_data[col], errors='coerce')
# Feature engineering

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

# One-hot encoding for position
position_dummies = pd.get_dummies(full_data['position'], prefix='pos')

# Combine all features into feature matrix X
feature_cols = [
    'pass_yds_per_game', 'rush_yds_per_game', 'rec_yds_per_game', 'rec_per_game',
    'rush_att_per_game', 'pass_att_per_game', 'targets_per_game', 'catch_rate',
    'rush_td_rate', 'rec_td_rate', 'pass_td_rate', 'int_rate', 'total_touchdowns',
    'total_yards', 'pos_rank_norm', 'ov_rank_norm'
]

X = pd.concat([full_data[feature_cols], position_dummies], axis=1)


print("here")
