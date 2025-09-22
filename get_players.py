from sleeper.api import get_all_players
import pickle
import pandas as pd



def pickle_players():
    """
    Get all players
    Don't call this if you already have your pickle. Sleeper doesn't like getting a ton of requests.
    """
    # get all players in a sport
    nfl_players = get_all_players(sport="nfl")

    with open("all_players.pkl", "wb") as f:
        pickle.dump(nfl_players, f)


def clean_data(nfl_players):
    """
    Put data into a nice pandas dataframe.

    :param nfl_players:
        takes in the dictionary of dictionaries that the sleeper API returns
    :return:
        Nothing
    """
    # Convert dictionary-of-dictionaries to DataFrame
    nfl_players = pd.DataFrame.from_dict(nfl_players, orient='index')

    # Remove free agents (players with no team assigned)
    nfl_players = nfl_players[nfl_players['team'].notna() & (nfl_players['team'] != '')]

    # Optional: keep only relevant columns
    columns_to_keep = ['player_id', 'full_name', 'position', 'team', 'team_abbr', 'years_exp', 'fantasy_positions',
                       'status', 'age', 'college']
    nfl_players = nfl_players[columns_to_keep]

    # Reset index
    nfl_players.reset_index(drop=True, inplace=True)

    with open("all_players.pkl", "wb") as f:
        pickle.dump(nfl_players, f)


def get_pickled_players():
    """
    Once you have your pickle, you can get the dataframe with this.
    :return:
        A dataframe of all players that are not a free agent.
    """
    with open("all_players.pkl", "rb") as f:
        return pickle.load(f)

