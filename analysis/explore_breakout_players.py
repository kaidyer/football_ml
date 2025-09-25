import pickle, os
from retrieve.espn_fantasy import getFA
from pathlib import Path
from retrieve.get_players import get_full_data


def crossRefFA():
    full_data = get_full_data()
    listFA = getFA(True)
    bestFA = {}
    for player in full_data["full_name"]:
        if player in listFA:
            bestFA[player] = full_data[full_data["full_name"] == player]["Breakout_Prob"].iloc[0]
    sorted_dict_by_value = dict(sorted(bestFA.items(), key=lambda item: item[1]))
    return sorted_dict_by_value


for player, value in crossRefFA().items():
    print(player," ", value)