import pickle
from espn_fantasy import getFA

def get_full_data():
    with open("full_data.pkl", "rb") as f:
        return pickle.load(f)


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