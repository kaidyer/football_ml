import pickle

def get_full_data():
    with open("full_data.pkl", "rb") as f:
        return pickle.load(f)

full_data = get_full_data()
my_roster = [
    "Drake Maye",
    "Chase Brown",
    "James Connor",
    "Tee Higgins",
    "Justin Jefferson",
    "Travis Kelce",
    "D'Andre Swift",
    "Parker Romo"
]
backups = [
    "Kenneth Gainwell",
    "Kareem Hunt",
    "Kyle Pitts Sr",
    "Jayden Reed",
    "Rashid Shaheed",
    "C.J. Stroud",
    "Darren Waller"
]

for player in my_roster:
    print(full_data[full_data["full_name"] == player])

for player in backups:
    print(full_data[full_data["full_name"] == player])