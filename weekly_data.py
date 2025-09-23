import requests, time, pickle
import pandas as pd
from bs4 import BeautifulSoup

def build_fantasy_data():
    url = f"https://www.pro-football-reference.com/years/2025/fantasy.htm"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve data for overall fantasy stats")

    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', id='fantasy')
    if table is None:
        print(f"No fantasy table found for overall stats")

    df = pd.read_html(str(table))[0]

    # Remove multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(1)

    fantasy_data = df

    time.sleep(3)  # be polite and avoid hammering the server

    # Clean data: drop rows where Player is 'Player' (header rows repeated), and drop rows with all NaNs
    fantasy_data = fantasy_data[fantasy_data['Player'] != 'Player']
    fantasy_data = fantasy_data.dropna(how='all')

    # Reset index
    fantasy_data = fantasy_data.reset_index(drop=True)

    # Save to pickle and CSV
    fantasy_data.to_pickle('fantasy_overall_2025.pkl')


def get_fantasy_data():
    with open("fantasy_overall_2025.pkl", "rb") as f:
        return pickle.load(f)
