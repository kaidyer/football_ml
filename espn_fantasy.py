# Football API
from espn_api.football import League

league = League(league_id=607716459, year=2025)

def getFA(nameStrings=False):
    listFA = league.free_agents(size=1000)
    if nameStrings:
        return [player.name for player in listFA]
    else:
        return listFA