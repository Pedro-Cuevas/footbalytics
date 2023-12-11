import requests
import pandas as pd

class ApiCaller():
    # comenzamos con algunos datos básicos
    def __init__(self,year):
        self.year = year
        self.base_url = "https://v3.football.api-sports.io/"
        self.api_key = "dfdb906f7d7e282404f86ed0e3145a20"
    
    # función para hacer llamadas a la API
    def api_call(self,endpoint):
        url = self.base_url + endpoint
        headers = {
            'x-rapidapi-host': "v3.football.api-sports.io",
            'x-rapidapi-key': self.api_key
            }
        response = requests.request("GET", url, headers=headers)
        return response.json()
    
    # función para obtener los equipos de una liga
    def get_leagues(self):
        response = self.api_call("leagues?season=" + str(self.year))
        return pd.json_normalize(response['response'], sep='_').drop("seasons",axis=1)
        #return pd.read_csv('leagues.csv')
    
    # función para obtener los equipos de una liga a partir del id de liga
    def get_teams_from_league(self,league_id):
        response = self.api_call("teams?league=" + str(league_id) + "&season=" + str(self.year))
        return pd.json_normalize(response['response'], sep='_')
        #return pd.read_csv('la_liga.csv')
    
    # función para obtener las estadísticas de un equipo a partir de team ID y league ID
    def get_team_stats(self,team_id,league_id):
        response = self.api_call("/teams/statistics?" + "season=" + str(self.year) + "&team="  + str(team_id) + "&league=" + str(league_id))
        return pd.json_normalize(response['response'], sep='_')
        #return pd.read_csv('real_madrid.csv')