from src.main.python.games.Taxi import Taxi

if __name__ == "__main__":
    taxi = Taxi()
    agent = taxi.resolve_taxi_by_montecarlo()

    agent.entorno.render()