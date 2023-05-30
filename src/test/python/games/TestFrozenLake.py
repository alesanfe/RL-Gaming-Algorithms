from src.main.python.games.FrozenLake import FrozenLake

if __name__ == "__main__":
    frozen_lake = FrozenLake()
    agent = frozen_lake.resolve_frozen_lake_by_montecarlo()

    agent.entorno.render()