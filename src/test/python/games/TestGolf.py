from src.main.python.games.Golf import Golf

if __name__ == "__main__":
    golf = Golf()
    agent = golf.resolve_golf_by_montecarlo()

    agent.entorno.render()