import gym
from gym.wrappers import RecordEpisodeStatistics

from src.main.python.games.frozen_lake import FrozenLake
from src.main.python.games.game import Game
from src.main.python.games.game_comparator import GameComparator
from src.main.python.games.golf.golf import Golf
from src.main.python.games.golf.golf_env import GolfEnv
from src.main.python.games.taxi import Taxi


# ---------------------- #
# Test para Frozen Lake. #
# ---------------------- #

def test_compare_different_algorithms_frozen_lake():
    game = Game(environment=RecordEpisodeStatistics(gym.make('FrozenLake-v1')),
                discount_factor=0.9,
                learning_factor=0.1,
                iterations=500)

    frozen_lake = FrozenLake()

    test_game = GameComparator(game)

    agent = test_game.compare_different_algorithms()
    test_game.get_graphs("Frozen Lake")

    frozen_lake.show_policy(agent)


def test_compare_different_cases_frozen_lake_montecarlo():
    game = Game(environment=RecordEpisodeStatistics(gym.make('FrozenLake-v1')),
                discount_factor=0.9,
                learning_factor=0.1,
                iterations=500)

    frozen_lake = FrozenLake()

    test_game = GameComparator(game)

    agent = test_game.compare_different_cases(algorithm='Montecarlo', epsilon=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], alpha=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], gamma=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    frozen_lake.show_policy(agent)

def test_compare_different_cases_frozen_lake_q_learning():
    game = Game(environment=RecordEpisodeStatistics(gym.make('FrozenLake-v1')),
                discount_factor=0.9,
                learning_factor=0.1,
                iterations=500)

    frozen_lake = FrozenLake()

    test_game = GameComparator(game)

    agent = test_game.compare_different_cases(algorithm='Q-Learning', epsilon=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], alpha=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], gamma=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    frozen_lake.show_policy(agent)

def test_compare_different_cases_frozen_lake_sarsa():
    game = Game(environment=RecordEpisodeStatistics(gym.make('FrozenLake-v1')),
                discount_factor=0.9,
                learning_factor=0.1,
                iterations=500)

    frozen_lake = FrozenLake()

    test_game = GameComparator(game)

    agent = test_game.compare_different_cases(algorithm='Sarsa', epsilon=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], alpha=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], gamma=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    frozen_lake.show_policy(agent)

def test_compare_different_cases_frozen_lake_double_q_learning():
    game = Game(environment=RecordEpisodeStatistics(gym.make('FrozenLake-v1')),
                discount_factor=0.9,
                learning_factor=0.1,
                iterations=500)

    frozen_lake = FrozenLake()

    test_game = GameComparator(game)

    agent = test_game.compare_different_cases(algorithm='Double Q-Learning', epsilon=[0.1, 0.2, 0.3, 0.4, 0.5], alpha=[0.1, 0.2, 0.3, 0.4, 0.5], gamma=[0.1, 0.2, 0.3, 0.4, 0.5])

    frozen_lake.show_policy(agent)

# ---------------------- #
# Test para Taxi.        #
# ---------------------- #
def test_compare_different_algorithms_taxi():
    game = Game(environment=RecordEpisodeStatistics(gym.make('Taxi-v3')),
                discount_factor=0.9,
                learning_factor=0.1,
                iterations=500)

    taxi = Taxi()

    test_game = GameComparator(game)

    agent = test_game.compare_different_algorithms()
    test_game.get_graphs("Taxi")

    taxi.show_policy(agent)

def test_compare_different_cases_taxi_montecarlo():
    game = Game(environment=RecordEpisodeStatistics(gym.make('Taxi-v3')),
                discount_factor=0.9,
                learning_factor=0.1,
                iterations=500)

    taxi = Taxi()

    test_game = GameComparator(game)

    agent = test_game.compare_different_cases(algorithm='Montecarlo', epsilon=[0.25, 0.5, 0.75], alpha=[0.25, 0.5, 0.75], gamma=[0.25, 0.5, 0.75])

    taxi.show_policy(agent)

def test_compare_different_cases_taxi_q_learning():
    game = Game(environment=RecordEpisodeStatistics(gym.make('Taxi-v3')),
                discount_factor=0.9,
                learning_factor=0.1,
                iterations=500)

    taxi = Taxi()

    test_game = GameComparator(game)

    agent = test_game.compare_different_cases(algorithm='Q-Learning', epsilon=[0.25, 0.5, 0.75], alpha=[0.25, 0.5, 0.75], gamma=[0.25, 0.5, 0.75])

    taxi.show_policy(agent)

def test_compare_different_cases_taxi_sarsa():
    game = Game(environment=RecordEpisodeStatistics(gym.make('Taxi-v3')),
                discount_factor=0.9,
                learning_factor=0.1,
                iterations=500)

    taxi = Taxi()

    test_game = GameComparator(game)

    agent = test_game.compare_different_cases(algorithm='Sarsa', epsilon=[0.25, 0.5, 0.75], alpha=[0.25, 0.5, 0.75], gamma=[0.25, 0.5, 0.75])

    taxi.show_policy(agent)

def test_compare_different_cases_taxi_double_q_learning():
    game = Game(environment=RecordEpisodeStatistics(gym.make('Taxi-v3')),
                discount_factor=0.9,
                learning_factor=0.1,
                iterations=500)

    taxi = Taxi()

    test_game = GameComparator(game)

    agent = test_game.compare_different_cases(algorithm='Double Q-Learning', epsilon=[0.25, 0.5, 0.75], alpha=[0.25, 0.5, 0.75], gamma=[0.25, 0.5, 0.75])

    taxi.show_policy(agent)

# ---------------------- #
# Test para Golf.        #
# ---------------------- #

def test_compare_different_algorithms_golf():
    game = Game(environment=RecordEpisodeStatistics(GolfEnv()),
                discount_factor=0.9,
                learning_factor=0.1,
                iterations=500)

    golf = Golf()

    test_game = GameComparator(game)

    agent = test_game.compare_different_algorithms()
    test_game.get_graphs("Golf")

    golf.show_policy(agent)


def test_compare_different_cases_golf_montecarlo():
    game = Game(environment=RecordEpisodeStatistics(GolfEnv()),
                discount_factor=0.9,
                learning_factor=0.1,
                iterations=500)

    golf = Golf()

    test_game = GameComparator(game)

    agent = test_game.compare_different_cases(algorithm='Montecarlo', epsilon=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], alpha=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], gamma=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    golf.show_policy(agent)

def test_compare_different_cases_golf_q_learning():
    game = Game(environment=RecordEpisodeStatistics(GolfEnv()),
                discount_factor=0.9,
                learning_factor=0.1,
                iterations=500)

    golf = Golf()

    test_game = GameComparator(game)

    agent = test_game.compare_different_cases(algorithm='Q-Learning', epsilon=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], alpha=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], gamma=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    golf.show_policy(agent)

def test_compare_different_cases_golf_sarsa():
    game = Game(environment=RecordEpisodeStatistics(GolfEnv()),
                discount_factor=0.9,
                learning_factor=0.1,
                iterations=500)

    golf = Golf()

    test_game = GameComparator(game)

    agent = test_game.compare_different_cases(algorithm='Sarsa', epsilon=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], alpha=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], gamma=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    golf.show_policy(agent)

def test_compare_different_cases_golf_double_q_learning():
    game = Game(environment=RecordEpisodeStatistics(GolfEnv()),
                discount_factor=0.9,
                learning_factor=0.1,
                iterations=500)

    golf = Golf()

    test_game = GameComparator(game)

    agent = test_game.compare_different_cases(algorithm='Double Q-Learning', epsilon=[0.25, 0.5, 0.75], alpha=[0.25, 0.5, 0.75], gamma=[0.25, 0.5, 0.75])

    golf.show_policy(agent)

# -------------------------- #
# Test para probar entornos. #
# -------------------------- #

def test_compare_different_environments_montercarlo():
    game = Game(environment=None,
                discount_factor=0.9,
                learning_factor=0.1,
                iterations=500)

    test_game = GameComparator(game)

    agent = test_game.compare_different_environments(algorithm='Montecarlo')

    print(agent.tabla_q)

def test_compare_different_environments_q_learning():
    game = Game(environment=None,
                discount_factor=0.9,
                learning_factor=0.1,
                iterations=500)

    test_game = GameComparator(game)

    agent = test_game.compare_different_environments(algorithm='Q-Learning')
    print(agent.tabla_q)

def test_compare_different_environments_sarsa():
    game = Game(environment=None,
                discount_factor=0.9,
                learning_factor=0.1,
                iterations=500)

    test_game = GameComparator(game)

    agent = test_game.compare_different_environments(algorithm='Sarsa')
    print(agent.tabla_q)

def test_compare_different_environments_double_q_learning():
    game = Game(environment=None,
                discount_factor=0.9,
                learning_factor=0.1,
                iterations=500)

    test_game = GameComparator(game)

    agent = test_game.compare_different_environments(algorithm='Double Q-Learning')
    print(agent.tabla_q)

if __name__ == '__main__':
    print("\nTest para Frozen Lake.")
    test_compare_different_algorithms_frozen_lake()
    test_compare_different_cases_frozen_lake_montecarlo()
    test_compare_different_cases_frozen_lake_q_learning()
    test_compare_different_cases_frozen_lake_sarsa()
    test_compare_different_cases_frozen_lake_double_q_learning()

    print("\nTest para Taxi.")
    test_compare_different_algorithms_taxi()
    test_compare_different_cases_taxi_montecarlo()
    test_compare_different_cases_taxi_q_learning()
    test_compare_different_cases_taxi_sarsa()
    test_compare_different_cases_taxi_double_q_learning()

    print("\nTest para Golf.")
    test_compare_different_algorithms_golf()
    test_compare_different_cases_golf_montecarlo()
    test_compare_different_cases_golf_q_learning()
    test_compare_different_cases_golf_sarsa()
    test_compare_different_cases_golf_double_q_learning()

    print("\nTest para entornos.")
    test_compare_different_environments_montercarlo()
    test_compare_different_environments_q_learning()
    test_compare_different_environments_sarsa()
    test_compare_different_environments_double_q_learning()



