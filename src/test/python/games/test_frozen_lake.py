from src.main.python.games.frozen_lake import FrozenLake


def test_frozen_lake():
    iterations = 1000000
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.99

    frozen_lake = FrozenLake(iterations=iterations)

    # Resolución utilizando Montecarlo con inicios exploratorios
    montecarlo_agent = frozen_lake.resolve_frozen_lake_by_montecarlo()
    print("Montecarlo:")
    frozen_lake.show_policy(montecarlo_agent)
    frozen_lake.print_stats()

    # Resolución utilizando Q-Learning
    q_learning_agent = frozen_lake.resolve_frozen_lake_by_q_learning(epsilon)
    print("Q-Learning:")
    frozen_lake.show_policy(q_learning_agent)
    frozen_lake.print_stats()

    # Resolución utilizando Sarsa
    sarsa_agent = frozen_lake.resolve_frozen_lake_by_sarsa(epsilon, alpha, gamma)
    print("Sarsa:")
    frozen_lake.show_policy(sarsa_agent)
    frozen_lake.print_stats()

    # Resolución utilizando Double Q-Learning
    double_q_learning_agent = frozen_lake.resolve_frozen_lake_by_double_q_learning(epsilon, alpha, gamma)
    print("Double Q-Learning:")
    frozen_lake.show_policy(double_q_learning_agent)
    frozen_lake.print_stats()


if __name__ == "__main__":
    test_frozen_lake()
