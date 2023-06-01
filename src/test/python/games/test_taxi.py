from src.main.python.games.Taxi import Taxi

def test_taxi():
    iterations = 1000
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.99

    taxi = Taxi(iterations=iterations)

    # Resolución utilizando Montecarlo IE
    montecarlo_agent = taxi.resolve_taxi_by_montecarlo()
    print("Montecarlo IE:")
    taxi.print_stats()

    # Resolución utilizando Q-Learning
    q_learning_agent = taxi.resolve_taxi_by_q_learning(epsilon)
    print("Q-Learning:")
    taxi.print_stats()

    # Resolución utilizando Sarsa
    sarsa_agent = taxi.resolve_taxi_by_sarsa(epsilon, alpha, gamma)
    print("Sarsa:")
    taxi.print_stats()

    # Resolución utilizando Double Q-Learning
    double_q_learning_agent = taxi.resolve_taxi_by_double_q_learning(epsilon, alpha, gamma)
    print("Double Q-Learning:")
    taxi.print_stats()

if __name__ == "__main__":
    test_taxi()