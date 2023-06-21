from src.main.python.games.golf.golf import Golf


def test_golf():
    iterations = 1000
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.99

    golf = Golf(iterations=iterations)

    # Resolución utilizando Montecarlo IE
    montecarlo_agent = golf.resolve_golf_by_montecarlo()
    print("Montecarlo IE:")
    golf.show_policy(montecarlo_agent)
    golf.print_stats()

    # Resolución utilizando Q-Learning
    q_learning_agent = golf.resolve_golf_by_q_learning(epsilon)
    print("Q-Learning:")
    golf.show_policy(q_learning_agent)
    golf.print_stats()

    # Resolución utilizando Sarsa
    sarsa_agent = golf.resolve_golf_by_sarsa(epsilon, alpha, gamma)
    print("Sarsa:")
    golf.show_policy(sarsa_agent)
    golf.print_stats()

    # Resolución utilizando Double Q-Learning
    double_q_learning_agent = golf.resolve_golf_by_double_q_learning(epsilon, alpha, gamma)
    print("Double Q-Learning:")
    golf.show_policy(double_q_learning_agent)
    golf.print_stats()


if __name__ == '__main__':
    test_golf()
