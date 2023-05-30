
# Reinforcement Learning Games with SARSA
This repository contains three simple games implemented using the SARSA (State-Action-Reward-State-Action) algorithm in the field of reinforcement learning. The SARSA algorithm is an on-policy temporal difference learning method that allows an agent to learn the optimal policy by iteratively updating its value function based on the observed rewards.

# Games Included
1. Frozen Lake

- **Objective**: Navigate a frozen lake from the starting point to the goal without falling into any holes.
- **State Representation**: Integer representing the current position on the grid.
- **Actions**: Move left, down, right, or up.
- **Rewards**: +1 for reaching the goal, 0 otherwise.
- **Environment**: 5x5 grid with a taxi, a passenger, and colored positions (red, green, yellow, blue) representing the passenger's position and the destination.

2. Taxi

- **Objective**: Move a taxi from its initial position to the passenger's location, pick up the passenger, and drop them off at the desired destination as quickly as possible.
- **State Representation**: Integer encoding the taxi's position, the passenger's position, and the destination.
- **Actions**: Move south, north, east, west, pick up the passenger, or drop off the passenger.
- **Rewards**: +20 for successfully dropping off the passenger, -10 for attempting to pick up or drop off the passenger in an incorrect location, -1 for any other action.
- **Environment**: 5x5 grid with a taxi, a passenger, and colored positions (red, green, yellow, blue) representing the passenger's position and the destination.

3. Golf

- **Objective**: Guide a golf ball from the starting position to the hole with the fewest possible strokes.
- **State Representation**: Grid coordinates of the ball's current position.
- **Actions**: Choose a club (two options) and specify the force and direction of the stroke.
- **Rewards**: Vary based on the success of the stroke.
- **Environment**: Customizable golf course with varying terrain and obstacles.
