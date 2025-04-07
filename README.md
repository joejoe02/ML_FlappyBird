# Flappy Bird Reinforcement Learning Agent

This project was part of a machine learning course where I explored reinforcement learning by training an agent to play Flappy Bird. I started with a basic Q-learning approach and then moved on to Deep Q-learning to handle more complex gameplay situations.

## Project Goals

- Implement a Q-learning agent for Flappy Bird
- Extend the agent with Deep Q-learning using neural networks
- Experiment with different hyperparameters
- Compare learning stability and performance

## The Environment

The environment is a discrete, stochastic system. The agent chooses between two actions: flap or do nothing. The state includes:

- Bird's vertical position
- Distance to the next pipe
- Height of the next pipe
- Bird’s vertical velocity

These inputs are enough to represent the state while following the Markov property.

## Q-learning Implementation

I started by breaking continuous state values into 15 intervals and used a table to store Q-values for different actions. The reward system was as follows:

- +1 for passing a pipe
- 0 for flapping
- -5 for crashing

I used epsilon-greedy exploration (ε = 0.1) and a learning rate of 0.1. Over 500 episodes, the agent gradually improved, though performance fluctuated heavily.

## Deep Q-learning

I then implemented Deep Q-learning with two neural networks (online and target). The agent used experience replay and an epsilon-greedy strategy with decaying epsilon to balance exploration and exploitation. This version handled the state space much better and allowed the agent to learn more general behaviors.

### Key Features:
- Experience replay
- Target network updates
- Two hidden layers in the neural network
- Epsilon decay over time

## Hyperparameter Experiments

I experimented with different parameters to see their impact on learning stability:

- **Epsilon Decay**: 0.999 gave more stable learning than faster decay.
- **Batch Size**: 50 worked best by smoothing out randomness.
- **Target Update Interval**: Updating every 70 steps provided more consistent Q-value targets.

## Final Best Settings

| Parameter              | Value     |
|------------------------|-----------|
| Epsilon Decay          | 0.999     |
| Batch Size             | 50        |
| Target Update Interval | 70        |

These settings helped reduce score spikes and gave more reliable learning performance.

## Conclusion

This project gave me hands-on experience with reinforcement learning and showed how small tweaks in hyperparameters can affect an agent’s ability to learn. Deep Q-learning proved to be far more effective than basic Q-learning for this type of environment.
