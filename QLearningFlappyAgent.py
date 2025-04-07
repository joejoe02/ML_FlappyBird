from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import random

class FlappyAgent:
    def __init__(self):
        # Discretize state space into intervals for y-position, pipe distance, pipe gap, and velocity
        self.intervals = {
            "player_y": 3,
            "next_pipe_top_y": 5,
            "next_pipe_dist_to_player": 5,
            "player_vel": 1
        }
        self.actions = [0, 1]  # 0 = flap, 1 = do nothing
        self.q_table = {}  # Initialize Q-table as a dictionary
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.1  # Exploration rate

    def discretize_state(self, state):
        """Converts continuous game state into discrete intervals for Q-table."""
        player_y = state['player_y'] // self.intervals['player_y']
        pipe_y = state['next_pipe_top_y'] // self.intervals['next_pipe_top_y']
        dist_to_pipe = state['next_pipe_dist_to_player'] // self.intervals['next_pipe_dist_to_player']
        velocity = state['player_vel'] // self.intervals['player_vel']
        return (player_y, pipe_y, dist_to_pipe, velocity)

    
    def reward_values(self):
        """ returns the reward values used for training
        
            Note: These are only the rewards used for training.
            The rewards used for evaluating the agent will always be
            1 for passing through each pipe and 0 for all other state
            transitions.
        """
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}
    
    def observe(self, s1, a, r, s2, end):
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.
            
            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """
        """ Q-learning update based on observed state transition. """
        state1 = self.discretize_state(s1)
        state2 = self.discretize_state(s2)
        
        # Initialize Q-values if state-action pairs are not yet in table
        if state1 not in self.q_table:
            self.q_table[state1] = [0.0 for _ in self.actions]  # Initialize to encourage exploration
        if state2 not in self.q_table:
            self.q_table[state2] = [0.0 for _ in self.actions]
        
        # Q-learning update rule
        best_next_action = np.argmax(self.q_table[state2])
        self.q_table[state1][a] += self.alpha * (r + self.gamma * self.q_table[state2][best_next_action] * (1 - end) - self.q_table[state1][a])
        

    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """

        # Discretize the current state
        discrete_state = self.discretize_state(state)

        # Initialize Q-values if state is not in Q-table
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = [0.0 for _ in self.actions]

        # Conservative flapping based on positional logic
        if state['player_y'] > state['next_pipe_bottom_y'] - 53 :
            # Only flap if significantly below the bottom of the pipe opening
            return 0  # Flap
        elif state['player_y'] < state['next_pipe_top_y'] + 48:
            # Do nothing if at or above the top of the pipe opening
            return 1  # Do nothing

        # Apply epsilon-greedy if not in these positional conditions
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)  # Explore
        else:
            return np.argmax(self.q_table[discrete_state])

    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """
        print("state: %s" % state)
        """ Greedy policy after training """
        discrete_state = self.discretize_state(state)
        if discrete_state not in self.q_table:
            return 0
        return np.argmax(self.q_table[discrete_state])

def run_game(nb_episodes, agent, training=True):
    """Runs episodes of the game with the agent making moves and tracks the number of flaps per episode."""
    reward_values = agent.reward_values() if training else {"positive": 1.0, "tick": -0.01, "loss": -5.0}
    
    env = PLE(FlappyBird(), fps=30, display_screen=True, force_fps=True, reward_values=reward_values)
    env.init()

    while nb_episodes > 0:
        score = 0
        flap_count = 0  # Initialize flap count for each episode
        env.reset_game()
        
        while not env.game_over():
            current_state = env.game.getGameState()
            action = agent.training_policy(current_state) if training else agent.policy(current_state)
            
            # Increment flap count if action is 0 (flap)
            if action == 0:
                flap_count += 1
            
            reward = env.act(env.getActionSet()[action])
            score += reward
            
            if training:
                next_state = env.game.getGameState()
                agent.observe(current_state, action, reward, next_state, env.game_over())
        
        # Print only the summary at the end of each episode
        print(f"Episode Summary - Score: {score}, Flaps: {flap_count}")
        nb_episodes -= 1
        agent.epsilon = max(0.01, agent.epsilon * 0.999)  # decay epsilon




def train(nb_episodes, agent):
    reward_values = agent.reward_values()
    
    env = PLE(FlappyBird(), fps=30, display_screen=True, force_fps=False, rng=None, reward_values=reward_values)
    env.init()

    while nb_episodes > 0:
        score = 0
        flap_count = 0  # Initialize flap count for each episode
        env.reset_game()
        
        while not env.game_over():
            state = env.game.getGameState()
            action = agent.training_policy(state)
            
            # Increment flap count if action is 0 (flap)
            if action == 0:
                flap_count += 1
            
            reward = env.act(env.getActionSet()[action])
            score += reward

            # Let the agent observe the current state transition
            new_state = env.game.getGameState()
            agent.observe(state, action, reward, new_state, env.game_over())

        # Print only the episode summary
        print(f"Episode Summary - Score: {score}, Flaps: {flap_count}")
        
        nb_episodes -= 1
        score = 0
        agent.epsilon = max(0.01, agent.epsilon * 0.999)  # decay epsilon


agent = FlappyAgent()
train(500, agent)
