from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from collections import deque
from sklearn.neural_network import MLPRegressor
from collections import deque

class DeepQAgent:
    def __init__(self):
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.01  # Exploration rate
        self.epsilon_decay = 0.975 # Decay rate for exploration
        self.learning_rate = 0.05 # Learning rate
        self.batch_size = 2 # Batch size
        self.replay_buffer = deque(maxlen=100000)  # Replay buffer
        self.target_update_interval = 4  # Update target every 10 steps
        self.steps = 0  # Step counter

        # Main and target networks
        self.q_network = MLPRegressor(hidden_layer_sizes=(100, 10), activation='logistic', 
                                      learning_rate_init=self.learning_rate, max_iter=1, warm_start=True)
        self.target_network = MLPRegressor(hidden_layer_sizes=(100, 10), activation='logistic', 
                                           learning_rate_init=self.learning_rate, max_iter=1, warm_start=True)
        
        # Placeholder for initialization
        init_state = np.zeros((1, 4))
        init_target = np.zeros((1, 2))
        self.q_network.partial_fit(init_state, init_target)
        self.target_network.partial_fit(init_state, init_target)

    def normalize_state(self, state):
        """ Normalizes the state values to [-1, 1] """
        max_player_y, max_pipe_y, max_pipe_dist, max_vel = 400, 400, 300, 10
        normalized = np.array([
            2 * (state['player_y'] / max_player_y) - 1,
            2 * (state['next_pipe_top_y'] / max_pipe_y) - 1,
            2 * (state['next_pipe_dist_to_player'] / max_pipe_dist) - 1,
            2 * (state['player_vel'] / max_vel) - 1
        ])
        return normalized

    def store_transitions(self, s1, a, r, s2, end):
        """ Stores a transition in the replay buffer """
        self.replay_buffer.append((s1, a, r, s2, end))
        self.train()

    def train(self):
        """ Run experience replay if enough samples exist in the buffer """
        if len(self.replay_buffer) >= self.batch_size:
            self.experience_replay()

        # Update target network at specified interval
        self.steps += 1
        if self.steps % self.target_update_interval == 0:
            self.target_network.set_params(**self.q_network.get_params())

    def experience_replay(self):
        """ Sample a batch of experiences and train the Q-network """
        batch = random.sample(self.replay_buffer, self.batch_size)
        X_train, y_train = [], []

        for s1, a, r, s2, end in batch:
            s1_normalized = self.normalize_state(s1).reshape(1, -1)
            s2_normalized = self.normalize_state(s2).reshape(1, -1)

            q_values = self.q_network.predict(s1_normalized).flatten()
            q_next_values = self.target_network.predict(s2_normalized).flatten()

            # Target Q-value for current action
            target_q = r if end else r + self.gamma * np.max(q_next_values)
            q_values[a] = target_q  # Update the Q-value for the action taken

            X_train.append(s1_normalized.flatten())
            y_train.append(q_values)

        self.q_network.partial_fit(X_train, y_train)

    def training_policy(self, state):
            """ Returns the index of the action that should be done in state while training the agent.
                Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).
            """

            # Define key parameters
            pipe_buffer = 75  # Distance threshold for triggering flapping based on pipe proximity
            safe_zone = 46  # Safe zone near the pipe top for avoiding flapping
            flap_margin = 48  # Margin for avoiding unnecessary flapping near the pipe top

            # Check current position in relation to next pipe
            player_y = state['player_y']
            next_pipe_top_y = state['next_pipe_top_y']
            next_pipe_bottom_y = state['next_pipe_bottom_y']
            next_pipe_dist = state['next_pipe_dist_to_player']

            # 1. Avoid flapping when far above a lower pipe (Increased Drop Threshold)
            if player_y > next_pipe_top_y + safe_zone * 1.5 and next_pipe_dist < pipe_buffer:
                print("Descending: Do nothing")
                return 1  # Do nothing to allow natural descent

            # 2. Minimal flapping near high-altitude pipes to avoid overshooting
            if player_y < next_pipe_top_y + flap_margin and player_y > next_pipe_bottom_y:
                print("In safe zone near pipe: Do nothing")
                return 1  # Avoid unnecessary flapping when already in a safe position

            # 3. Detect if the next pipe is significantly higher and flap more aggressively
            if next_pipe_dist < pipe_buffer and (next_pipe_top_y - player_y > safe_zone * 2.5):
                print("High pipe detected: Flap aggressively")
                return 0  # Flap to gain height aggressively for high pipe

            # 4. Flap if below the safe zone for the upcoming pipe
            if player_y < next_pipe_top_y - safe_zone:
                print("Approaching gap: Flap to reach safe zone")
                return 0  # Flap to reach the next pipe’s gap

            # Epsilon-greedy action selection for exploration/exploitation
            if np.random.rand() < self.epsilon:
                return np.random.choice(self.actions)  # Random action for exploration
            else:
                state_discretized = self.discretize_state(state)
                return np.argmax(self.q_table.get(state_discretized, [0, 0]))  # Exploit learned Q-values


    def policy(self, state):
        """ Returns action after training (greedy) """
        state_normalized = self.normalize_state(state).reshape(1, -1)
        q_values = self.q_network.predict(state_normalized)
        return np.argmax(q_values)

    


class FlappyAgent(DeepQAgent):
    def __init__(self):
        super().__init__()
        # Define maximum values to create 15 intervals for each attribute
        self.max_player_y = 400      # Approximate max vertical position for the bird
        self.max_pipe_top_y = 400    # Approximate max top position for the pipe
        self.max_pipe_dist_to_player = 300  # Approximate max horizontal distance to the pipe
        self.velocity_intervals = 5  # Number of intervals for velocity (can be smaller if desired)
       
        self.interval_size = {
            "player_y": self.max_player_y / 15,
            "next_pipe_top_y": self.max_pipe_top_y / 15,
            "next_pipe_dist_to_player": self.max_pipe_dist_to_player / 15,
            "player_vel": 1  # Assuming velocity does not need further division
        }
        self.actions = [0, 1]  # 0 = flap, 1 = do nothing
        self.q_table = {}  # Initialize Q-table as a dictionary
        self.alpha = 0.1  
        self.gamma = 0.99 
        self.epsilon = 0.1  
        self.epsilon_decay = 0.998 


    def discretize_state(self, state):
        """Converts continuous game state into 15 intervals for each Q-table attribute."""
        player_y = int(state['player_y'] / self.interval_size['player_y'])
        pipe_y = int(state['next_pipe_top_y'] / self.interval_size['next_pipe_top_y'])
        dist_to_pipe = int(state['next_pipe_dist_to_player'] / self.interval_size['next_pipe_dist_to_player'])
        velocity = state['player_vel']  


        # Bound each discretized value to ensure it stays within 0-14 for 15 intervals
        player_y = min(player_y, 14)
        pipe_y = min(pipe_y, 14)
        dist_to_pipe = min(dist_to_pipe, 14)
       
        return (player_y, pipe_y, dist_to_pipe, velocity)


   
    def reward_values(self):
        """ returns the reward values used for training
       
            Note: These are only the rewards used for training.
            The rewards used for evaluating the agent will always be
            1 for passing through each pipe and 0 for all other state
            transitions.
        """
        return {"positive": 1.0, "tick": -0.01, "loss": -5.0}
   
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
        # print("state: %s" % state)


        # Discretize the current state
        discrete_state = self.discretize_state(state)


        # Initialize Q-values if state is not in Q-table
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = [0.0 for _ in self.actions]
       
        pipe_buffer = 80
        safe_zone = 45
        flap_margin = 48.5


        # Avoid flapping when near the upper boundary to avoid overshooting
        if state['player_y'] < state['next_pipe_top_y'] + flap_margin:
            return 1  # Do nothing
       
       
        if state['next_pipe_top_y'] > state['player_y'] + safe_zone and state['next_pipe_dist_to_player'] < pipe_buffer:
            return 1  # Avoid flapping for significant descent


        # Ensure there's space before flapping if near the pipe bottom
        if state['player_y'] > state['next_pipe_bottom_y'] - safe_zone:
            return 0  # Flap
       
        # Detect if the following pipe is significantly higher and increase flapping
        if state['next_pipe_dist_to_player'] < pipe_buffer and (state['next_pipe_top_y'] - state['player_y'] > safe_zone * 2.5):
            return 0  # Flap aggressively to gain height


        # Flap to approach the next pipe opening only if not already close to it
        if state['next_pipe_dist_to_player'] < pipe_buffer and state['player_y'] < state['next_pipe_top_y'] - flap_margin:
            return 0  # Flap to get closer


        # Avoid overshooting the next pipe if the gap is larger
        if state['next_pipe_top_y'] > state['player_y'] + safe_zone:
            return 1  # Drop more


        # Flap for pipes that are slightly higher and within range
        if state['next_pipe_top_y'] < state['player_y'] + flap_margin and state['next_pipe_dist_to_player'] < pipe_buffer:
            return 0  # Gain height for close, slightly higher pipe
       
        if state['next_pipe_top_y'] > state['player_y']:
            return 1  # Do nothing if the next pipe is lower


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
    """ Runs episodes of the game with the agent making moves """
    reward_values = agent.reward_values() if training else {"positive": 1.0, "tick": 0.0, "loss": 0.0}
   
    env = PLE(FlappyBird(), fps=30, display_screen=True, force_fps=True, reward_values=reward_values)
    env.init()


    while nb_episodes > 0:
        score = 0
        env.reset_game()
       
        while not env.game_over():
            current_state = env.game.getGameState()
            action = agent.training_policy(current_state) if training else agent.policy(current_state)
            reward = env.act(env.getActionSet()[action])
            score += reward
           
            if training:
                next_state = env.game.getGameState()
                agent.observe(current_state, action, reward, next_state, env.game_over())
       
        print(f"Score for this episode: {score}")
        nb_episodes -= 1
        # agent.epsilon = max(0.01, agent.epsilon * agent.epsilon_decay)  # decay epsilon






def train(nb_episodes, agent):
    reward_values = agent.reward_values()
   
    env = PLE(FlappyBird(), fps=30, display_screen=True, force_fps=False, rng=None,
            reward_values=reward_values)
    env.init()


    scores = []  # To store the score for each episode
    target_update_interval = 100 # to update target network every 100 steps
    steps = 0 # counter for steps to control target network updates


    while nb_episodes > 0:
        score = 0
        env.reset_game()
       
        while not env.game_over():
            state = env.game.getGameState()
            action = agent.training_policy(state)
            reward = env.act(env.getActionSet()[action])
            new_state = env.game.getGameState()
            agent.store_transitions(state, action, reward, new_state, env.game_over())
            score += reward
            steps += 1

            # update target network every target_update_interval steps
            if steps % target_update_interval == 0:
                agent.target_network.set_params(**agent.q_network.get_params())


        print(f"Score for this episode: {score}")
        scores.append(score)  # Record the episode score
        nb_episodes -= 1
        agent.epsilon = max(0.01, agent.epsilon * agent.epsilon_decay)  # Decay epsilon for exploration


    # Plot the scores to show progress
    plt.plot(scores)
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.title("Training Progress of Flappy Bird Agent")
    plt.show()


agent = FlappyAgent()
train(16, agent)
