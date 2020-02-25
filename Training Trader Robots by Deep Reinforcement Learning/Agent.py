
import numpy as np
import random

from collections import deque

from keras.models import Sequential
from keras.layers import Dense, Flatten


# Deep Q-learning Agent
class Agent:

    def __init__(self, look_back, action_size, n_features):

        self.look_back = look_back          # fixed window of historical prices
        self.action_size = action_size      # buy, sell, hold
        self.n_features = n_features
        
        self.memory = deque(maxlen=3000)    # list of experiences

        self.gamma = 0.95                   # discount rate
        self.epsilon = 1.0                  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.model = self.create_DQN()


    def create_DQN(self):
        """
        Function create_DQN to implement the deep Q-network as a MLP
        """

        # input: stock price
        # output: decision

        model = Sequential()
        model.add(Dense(30, input_shape=(self.look_back, self.n_features), activation='relu'))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(5, activation='relu'))
        model.add(Flatten())
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        return model


    def remember(self, state, action, reward, next_state, done):
        """
        Function remember to store states, actions ad rewards by appending elements to the memory list
        """
        self.memory.append((state, action, reward, next_state, done))


    def replay(self, batch_size):
        """
        Function replay to train the deep Q-network according to the experience replay strategy
        """
        
        # Random minibatch of experiences
        mini_batch = random.sample(self.memory, batch_size)

        # Information from each experience
        for state, action, reward, next_state, done in mini_batch:

            if done:
                # End of episode, make our target reward
                target = reward

            else:
                # estimate the future discounted reward
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            # Calculate the target associated to the current state
            target_f = self.model.predict(state)
            # Update the Q-value for the action according to Belmann equation
            target_f[0][action] = target

            # Train the DQN with the state and target_t
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def act(self, state):
        """
        Function act to decide which action to take according to the epsilon-greedy policy
        """

        if np.random.rand() <= self.epsilon:
            # The agent acts at random
            return np.random.randint(self.action_size)

        # Predict the Q-values based on the current state
        act_values = self.model.predict(state)

        # The agent take the action with higher Q-value
        return np.argmax(act_values[0])
