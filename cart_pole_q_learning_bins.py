import gym
import numpy as np
import matplotlib.pyplot as plt

def BuildState(features):
    
    return int("".join(map(lambda feature: str(int(feature)), features)))

def ToBin(value, bins):
    
    return np.digitize(x=[value], bins=bins)[0]

class FeatureTransformer:
    
    def __init__(self):
        
        self.cart_position_bins = np.linspace(-2.4, 2.4, 9)
        self.cart_velocity_bins = np.linspace(-2, 2, 9)
        self.pole_angle_bins = np.linspace(-0.4, 0.4, 9)
        self.pole_velocity_bins = np.linspace(-3.5, 3.5, 9)
        
    def transform(self, observation):
        
        cart_pos, cart_vel, pole_angle, pole_vel = observation
        
        return BuildState([
            ToBin(cart_pos, self.cart_position_bins),
            ToBin(cart_vel, self.cart_velocity_bins),
            ToBin(pole_angle, self.pole_angle_bins),
            ToBin(pole_vel, self.pole_velocity_bins)])
    
class Model:
    
    def __init__(self, env, feature_transformer):
        
        self.env = env
        self.feature_transformer = feature_transformer
        
        num_states = 10 ** env.observation_space.shape[0]
        num_actions = env.action_space.n
        self.Q = np.random.uniform(low=-1, high=1, size=(num_states, num_actions))
        
    def predict(self, s):
        
        x = self.feature_transformer.transform(s)
        
        return self.Q[x]
    
    def update(self, s, a, G):
        
        x = self.feature_transformer.transform(s)
        self.Q[x, a] += 10e-3 * (G - self.Q[x, a])
        
    def sample_action(self, s, eps):
        
        if np.random.random() < eps:
            
            return self.env.action_space.sample()
        
        else:
            
            p = self.predict(s)
            
            return np.argmax(p)

def PlayOne(model, eps, gamma):
    
    observation = env.reset()
    total_reward, i, done = 0, 0, False

    while not done and i < 10000:
        
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, info = env.step(action)
        total_reward += reward
        
        if done and i < 199:
            
            reward = -300  
            
        G = reward + gamma * np.max(model.predict(observation))
        model.update(prev_observation, action, G)        
        i += 1
        
    return total_reward

def PlotRunningAverage(total_rewards):
    
    N = len(total_rewards)
    running_avg = np.empty(N)
    
    for t in range(N):
        
        running_avg[t] = total_rewards[max(0, t - 100):(t + 1)].mean()
        
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()

env = gym.make('CartPole-v0')
ft = FeatureTransformer()
model = Model(env, ft)
N, gamma = 1000, 0.9
total_rewards = np.empty(N)

for n in range(N):
    
    eps = 1.0 / np.sqrt(n + 1)
    total_reward = PlayOne(model, eps, gamma)
    total_rewards[n] = total_reward
    
    if (n + 1) % 100 == 0:

        print("Average reward for last 100 episodes:", total_rewards[n - 99:n + 1].mean())
        
PlotRunningAverage(total_rewards)