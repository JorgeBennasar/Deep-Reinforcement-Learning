import gym
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler

class SGDRegressor:
    
  def __init__(self, D):
      
    self.w = np.random.randn(D) / np.sqrt(D)
    self.lr = 0.1
    
  def partial_fit(self, X, Y):
      
    self.w += self.lr*(Y - X.dot(self.w)).dot(X)
    
  def predict(self, X):
      
    return X.dot(self.w)

class FeatureTransformer:
    
  def __init__(self, env, n_components=1000): 
      
    observation_examples = observation_examples = np.random.random((20000, 4)) * 2 - 1
    scaler = StandardScaler()
    scaler.fit(observation_examples)
    featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=0.05, n_components=n_components)),
            ("rbf2", RBFSampler(gamma=1.0, n_components=n_components)),
            ("rbf3", RBFSampler(gamma=0.5, n_components=n_components)),
            ("rbf4", RBFSampler(gamma=0.1, n_components=n_components))
            ])
    example_features = featurizer.fit_transform(scaler.transform(observation_examples))
    self.dimensions = example_features.shape[1]
    self.scaler = scaler
    self.featurizer = featurizer
    
  def transform(self, observations):
      
    scaled = self.scaler.transform(observations)
    
    return self.featurizer.transform(scaled)

class Model:
    
  def __init__(self, env, feature_transformer):
      
    self.env = env
    self.models = []
    self.feature_transformer = feature_transformer
    
    for i in range(env.action_space.n):
        
      model = SGDRegressor(feature_transformer.dimensions)
      model.partial_fit(feature_transformer.transform([env.reset()]), [0])
      self.models.append(model)
      
  def predict(self, s):
      
    X = self.feature_transformer.transform([s])
    
    return np.stack([m.predict(X) for m in self.models]).T

  def update(self, s, a, G):
      
    X = self.feature_transformer.transform([s])
    self.models[a].partial_fit(X, [G])
    
  def sample_action(self, s, eps):
      
    if np.random.random() < eps:
        
      return self.env.action_space.sample()
  
    else:
        
      return np.argmax(self.predict(s))

def PlayOne(model, env, eps, gamma):
    
  observation = env.reset()
  totalreward, i, done = 0, 0, False

  while not done and i < 10000:
      
    action = model.sample_action(observation, eps)
    prev_observation = observation
    observation, reward, done, info = env.step(action)
    next_ = model.predict(observation)
    G = reward + gamma*np.max(next_[0])
    model.update(prev_observation, action, G)
    totalreward += reward
    i += 1
    
  return totalreward

def PlotRunningAverage(totalrewards):
    
  N = len(totalrewards)
  running_avg = np.empty(N)
  
  for t in range(N):
      
    running_avg[t] = totalrewards[max(0, t - 100):(t + 1)].mean()
    
  plt.plot(running_avg)
  plt.title("Running Average")
  plt.show()

env = gym.make('CartPole-v0')
ft = FeatureTransformer(env)
model = Model(env, ft)
N, gamma = 1000, 0.99 
totalrewards = np.empty(N)
  
for n in range(N):
      
  eps = 1.0 / np.sqrt(n + 1)
  totalreward = PlayOne(model, env, eps, gamma)
  totalrewards[n] = totalreward
    
  if (n + 1) % 100 == 0:

    print("Average reward for last 100 episodes:", totalrewards[n - 99:n + 1].mean())
  
PlotRunningAverage(totalrewards)