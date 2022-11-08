import gym
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler

class BaseModel:
    
  def __init__(self, D):
      
    self.w = np.random.randn(D) / np.sqrt(D)
    
  def partial_fit(self, X, Y, eligibility, lr=10e-3):
      
    self.w += lr * (Y - X.dot(self.w)) * eligibility
    
  def predict(self, X):
      
    X = np.array(X)
    
    return X.dot(self.w)

class FeatureTransformer:
    
  def __init__(self, env, n_components=1000): 
      
    observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
    scaler = StandardScaler()
    scaler.fit(observation_examples)
    featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=n_components))
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
    D = feature_transformer.dimensions
    self.eligibilities = np.zeros((env.action_space.n, D))
    
    for i in range(env.action_space.n):
        
      model = BaseModel(D)
      self.models.append(model)
      
  def predict(self, s):
      
    X = self.feature_transformer.transform([s])
    
    return np.array([m.predict(X)[0] for m in self.models])

  def update(self, s, a, G, gamma, lambda_):
      
    X = self.feature_transformer.transform([s])
    self.eligibilities *= gamma*lambda_
    self.eligibilities[a] += X[0]
    self.models[a].partial_fit(X[0], G, self.eligibilities[a])
    
  def sample_action(self, s, eps):
      
    if np.random.random() < eps:
        
      return self.env.action_space.sample()
  
    else:
        
      return np.argmax(self.predict(s))

def PlayOne(model, env, eps, gamma, lambda_):
    
  observation = env.reset()
  totalreward, i, done = 0, 0, False

  while not done and i < 10000:
      
    action = model.sample_action(observation, eps)
    prev_observation = observation
    observation, reward, done, info = env.step(action)
    G = reward + gamma * np.max(model.predict(observation)[0])
    model.update(prev_observation, action, G, gamma, lambda_)
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

env = gym.make('MountainCar-v0')
ft = FeatureTransformer(env)
model = Model(env, ft)
N, gamma, lambda_ = 300, 0.99, 0.7
totalrewards = np.empty(N)

for n in range(N):
    
  eps = 0.1 * (0.97 ** n)
  totalreward = PlayOne(model, env, eps, gamma, lambda_)
  totalrewards[n] = totalreward
  
  if (n + 1) % 10 == 0:
      
    print("Average reward for last 10 episodes:", totalrewards[n - 9:n + 1].mean())
    
PlotRunningAverage(totalrewards)