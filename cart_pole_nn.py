import gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler

class Network(nn.Module):
    
    def __init__(self, layers):
        
        super(Network, self).__init__()
        self.hidden = nn.ModuleList()
        
        for n_i, n_o in zip(layers, layers[1:]):
            
            self.hidden.append(nn.Linear(n_i, n_o))
            
    def forward(self, x):
        
        L = len(self.hidden)
        
        for(l, linear_transform) in zip(range(L), self.hidden):
            
            if l < L - 1:
                
                x = torch.relu(linear_transform(x))
                
            else:
                
                x = linear_transform(x)
                
        return x
    
def PartialFit(x, y, net):
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
    yhat = net(x).view(1,-1)
    loss = criterion(yhat, y.view(1,-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
class FeatureTransformer:
    
  def __init__(self, env, n_components=1000): 
      
    observation_examples = np.random.random((20000, 4)) * 2 - 1
    scaler = StandardScaler()
    scaler.fit(observation_examples)
    featurizer = FeatureUnion([
            ("rbf", RBFSampler(gamma=0.5, n_components=n_components))
            ])
    example_features = featurizer.fit_transform(scaler.transform(observation_examples))
    self.dimensions = example_features.shape[1]
    self.scaler = scaler
    self.featurizer = featurizer
    
  def transform(self, observations):
      
    scaled = self.scaler.transform(observations)
    
    return self.featurizer.transform(scaled)

class Model:
    
  def __init__(self, env, feature_transformer, hidden_layers, hidden_units):
      
    self.env = env
    self.nets = []
    self.feature_transformer = feature_transformer
    layers = [self.feature_transformer.dimensions]
    
    if hidden_layers == 1:
        
        layers.append(hidden_units)
        
    else:
        
        for hl in range(hidden_layers):
            
            layers.append(hidden_units[hl])
            
    layers.append(1)
    
    for i in range(env.action_space.n):
        
      net = Network(layers)
      self.nets.append(net)
      
  def predict(self, s):       
      
    x = self.feature_transformer.transform([s])
    x = torch.from_numpy(x).type(torch.FloatTensor)
    
    return np.stack([net(x).detach().numpy() for net in self.nets]).T

  def update(self, s, a, G):
      
    x = np.array(self.feature_transformer.transform([s]))
    x = torch.from_numpy(x).type(torch.FloatTensor)
    G = torch.from_numpy(G).type(torch.FloatTensor)
    PartialFit(x, G, self.nets[a])    
    
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
    G = np.array(reward + gamma*np.max(next_[0]))
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
model = Model(env, ft, 1, 10)
N, gamma = 1000, 0.99
totalrewards = np.empty(N)

for n in range(N):
    
  eps = 1.0 / np.sqrt(n + 1)
  totalreward = PlayOne(model, env, eps, gamma)
  totalrewards[n] = totalreward
  
  if (n + 1) % 100 == 0:
      
    print("Average reward for last 100 episodes:", totalrewards[n - 99:n + 1].mean())
    
PlotRunningAverage(totalrewards)