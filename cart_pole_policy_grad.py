import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.distributions import Categorical

env = gym.make('CartPole-v0') 

class Policy(nn.Module):
    
  def __init__(self, layers):
      
    super(Policy, self).__init__()
    self.hidden = nn.ModuleList()
    
    for n_i, n_o in zip(layers, layers[1:]):
        
      self.hidden.append(nn.Linear(n_i, n_o))
      
    self.save_log_probs = []
    self.save_rewards = []

  def forward(self, x):
      
    L = len(self.hidden)
    
    for(l, linear_transform) in zip(range(L), self.hidden):
        
      if l < L - 1:
          
        x = F.relu(linear_transform(x))
        
      else:
          
        x = F.softmax(linear_transform(x), dim=1)
        
    return x

layers = [4, 10, 10, 2]
policy = Policy(layers)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    
def SelectAction(state):
    
  state = torch.from_numpy(state).float().unsqueeze(0)
  probs = policy(state)
  m = Categorical(probs)
  action = m.sample()
  policy.save_log_probs.append(m.log_prob(action))
  
  return action.item()

def FinishEpisode(gamma, eps):
    
  R, policy_loss, rewards = 0, [], []
  
  for r in policy.save_rewards[::-1]:
      
    R = r + gamma * R
    rewards.insert(0, R)
    
  rewards = torch.tensor(rewards)
  rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
  
  for log_prob, reward in zip(policy.save_log_probs, rewards):
      
    policy_loss.append(-log_prob * reward)
    
  optimizer.zero_grad()
  policy_loss = torch.cat(policy_loss).sum()
  policy_loss.backward()
  optimizer.step()
  
  del policy.save_rewards[:]
  del policy.save_log_probs[:]

def PlotRunningAverage(totalrewards):
    
  N = len(totalrewards)
  running_avg = np.empty(N)
  
  for t in range(N):
      
    running_avg[t] = totalrewards[max(0, t - 100):(t + 1)].mean()
    
  plt.plot(running_avg)
  plt.title("Running Average")
  plt.show()
  
N , gamma, eps = 1000, 0.95, 1e-8
totalrewards = np.empty(N)

for n in range(N):
    
  state = env.reset()
  totalreward, i, done = 0, 0, False
  
  while not done and i < 10000:
      
    action = SelectAction(state)
    state, reward, done, _ = env.step(action)
    totalreward += reward
    policy.save_rewards.append(reward)
    i += 1
    
  totalrewards[n] = totalreward
  FinishEpisode(gamma, eps)
  
  if (n + 1) % 100 == 0:
      
    print("Average reward for last 100 episodes:", totalrewards[n - 99:n + 1].mean())
    
PlotRunningAverage(totalrewards)