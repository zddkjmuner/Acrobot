import numpy as np
import gym
from collections import deque
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import matplotlib.pyplot as plt
torch.manual_seed(0)

# Def policy as Neural Network parametraizion


class PolicyNN(nn.Module):
    def __init__(self, inpt_size=6, outpt_size=3, h_size1=32, h_size2=64):
        super(PolicyNN, self).__init__()
        self.FC1 = nn.Linear(inpt_size, h_size1)
        self.FC2 = nn.Linear(h_size1, outpt_size)

    def forward(self, x):
        ans = F.relu(self.FC1(x))
        ans = self.FC2(ans)
        output = F.softmax(ans, dim=1)
        return output
    
    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item() - 1, m.log_prob(action)


env = gym.make('Acrobot-v1')
env.seed(0)
print('observation space:', env.observation_space)
print('action space:', env.action_space)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = PolicyNN().to(device)
rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=rate)


# Reinforce parameters:
NUM_EP = 8000
TIME_STEP = 1000
GAMMA = 0.99
EACH_T = 100

def Update(num_ep, time_step, gamma, each_t):
    scores_deque = deque(maxlen=100)
    scores = []
    each_scores = []
    for i_episode in range(1, num_ep+1):
        saved_log_probs = []
        rewards = []
        state = env.reset()
        for t in range(time_step):
            action, log_prob = model.get_action(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break 
        each_scores += rewards
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))
        
        discounts = [gamma**i for i in range(len(rewards)+1)]
        R = sum([a*b for a,b in zip(discounts, rewards)])
        
        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        if i_episode % each_t == 0:
            torch.save(model.state_dict(), 'policy_model.pth')
            print('Epoch:{}/{} Reward: {:.4f}'.format(i_episode, NUM_EP,np.mean(scores_deque)))
    # print()
    return scores, each_scores

scores, each_s = Update(NUM_EP, TIME_STEP, GAMMA, EACH_T)
plt.figure()
# plt.plot(range(len(each_s)), each_s)
plt.plot(range(len(scores)), scores)
plt.xlabel("Number of Iterations")
plt.ylabel("Rewards")
plt.title("Plot of Rewards VS Iterations")
plt.show()
# print(len(scores))
