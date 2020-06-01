from acrobot import AcrobotEnv

import gym
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

from gym.wrappers.monitoring.video_recorder import VideoRecorder

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
state_dict = torch.load('policy_model.pth')

model = PolicyNN()
model.load_state_dict(state_dict)
model = model.to(device)

env = gym.make('Acrobot-v1')
def robo_view():
    # env = AcrobotEnv()
    env = gym.make('Acrobot-v1')
    video = VideoRecorder(env, path='./acrobot.mp4', enabled=True)
    state = env.reset()
    
    for i in range(10000):
        video.capture_frame()
        action, _ = model.get_action(state)
        env.render()
        state, rewards, finish, _ = env.step(action)
        if finish:
            break
        time.sleep(0.1)
    env.close()

robo_view()

# env = AcrobotEnv()

state = env.reset()
# done = False
# num_steps = 0
# while not done:
#     state, reward, done, _  = env.step(0)
#     env.render()
#     num_steps += 1
# print(num_steps)
# state = env.reset()
# env.close()
