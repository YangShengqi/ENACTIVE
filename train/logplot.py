# For creating paper style performance comparation figures
# Curve Plotting by using seaborn and matplotlib
# haiyinpiao@qq.com

import matplotlib.pyplot as plt
import os
import pickle
import seaborn as sns
from glob import glob

logroot = './logplot/'
# log_names = os.listdir(logroot)
log_names = glob(logroot+'logplot*.pkl')

Iterations = []
Fighter_Avg_Rewards = []
Fighter_Avg_Positive_Rewards = []
Bandit_Avg_Rewards = []
Bandit_Avg_Positive_Rewards = []

for name in log_names:
    with open(os.path.join(name), "rb") as f:
        logplot = pickle.load(f)
        Iterations = logplot["Iterations"]
        Fighter_Avg_Rewards.append(logplot["Fighter_Avg_Rewards"])
        Fighter_Avg_Positive_Rewards.append(logplot["Fighter_Avg_Positive_Rewards"])
        Bandit_Avg_Rewards.append(logplot["Bandit_Avg_Rewards"])
        Bandit_Avg_Positive_Rewards.append(logplot["Bandit_Avg_Positive_Rewards"])

linestyle = ['-', '--', ':', '-.']
color = ['r', 'g', 'b', '#34495e']

plt.figure(1)
sns.set(style="whitegrid")
sns.tsplot(time=logplot["Iterations"], data=Fighter_Avg_Rewards, color=color[0], linestyle=linestyle[0], condition="Agent")
sns.tsplot(time=logplot["Iterations"], data=Bandit_Avg_Rewards, color=color[2], linestyle=linestyle[0], condition="Opponent")

plt.ylabel("Average Rewards")
plt.xlabel("Iterations")

# plt.show()

plt.figure(2)
sns.set(style="whitegrid")
sns.tsplot(time=logplot["Iterations"], data=Fighter_Avg_Positive_Rewards, color=color[1], linestyle=linestyle[0], condition="Agent")
sns.tsplot(time=logplot["Iterations"], data=Bandit_Avg_Positive_Rewards, color=color[3], linestyle=linestyle[0], condition="Opponent")

plt.ylabel("Average Positive Rewards")
plt.xlabel("Iterations")

plt.show()
