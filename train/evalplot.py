# For creating paper style performance comparation figures
# Curve Plotting by using seaborn and matplotlib
# haiyinpiao@qq.com

import matplotlib.pyplot as plt
import os
import pickle
import seaborn as sns
import pandas as pd

logroot = './logplot/'
with open(os.path.join(logroot+"evalplot"+".pkl"), "rb") as f:
    evalplot = pickle.load(f)


# linestyle = ['-', '--', ':', '-.']
# color = ['r', 'g', 'b', 'k']

df = pd.DataFrame(evalplot)

sns.set(style="whitegrid")
# sns.catplot(x="Eval_Iterations", y="Eval_Mean_Rewards", hue="Eval_Agents",data=df, saturation=.5,kind="bar", ci=None, aspect=.6)
sns.barplot(x="Iterations", y="Game results", hue="Agents",data=df)
plt.title("Win rates between SRAMA2C and expert hand-crafted bot")

plt.show()
