import numpy as np
import matplotlib.pyplot as plt
import sys


reward=np.load('reward.npy')
cost=np.load('cost.npy')

print(f"reward: {reward.shape}, cost: {cost.shape}")


plt.plot(reward)
plt.plot(-cost)
plt.legend(["reward","cost"])
plt.ylabel('Avg return and Cost Values')
plt.savefig("plot_01.png")

import matplotlib.pyplot as plt

# Assuming 'reward' and 'cost' are lists or arrays containing the respective data

# Create a figure and two subplots
fig, axs = plt.subplots(1, 2, figsize=(6, 3))

# Plot reward on the first subplot
axs[0].plot(reward)
axs[0].set_ylabel('Average return')
axs[0].set_xlabel('Epochs')

# Plot cost on the second subplot
axs[1].plot(-cost,'r')  # Negate cost to make it positive for visualization
axs[1].set_ylabel('Cost Values')
axs[1].set_xlabel('Epochs')

# Add legend to each subplot
axs[0].legend(['reward'])
axs[1].legend(['cost'])

# Adjust layout to prevent overlapping
plt.tight_layout()

# Save the figure
plt.savefig("plot_01.pdf")

# Show the plot



# Define the directory path
import os

reward=np.load('logs/0.1reward.npy')
reward2=np.load('logs/0.2reward.npy')
reward3=np.load('logs/0.3reward.npy')
reward4=np.load('logs/0.4reward.npy')
reward5=np.load('logs/0.5reward.npy')
reward6=np.load('logs/0.6reward.npy')
reward7=np.load('logs/0.7reward.npy')
reward8=np.load('logs/0.8reward.npy')
reward9=np.load('logs/0.9reward.npy')

cost=np.load('logs/0.1cost.npy')
cost2=np.load('logs/0.2cost.npy')
cost3=np.load('logs/0.3cost.npy')
cost4=np.load('logs/0.4cost.npy')
cost5=np.load('logs/0.5cost.npy')
cost6=np.load('logs/0.6cost.npy')
cost7=np.load('logs/0.7cost.npy')
cost8=np.load('logs/0.8cost.npy')
cost9=np.load('logs/0.9cost.npy')


fig, axs = plt.subplots(1, 2, figsize=(6, 3))

axs[0].plot(reward,label='0.1')
axs[0].plot(reward2,label='0.2')
axs[0].plot(reward3,label='0.3')
axs[0].plot(reward4,label='0.4')
axs[0].plot(reward5,label='0.5')
axs[0].plot(reward6,label='0.6')
axs[0].plot(reward7,label='0.7')
axs[0].plot(reward8,label='0.8')
axs[0].plot(reward9,label='0.9')
axs[0].set_ylabel('Average return')
axs[0].set_xlabel('Epochs')

# Plot cost on the second subplot
axs[1].plot(-cost,label='0.1')
axs[1].plot(-cost2,label='0.2')
axs[1].plot(-cost3,label='0.3')
axs[1].plot(-cost4,label='0.4')
axs[1].plot(-cost5,label='0.5')
axs[1].plot(-cost6,label='0.6')
axs[1].plot(-cost7,label='0.7')
axs[1].plot(-cost8,label='0.8')
axs[1].plot(-cost9,label='0.9')

axs[1].set_ylabel('Cost Values')
axs[1].set_xlabel('Epochs')

# Add legend to each subplot
#axs[0].legend(['reward'])
#axs[1].legend(['cost'])
axs[0].legend(ncols=2,frameon=False)
axs[1].legend(ncol=2, loc='best',frameon=False)  # Legend with 2 columns, best location
# Adjust layout to prevent overlapping
plt.tight_layout()
plt.savefig("plot_1.pdf")
