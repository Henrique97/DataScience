import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
df = pd.concat([pd.read_csv("hinselmann.csv"), pd.read_csv("green.csv"), \
                pd.read_csv("schiller.csv")])
df = df.drop(['experts::0', 'experts::1', 'experts::2', \
              'experts::3', 'experts::4', 'experts::5', \
              'consensus'], axis = 1)

fig = plt.figure(figsize=(20,5))
ax = fig.add_subplot(1, 1, 1)

# Plot mean and standard deviation for each feature
x = df.columns.tolist()
mean = df.mean(axis=0)
std = df.std(axis=0)

plt.plot(x, mean, 'o', color='red')
plt.plot(x, std, 'o', color='blue')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.xticks(rotation=90)

plt.show()

# Plot Quartiless for each feature
fig = plt.figure(figsize=(20,5))
ax = fig.add_subplot(1, 1, 1)

df.boxplot(x, showfliers = False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.xticks(rotation=90)

plt.show()