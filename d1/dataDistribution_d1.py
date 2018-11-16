import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
testSet = pd.concat([pd.read_csv("all_mean_in_class_test.csv"), pd.read_csv("all_mean_in_class_training.csv")])
testSet = testSet.drop(['Unnamed: 0', 'class'], axis = 1)

fig = plt.figure(figsize=(20,5))
ax = fig.add_subplot(1, 1, 1)

# Plot mean and standard deviation for each feature
x = testSet.columns.tolist()
mean = testSet.mean(axis=0)
std = testSet.std(axis=0)

plt.plot(x, mean, 'o', label = 'Mean', color='red')
plt.plot(x, std, 'o', label = 'Standard Deviation', color='blue')
plt.legend()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.xticks(rotation=90)

plt.show()

# Plot Quartiless for each feature
fig = plt.figure(figsize=(20,5))
ax = fig.add_subplot(1, 1, 1)

testSet.boxplot(x, showfliers = False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.xticks(rotation=90)

plt.show()