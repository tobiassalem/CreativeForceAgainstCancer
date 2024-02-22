# === [Load libraries] ===
from pandas import read_csv
from matplotlib import pyplot as plt

# === [Load the dataset] ===
dataset = read_csv("data/cancer_data.csv")

# === [Look at the dataset] ===
# shape
print(dataset.shape)

# head
print(dataset.head(20))

# descriptions = statistical summary
print(dataset.describe())

# class distribution
print(dataset.groupby('diagnosis(1=m, 0=b)').size())

# box plot
dataset.plot(kind='box')
plt.show()