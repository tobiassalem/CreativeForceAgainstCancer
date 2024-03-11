# === [Load libraries] ===
from pandas import read_csv
from matplotlib import pyplot as plt

# === [Load the dataset] ===
dataset = read_csv("data/cancer_data.csv")

# === [Look at the dataset] ===
# shape
print("Dataset shape")
print(dataset.shape)

x = dataset[1:]
print("x = dataset[1:] shape")
print(x.shape)

# head
print(dataset.head(20))

# descriptions = statistical summary
print(dataset.describe())

# class distribution
print("Class distribution")
print(dataset.groupby('diagnosis(1=m, 0=b)').size())

# box plot
dataset.plot(kind='box')
plt.show()