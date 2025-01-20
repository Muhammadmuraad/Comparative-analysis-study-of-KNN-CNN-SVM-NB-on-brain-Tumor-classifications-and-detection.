import numpy as np
import matplotlib.pyplot as plt
# creating the dataset
data = {'svm':100, 'knn':86, 'navie bayes':80,
		}
algorithm= list(data.keys())
accuracy= list(data.values())

fig = plt.figure(figsize = (5, 5))

# creating the bar plot
plt.bar(algorithm, accuracy, color ='green',
		width = 0.2)

plt.xlabel("data")
plt.ylabel("ACCuracy")
plt.title("Accuracy of model")
plt.show()