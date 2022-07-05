import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('mnist.csv')
data = df.values
x = data[:, 1:]
y = data[:, 0]
spli = int(0.8*x.shape[0])
x_train = x[:spli, :]
y_train = y[:spli]
x_test = x[spli:, :]
y_test = y[spli:]


def dist(x1, x2):
    return np.sqrt(sum((x1-x2)**2))


def knn(X, Y, queryPoint, k=5):
    vals = []
    m = X. shape[0]
    for i in range(m):
        d = dist(queryPoint, x[i])
        vals.append((d, Y[i]))
    vals = sorted(vals)
    # Nearest/First K points
    vals = vals[:k]
    vals = np. array(vals)
    # print(vals)
    new_vals = np.unique(vals[:, 1], return_counts=True)
    # print(new vals)
    index = new_vals[1].argmax()
    pred = new_vals[0][index]
    return int(pred)


def img(x, y):
    img = x.reshape((28, 28))
    plt.imshow(img, cmap='gray')
    plt.show()
    print(y)


n = x_test.shape[0]
n = 1000
count = 0
for i in range(n):
    pred = knn(x_train, y_train, x_test[i], k=5)
    # print(pred)
    # print(y_test[i])
    if(pred == y_test[i]):
        count += 1
l = count/n
print("Accuracy = ", l*100, "%")
