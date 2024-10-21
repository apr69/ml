import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

data = pd.read_csv("digits_data.csv")
data.head()

data.describe()

data['label'].unique()

plt.figure(figsize=(10,6))
sns.countplot(x = "label", palette = "Set1",data = data)

image = data.iloc[7,1:]
image = image.values.reshape(28,28)
plt.imshow(image, cmap = 'gray')

y = data['label']
X = data.drop(columns = 'label')

X = X/255.0
print("X: ", X.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 10)

model = SVC(kernel = 'linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_pred

index = int(input("Index: "))
print(f"Predicted value : ",y_pred[index])
print("\nActual image: ")
X_test = X_test * 255
image = X_test.iloc[index, 0:]
image = image.values.reshape(28,28)
plt.imshow(image, cmap='gray')

from sklearn.metrics import accuracy_score
print("Accuracy : ",accuracy_score(y_test, y_pred))
