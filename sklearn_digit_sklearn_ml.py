from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

digits = datasets.load_digits()
X = digits['data']
y = digits['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))  # 0.8367003367003367
