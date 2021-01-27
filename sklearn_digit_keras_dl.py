from sklearn import datasets
from sklearn.model_selection import train_test_split

# get data
digits = datasets.load_digits()
X = digits['data']
y = digits['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
