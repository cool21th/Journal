[Machine Learning Algorithm Recipes in scikit-learn](https://machinelearningmastery.com/get-your-hands-dirty-with-scikit-learn-now/)

이번 글은 처음 머신러닝을 접하거나, 하고싶은 데 망설이고 있는 사람들을 위한 가이드입니다. 
모든 알고리즘을 해볼 필요는 없습니다. 
코드를 돌려보면서 알고리즘에 대한 공부도 함께 진행하면 쉽게, Scikit-learn에 익숙해질 것입니다. 

### Logistic Regression

Logistic regression은 0과 1사이에서의 확률을 예측하는 모델입니다. 
다중 클래스 모델인 경우, 하나의 클래스당 한개의 모델이 적용됩니다. 

다음 예제는 쉽게 볼수 있는 꽃 예측 입니다

```python

# Logistic Regression
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
# load the iris datasets
dataset = datasets.load_iris()
# fit a logistic regression model to the data
model = LogisticRegression()
model.fit(dataset.data, dataset.target)
print(model)
# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
```

### Naive Bayes

Naive Bayes 모델은 Bayes 이론을 사용해서 클래스 변수의 각 속성들의 조건부 관계를 가지고 만듭니다. 

```python
# Gaussian Naive Bayes
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
# load the iris datasets
dataset = datasets.load_iris()
# fit a Naive Bayes model to the data
model = GaussianNB()
model.fit(dataset.data, dataset.target)
print(model)
# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
```

### k-Nearest Neighbor

k-Nearest Neighbor(kNN) 은 데이터들 사이에 거리를 기준으로 유사 데이터들로 구분하는 모델입니다. 

```python

# k-Nearest Neighbor
from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
# load iris the datasets
dataset = datasets.load_iris()
# fit a k-nearest neighbor model to the data
model = KNeighborsClassifier()
model.fit(dataset.data, dataset.target)
print(model)
# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
```

### Classification and Regression Tree

Classification and Regression Tree(CART) 는 흔히 말하는 Tree 기반의 모델알고리즘 입니다. 
Tree 기반 모델의 가장 큰 장점은 데이터를 잘 분리하거나 잘 예측할수 있도록 데이터를 구성하도록 해줍니다.

```python

# Decision Tree Classifier
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
# load the iris datasets
dataset = datasets.load_iris()
# fit a CART model to the data
model = DecisionTreeClassifier()
model.fit(dataset.data, dataset.target)
print(model)
# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

```

### Support Vector Machine

SVM(Support Vector Machine)은 두개의 그룹으로 클래스를 잘 분리한 공간으로 변환시키는 방법입니다.
멀티 클래스에 대한 분류와 최소 허용오차 회귀 모델을 만들 수 있습니다. 

```python
# Support Vector Machine
from sklearn import datasets
from sklearn import metrics
from sklearn.svm import SVC
# load the iris datasets
dataset = datasets.load_iris()
# fit a SVM model to the data
model = SVC()
model.fit(dataset.data, dataset.target)
print(model)
# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
```










