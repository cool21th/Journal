[Machine Learning Algorithm Recipes in scikit-learn](https://machinelearningmastery.com/get-your-hands-dirty-with-scikit-learn-now/)

이번 글은 처음 머신러닝을 접하거나, 하고싶은 데 망설이고 있는 사람들을 위한 가이드입니다. 
모든 알고리즘을 해볼 필요는 없습니다. 
코드를 돌려보면서 알고리즘에 대한 공부도 함께 진행하면 쉽게, Scikit-learn에 익숙해질 것입니다. 



## Basic Algorithm

#### Logistic Regression

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

#### Naive Bayes

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

#### k-Nearest Neighbor

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

#### Classification and Regression Tree

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

#### Support Vector Machine

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


##[Additional Regression Model](https://machinelearningmastery.com/spot-check-regression-machine-learning-algorithms-python-scikit-learn/)


1. Linear Regression

Linear Regression은 Input Variable들을 Gaussian 분포를 따른다고 가정하기에 Input variable은 Output Variable만 관련있다고 정의합니다. 

```python

# Linear Regression
import pandas
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pandas.read_csv(url, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = LinearRegression()
scoring = 'neg_mean_squared_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())

```

2. Ridge Regression

Ridge Regression은 Linear Regression을 확장한 것으로 coefficient value의 제곱의 합으로 측정하여
모델을 최소화 하기위해 변경된 손실 함수를 확장한 개념입니다. L2Norm 이라고도 합니다. 


```python

# Ridge Regression
import pandas
from sklearn import model_selection
from sklearn.linear_model import Ridge
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pandas.read_csv(url, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = Ridge()
scoring = 'neg_mean_squared_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())

```


3. LASSO Regression

Ridge와 마찬가지로 모델의 복잡성을 최소화 하는 방법으로 coefficient 값의 절대값의 합으로 손실함수를 변경합니다.


```python

# Lasso Regression
import pandas
from sklearn import model_selection
from sklearn.linear_model import Lasso
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pandas.read_csv(url, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = Lasso()
scoring = 'neg_mean_squared_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())

```


4. ElasticNet Regression

Ridge와 Lasso를 함께 사용하기 위한 목적입니다.

```python

# ElasticNet Regression
import pandas
from sklearn import model_selection
from sklearn.linear_model import ElasticNet
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pandas.read_csv(url, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = ElasticNet()
scoring = 'neg_mean_squared_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())

```



지금까지는 알고리즘을 적용해서 결과에 대해 보는 것을 보았고 , 이번에는 모델에 대해 평가하는 방법에 대해 이야기 하겠습니다. 
머신러닝에서 자주 사용하는 모델링 방법은 Classification(분류) 과 Regression(회귀)가 있습니다. 
먼저 Classification에 대해 언급하도록 하겠습니다. 

[Metrics To Evaluate Machine Learning Algoritms in Python](https://machinelearningmastery.com/metrics-evaluate-machine-learning-algorithms-python/)

## Classification metrics

classification에서 평가하는 방법은 classification accuracy, Log Loss, Area Under ROC Curve, Confusion Matrix, Classification Report 입니다

#### Classification accuracy

모든 예측한 값의 올바르게 예측한 비율을 측정하는 것입니다. 
가장 흔하게 쓰이는 방법이면서도, 가장 많이 잘못쓰이는 방법이기도 합니다. 
각 class에 동일한 수의 관측치가 있는 경우 적합하며, 모든 예측값과 오류 값이 중요하기도 하고, 그렇지 않기도 합니다. 

```python

# Cross Validation Classification Accuracy
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Accuracy: %.3f (%.3f)") % (results.mean(), results.std())

```


#### Log Loss

Log Loss(Logistic Loss)는 주어진 클래스와 예측 가능성을 기준으로 평가하는 방법을 취합니다. 

0, 1 사이로 측정값이 나오며, 예측의 신뢰의 비율에 따라 정확한 예측은 보상을 받고, 오류는 페널티를 받는 방식입니다. 

가장 밀접한 방법으로는 cross-entropy 입니다. 


```python

# Cross Validation Classification LogLoss
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = LogisticRegression()
scoring = 'neg_log_loss'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Logloss: %.3f (%.3f)") % (results.mean(), results.std())

```


#### Area Uner ROC Curve

ROC curve 아래 면적으로 평가하는 방법입니다. 
면적이 1이면 classification능력이 최대치를 의미합니다. 


```python

# Cross Validation Classification ROC AUC
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = LogisticRegression()
scoring = 'roc_auc'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("AUC: %.3f (%.3f)") % (results.mean(), results.std())


```


#### Confusion Matrix

Confusion Matrix는 2개 이상의 멀티 클래스에 대한 Accuracy를 측정하는 방법입니다

```python

# Cross Validation Classification Confusion Matrix
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
model = LogisticRegression()
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
matrix = confusion_matrix(Y_test, predicted)
print(matrix)

```


#### Classification Report

Scikit-learn에서는 Report 기능을 통해 편리하게 모델의 정확성을 파악할 수 있도록 해줍니다. 

```python

# Cross Validation Classification Report
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
model = LogisticRegression()
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
report = classification_report(Y_test, predicted)
print(report)

```

## Regression Metrics

Regression Metrics 는 Mean Absolute error, mean Squared error, R^2 이렇게 3가지 방법이 있습니다. 

#### Mean Absolute Error

예측값과 실제값 사이의 차이의 평균을 의미하며, 예측이 얼마나 잘못되고 있는지를 볼수 있습니다. 

```python

# Cross Validation Regression MAE
import pandas
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pandas.read_csv(url, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = LinearRegression()
scoring = 'neg_mean_absolute_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MAE: %.3f (%.3f)") % (results.mean(), results.std())

```


#### Mean Squared Error

오차에 대한 크기를 제곱한 값을 의미하며, root를 취한값은 RMSE 입니다.


```python

# Cross Validation Regression MSE
import pandas
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pandas.read_csv(url, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = LinearRegression()
scoring = 'neg_mean_squared_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MSE: %.3f (%.3f)") % (results.mean(), results.std())


```

#### R^2 Metrics

실제값에 대한 예측한 값의 적합도를 나타냅니다. 통계학적으로 결정계수(coefficient of determination)이라고 합니다

0과 1사이의 값으로 구성되면 1이 될수록 정확히 예측함을 말합니다.


```python

# Cross Validation Regression R^2
import pandas
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pandas.read_csv(url, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = LinearRegression()
scoring = 'r2'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("R^2: %.3f (%.3f)") % (results.mean(), results.std())

```

## [Comapare Machine Learning Algorithms](https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/)

Scikit-learn으로 머신러니을 개발하면서, 여러 모델들간의 성능을 비교하는 것이 매우 중요합니다. 

가장 좋은 머신러닝 모델을 선택하는 것은 어렵습니다. 각각의 모델의 성능 특성이 다르기 때문입니다. 

또한 Resampling 방법중 하나인 Cross validation 을 통해 모델이 얼마나 정확한지도 확인할 수 있습니다. 
그리고 Matplotlib를 이용해 간단한 시각화를 통해 명확하게 성능 구분도 가능합니다. 



다음은 6개의 모델들을 비교하는 코드 입니다. 
- Logistic Regression, Linear Discriminant Analysis, K-Nearest Neighbors, Classification and Regression tress, Naive bayes, SVM

```python

# Compare Algorithms
import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

```






