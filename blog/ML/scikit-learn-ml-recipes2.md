Ensemble, Pipeline 등의 기법을  Scikit-learn을 가지고 손쉽게 활용하는 부분에 대해 이야기해보겠습니다. 


## Ensemble
[Ensemble Machine Learning Algorithms in Python with scikit-learn](https://machinelearningmastery.com/ensemble-machine-learning-algorithms-python-scikit-learn/)

Ensemble 기법은 주어진 데이터셋을 활용해서 모델의 정확도를 높이는 기법중 하나입니다. 
주로 쓰는 방법 3가지를 이번 기회를 통해 배울 수 있습니다. 

- Boosting : 훈련 데이터셋을 서로다른 하위 샘플링을 통해 여러개 모델을 만드는 방식
- Bagging : 여러개의 모델을 순차적으로 훈련시키는 방법으로 먼저 만들어진 모델의 예측 오류를 조정하면서 훈련하는 방식
- Voting : 여러개의 모델을 만들고 간단한 통계를 사용하여 예측을 결합하는 방식


### Bagging Algorithm

1. Bagged Descision Trees

> Bagging 방식은 높은 분산을 가진 알고리즘에 효과적입니다. 가장 많이 쓰는 방법은 단연 Decision Tree 입니다. 

> 다음 예는 scikit-learn의 bagging classifier와 DecisionTreeclassifier를 혼합해서 사용하는 방법입니다. 

> ```python
> 
> # Bagged Decision Trees for Classification
> import pandas
> from sklearn import model_selection
> from sklearn.ensemble import BaggingClassifier
> from sklearn.tree import DecisionTreeClassifier
> url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
> names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
> dataframe = pandas.read_csv(url, names=names)
> array = dataframe.values
> X = array[:,0:8]
> Y = array[:,8]
> seed = 7
> kfold = model_selection.KFold(n_splits=10, random_state=seed)
> cart = DecisionTreeClassifier()
> num_trees = 100
> model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
> results = model_selection.cross_val_score(model, X, Y, cv=kfold)
> print(results.mean())
> 
> ```


2. Random Forest

> Random Forest 는 Bagged decision Tree방식을 확장한 형태로 각 트리의 훈련용 샘플들을 계속 변경하면서 가져옵니다. 
> 각 트리 들 사이의 상관관계를 줄이는 방법으로 구성되며, 분리되는 지점을 임의의 하위 집합으로 만 고려합니다.

> 다음 예를 통해 쉽게 확인할 수 있습니다.

> ```python
> 
> # Random Forest Classification
> import pandas
> from sklearn import model_selection
> from sklearn.ensemble import RandomForestClassifier
> url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
> names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
> dataframe = pandas.read_csv(url, names=names)
> array = dataframe.values
> X = array[:,0:8]
> Y = array[:,8]
> seed = 7
> num_trees = 100
> max_features = 3
> kfold = model_selection.KFold(n_splits=10, random_state=seed)
> model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
> results = model_selection.cross_val_score(model, X, Y, cv=kfold)
> print(results.mean())
> 
> ```


3. Extra Trees([Extremely Randomized Trees classifier](https://www.geeksforgeeks.org/ml-extra-tree-classifier-for-feature-selection/))

> Extra Trees 기법은  RandomForest 의 확장한 형태로, Forest tree 내에 상관관계가 적은 decision tree들을 결합하여 결과를 도출한 방식입니다. 
> scikit-learn을 활용한 예시는 다음과 같습니다

> ```python
> 
> # Extra Trees Classification
> import pandas
> from sklearn import model_selection
> from sklearn.ensemble import ExtraTreesClassifier
> url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
> names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
> dataframe = pandas.read_csv(url, names=names)
> array = dataframe.values
> X = array[:,0:8]
> Y = array[:,8]
> seed = 7
> num_trees = 100
> max_features = 7
> kfold = model_selection.KFold(n_splits=10, random_state=seed)
> model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)
> results = model_selection.cross_val_score(model, X, Y, cv=kfold)
> print(results.mean())
> 
> ```


### Boosting Algorithms



