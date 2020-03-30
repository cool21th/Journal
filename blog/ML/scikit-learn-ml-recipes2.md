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

Boosting Algoritms은 모델들을 순차적으로 훈련시키면서 앞선 모델의 error를 반영하면서 진행하는 방식입니다. 
일반적으로 AdaBoost 와 Stochastic Gradient Boosting 두가지 방식이 있는데 이 두개를 중심으로 설명하겠습니다. 
최근에는 많이 쓰는 것은 XGBoost, LightGBM 등이 있는데, 현재 정확도가 아주 높은 알고리즘으로 Kaggle에서 널리 쓰이고 있습니다.



1. AdaBoost

> AdaBoost 는 Boosting 계열의 Ensemble 모델 중 첫 성공적인 모델입니다. 
> 일반적으로 모델이 얼마나 데이터 셋을 쉽게 분류하는지, 
> 순차적인 모델이 앞선 모델의 알고리즘에 얼마나 영향을 받는지를 기준으로 동작합니다.

> 다음은 AdaBoost의 예제입니다. 

> ```python
> 
> # AdaBoost Classification
> import pandas
> from sklearn import model_selection
> from sklearn.ensemble import AdaBoostClassifier
> url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
> names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
> dataframe = pandas.read_csv(url, names=names)
> array = dataframe.values
> X = array[:,0:8]
> Y = array[:,8]
> seed = 7
> num_trees = 30
> kfold = model_selection.KFold(n_splits=10, random_state=seed)
> model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
> results = model_selection.cross_val_score(model, X, Y, cv=kfold)
> print(results.mean())
> 
> ```

3. Voting Ensemble

> Voting은 Machine learning의 Ensemble 알고리즘에서 가장 간단한 형태의 조합입니다
> 먼저 두개 이상의 독립형 모델을 만든 후, voting 기준을 정한뒤 최종 예측결과를 도출하는 것입니다. 
> 모델들의 가중치를 경험적으로나 손수 지정하는 것은 어렵습니다.
> 특히나 하위 모델들의 가장 좋은 가중치만을 선정해서 적용하는 Stacking 방식은 Scikit-learn에서 제공해주고 있지 않습니다. 

> Scikit-learn에서 VotingClassifier 클래스를 통해 기본적은 voting ensemble을 만들 수 있습니다. 
> 다음 예는 logistic regression과 DecisionTreeClassifier, SVC(Support vector machine for classifier)를 조합한 형태입니다. 

> ```python
> 
> # Voting Ensemble for Classification
> import pandas
> from sklearn import model_selection
> from sklearn.linear_model import LogisticRegression
> from sklearn.tree import DecisionTreeClassifier
> from sklearn.svm import SVC
> from sklearn.ensemble import VotingClassifier
> url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
> names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
> dataframe = pandas.read_csv(url, names=names)
> array = dataframe.values
> X = array[:,0:8]
> Y = array[:,8]
> seed = 7
> kfold = model_selection.KFold(n_splits=10, random_state=seed)
> # create the sub models
> estimators = []
> model1 = LogisticRegression()
> estimators.append(('logistic', model1))
> model2 = DecisionTreeClassifier()
> estimators.append(('cart', model2))
> model3 = SVC()
> estimators.append(('svm', model3))
> # create the ensemble model
> ensemble = VotingClassifier(estimators)
> results = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)
> print(results.mean())
> 
> ```



