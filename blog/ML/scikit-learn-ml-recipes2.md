Ensemble, Pipeline 등의 기법, 모델 저장 및 로드에   Scikit-learn을 가지고 손쉽게 활용하는 부분에 대해 이야기해보겠습니다. 


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


## Machine Learning Workflows with Pipelines

[Automated Machine Learning Workflows with Pipelines in Python and scikit-learn](https://machinelearningmastery.com/automate-machine-learning-workflows-pipelines-python-scikit-learn/)

Scikit-learn에서는 Machine learning workflow 를 정의해주는 표준적이고 일반적인 Pipeline을 제공해주고 있습니다.
Pandas로 데이터를 전처리하고, Scikit-learn에서 모델을 만들고, 평가하는 가장 일반적인 방법에서 시작하겟습니다.

Pipeline을 쓰는 이유는 데이터간의 상관관계를 줄이고, 모델의 성능을 올리는 효과적인 방법중 하나입니다.
Data Preparation과 Feature Extraction 과정에서 Pipeline을 활용한 방법을 설명하도록 하겠습니다.

### Pipeline 1: Data Preparation and Modeling

머신러닝 모델을 Production 상황에서 잘 돌아가려면, trainig set와 test set의 분리를 명확하게 해줘야 합니다. 

예를들어 Learning 전에 정규화 또는 표준화를 훈련데이터 셋 전체에 취하는 방법은 적절한 테스트 방법은 아닙니다. 
왜냐하면 훈련에 쓰일 데이터셋이 테스트에 쓰일 데이터셋에 영향을 받을 가능성이 높기 때문입니다. 

Pipeline을 만드는 것은 교차 검층 절차를 통해 표준화를 각 Fold에 한해서만 적용합니다. 

좀더 쉽게 이해하기 위해 다시 설명을 하면, 전체 데이터 대해 정규화, 표준화를 수행후 데이터를 여러개로 나눈다 하더라도
이미 전체 데이터의 영향을 받은 형태가 됩니다. 
하지만, 데이터를 특정기준으로 나누고(K_fold) 난 후, 나눠진 데이터에서만 정규화, 표준화를 진행한다면 해당 셋 내에서만 진행이 되어
좀더 성능 좋은 모델이 생성됩니다. 

이번 예시를 통해 Pipeline을 통해 Standardize the data와 LDA(Linear Discriminant Anlysis)를 적용한 모델 생성을 볼 수 있습니다. 

```python

# Create a pipeline that standardizes the data then creates a model
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# create pipeline
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('lda', LinearDiscriminantAnalysis()))
model = Pipeline(estimators)
# evaluate pipeline
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

```


### Pipeline 2: Feature Extraction and Modeling

Feature extraction 역시 데이터간의 영향을 받는 과정중 하나입니다. 
그래서, Data Preparation 과정과 마찬가지로 훈련 데이터셋의 데이터를 반드시 제한해줘야 합니다. 

Pipeline 모듈에서는 FeatureUnion 함수를 제공해 교차 검증을 수행하면서, 
훈련대상의 각 데이터 셋에서 여러 feature들을 선택하고 추출한 후 
대상 feature들을 결합이 가능하게 해줍니다. 
이렇게 결합된 feature 들로 최종 모델을 만들 수 있습니다.

다음 예시의 순서는 다음과 같습니다.

1. Feature Extraction with Principal Component Analysis (3 features)
2. Feature Extraction with Statistical Selection (6 features)
3. Feature Union
4. Learn a Logistic Regression Model

```python

# Create a pipeline that extracts features from the data then creates a model
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
# load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# create feature union
features = []
features.append(('pca', PCA(n_components=3)))
features.append(('select_best', SelectKBest(k=6)))
feature_union = FeatureUnion(features)
# create pipeline
estimators = []
estimators.append(('feature_union', feature_union))
estimators.append(('logistic', LogisticRegression()))
model = Pipeline(estimators)
# evaluate pipeline
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

```


## Save and Load Model

[Save and Load Machine Learning Models in Python with Scikit-learn](https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/)

지금까지 모델을 훈련하고 평가하는데 까지 배웠다면, 이제는 모델을 파일로 저장하고, 로딩하는 영역에 대해 논의하겠습니다. 


### Finalize Your Model with pickle

Pickle은 Python 객체를 Serialization 한 형태로 만드는 표준입니다. 
객체의 Serialization이 가지고 있는 의미는 Byte로 변환시켜 쉽게 저장하거나 전송할 수 있도록 하는 변환방법입니다. 

Pickle 라이브러리를 통해 우리는 쉽게 머신러닝 모델을 저장하고 로딩할 수 있습니다. 

다음 예를 통해 모델을 저장하고 로딩하는 것을 확인할 수 있습니다. 

```python

# Save Model Using Pickle
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
# Fit the model on training set
model = LogisticRegression()
model.fit(X_train, Y_train)
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)

```

### Finalize Your Model with joblib

Joblib 는 Python job의 pipeline을 제공하는 pScipy의 에코시스템중 하나입니다. 
이는 효과적으로 NumPy와 쉽게 연동을 지원해줍니다. 

많은 매개변수가 필요하거나 Knn(K-Nearest Neighbor)와 같은 전체 데이터 세트를 저장할 경우 유용한 방법입니다. 

아래 예를 통해 알 수 있듯이 pickle과 비교해서 사용하는 방법이 크게 다르지 않습니다.

```python

# Save Model Using joblib
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import joblib
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
# Fit the model on training set
model = LogisticRegression()
model.fit(X_train, Y_train)
# save the model to disk
filename = 'finalized_model.sav'
joblib.dump(model, filename)
 
# some time later...
 
# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, Y_test)
print(result)

```

### Tips for Finalizing Your Model

머신러닝 모델을 저장하고 로딩할 때 3가지를 기준으로 고려하면 향후 도움이 될 것입니다. 

- Python version : Python Major version 이 같아야 합니다. 때로는 Minor version도 확인이 필요할 수 있습니다. 
- Library version : 주요 library version이 같아야 합니다. NumPy와 Scikit-learn은 거의 제한을 두지는 않습니다.
- Manual Serialization : 학습된 모델의 매개 변수를 수동으로 다른 플랫폼으로 옮길 수 있습니다. 학습된 모델의 매개변수를 사용하는 것은 쉽게 코드로 구현이 가능합니다.

만들어진 모델의 라이브러리 버전을 기록을 해놓으면(requirements 등) 새로운 플랫폼에서 load할지 다시 훈련 시킬지를 판단하기 쉬워집니다.



