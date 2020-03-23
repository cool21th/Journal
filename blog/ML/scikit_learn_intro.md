[A Gentle Introduction to Scikit-Learn: A Python Machine Learning Library](https://machinelearningmastery.com/a-gentle-introduction-to-scikit-learn-a-python-machine-learning-library/)

### Overview

머신러닝을 떠나서, python 코딩을 하는 사람들중 Scikit-learn이라는 Library를 못본 사람은 없었을 것입니다.

Scikit-learn은 머신러닝 개발에서 부터 운영까지 충족시켜주는 아주 강력한 라이브러리입니다.


### Scikit-learn 소개

Scikit-learn 은 다양한 supervised, unsupervised Learning 알고리즘을 제공하는 Python Library 입니다

BSD 라이센스 정책을 따르며, 학업적/ 상업적으로 사용하는데 있어서 제약이 없습니다. 

Scikit-learn Library는 SciPy(Scientific Python)을 기반으로 구성되어 있으며, 
Numpy, SciPy, Matplotlib, IPython, Sympy, Pandas 등의 라이브러리들도 함께 구성하고 있습니다. 

- Numpy: Base n-dimensional array package
- SciPy: Fundamental library for scientific computin
- Matplotlib: Comprehensive 2D/3D plotting
- Sympy : Symbolic mathematics
- Pandas : Data Structure and analysis

SciKits은 SciPy 라이브러리의 확장형 모듈을 의미합니다. 
그래서 scikit-learn 은 SciPy 모듈과 learning 알고리즘을 제공하는 모듈이라고 쉽게 이해할 수 있습니다. 

주 사용 언어는 Python이지만, c 라이브러리들을 활용해서 LAPACK, LibSVM, cython을 활용하여 
배열 및 행렬 연산에 있어서 Numpy와 같은 성능을 발휘합니다. 


### Scikit-learn Feature

Scikit-learn 은 Numpy, Pandas와 다르게 데이터를 로딩, 조작, 요약하는데 중점을 두고 있지 않습니다. 

주요 제공하는 모델은 다음과 같습니다. 

- Clustering: KMeans 와 같이 Unlabeled 된 데이터 그룹핑 기능 제공(흔히 비지도 학습의 대표적)
- Cross Validation: unseen data(테스트 or 실제 들어올 데이터 등)에 대한 supervised model의 성능 측정 기능 제공
- Datasets : 모델의 추이를 확인 목적으로 특정 속성을 가진 테스트/ 생성용 데이터 세트 관련 기능제공
- Dimensionality Reduction: PCA(Principal component analyis)와 같이 요약, 시각화 및 feature 선택을 목적으로 데이터 속성(차원 등) 감소
- Ensemble methods: supervised 모델들의 예측 조합 기능 지원
- Feature extraction: 이미지, 텍스트 데이터 속성 정의
- Feature selection: supervised 모델들로부터 의미있는 속성 정의
- Parameter Tuning: supervised 모델의 최선의 예측결과 도출
- Manifold Learning: 복잡한 다차원의 데이터 요약및 묘사
- Supervised Learning: Linear, 나이브베이즈, Decision Tree, SVM, nn 등 다양한 모델 지원

### Example: Classification and Regression trees

```python
# Sample Decision Tree Classifier
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

참고자료: \
[scikit-learn homepage](https://scikit-learn.org/)\
[scikit-learn github page](https://github.com/scikit-learn)

