Scikit-learn을 활용하여, Data를 처리하는 방법에 대해 이야기 하고자합니다. 

dataset 생성 , 전처리 , input data 활용 등

## Generate Test Datasets

[How to Generate Test Datasets in Python with scikit-learn](https://machinelearningmastery.com/generate-test-datasets-python-scikit-learn/)

Scikit-learn(0.22.2 version)에서는 손쉽게 테스트 데이터를 만들수 있도록 제공해줍니다. 
multi-class classification, binary classification, linear regression 3가지 유형의 훈련에 맞는 테스트 데이터를 생성할 수 있습니다. 


### Classification

Classification 에서 가장 큰 어려움은 데이터에 labeling을 하는 부분입니다. 
자연어처리, 영상처리 등 supervised learning 기반의 learning들이 가지고 있는 공통적인 난제입니다. 

classification 테스트를 할수 있도록 3가지 유형(blobs, moons, circles)으로, scikit-learn을 통해 하나씩 보겠습니다. 

1. Blobs Classification Problem

> scikit-learn의 Gaussian 분포를 따르는 blob(점)을 make_blobs() 함수를 통해 쉽게 생성할 수 있습니다. 
> 간단히 말하면, labeling이 된 군집 데이터를 생성해주는 함수입니다. 
> 다음은 제어할 수 있는 [속성에 대한 정보](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html)입니다. 
> - n_samples(default=100): 표본 데이터의 수를 의미 / int -> 각 군집별로 개수를 의미하며, array -> 각 군집별 개수
> - n_features(default=2): 독립 변수의 수(배열 기준이므로 0,1,2  3개 생성)
> - centers(default=3): 생성할 클러스터의 수 혹은 중심
> - cluster_std(default=1.0): 클러스터의 표준펀차
> - center_box(default=(-10.0, 10.0)): 클러스터내 생성 범위
> - shuffle(default=True): 혼합
> - random_state(default=None):Random 기준 정의

> 기본적인 예시는 다음과 같습니다.

> ```python
> 
> from sklearn.datasets import make_blobs
> from matplotlib import pyplot
> from pandas import DataFrame
> # generate 2d classification dataset
> X, y = make_blobs(n_samples=100, centers=3, n_features=2)
> # scatter plot, dots colored by class value
> df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
> colors = {0:'red', 1:'blue', 2:'green'}
> fig, ax = pyplot.subplots()
> grouped = df.groupby('label')
> for key, group in grouped:
>     group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
> pyplot.show()
> 
> ```

2. Moons Classification Problem
> 소용돌이 형태 비선형 클래스의 Binary Classification 을 위한 테스트 데이터는 make_moons() 함수를 통해 만들 수 있습니다. 
> [속성 정보]https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html#sklearn.datasets.make_moons)는 다음과 같습니다. 
> - n_samples(default=100): 생성대상 표본 데이터수
> - shuffle(default=True): 혼합여부
> - noise(default=None): Gaussian 분포 표준편차를 따르는 noise data 추가 여부
> - random_state(default=None): Random 기준 정의

> ```python
> 
> from sklearn.datasets import make_moons
> from matplotlib import pyplot
> from pandas import DataFrame
> # generate 2d classification dataset
> X, y = make_moons(n_samples=100, noise=0.1)
> # scatter plot, dots colored by class value
> df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
> colors = {0:'red', 1:'blue'}
> fig, ax = pyplot.subplots()
> grouped = df.groupby('label')
> for key, group in grouped:
>     group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
> pyplot.show()
> 
> ```

3. Circle Classification Problem

> 동심원 형태의 데이터 분류를 테스트를 위해 make_circles() 함수를 제공합니다
> 



