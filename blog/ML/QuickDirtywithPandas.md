[Quick and Dirty Data Analysis with Pandas](https://machinelearningmastery.com/quick-and-dirty-data-analysis-with-pandas/)

### Overview

분석하기 위해 처음 raw 데이터를 보게 되면, 정리가 안되어 바로 머신러닝 모델링 하기에는 어려움이 있습니다
정리라는 기준은 테이블 형태만을 의미하는 것은 아닙니다. 
대부분의 데이터가 테이블 형태로 있다 하더라도, 칼럼의 기준이 업무에 적합할 뿐, 분석과는 동떨어져 있는 경우도 많기 때문입니다. 
머신러닝 표준 데이터 셋을 가지고 독자의 원하는 머신러닝을 적용하기에도 데이터를 정제해야 하는 경우가 많습니다. 
물론, 컨설팅이나 대회에 나갈때도 마찬가지 입니다. 

이번 글에서는 Pandas를 가지고 데이터를 빠르게 정제하고 분석하는 방법을 정리했으니, 작업하는데 도움이 되었으면 합니다. 
Pandas는 기본적으로 통계 모델 적용, 시각화, 데이터 조작등에 최적화 된 라이브러리 입니다. 

분석 대상 데이터는 Pima Indians diabets dataset이라고 아주 유명한 데이터 입니다. 

### Summarize Data

Pandas의 describe() 함수를 통해 기본적인 속성 정보를 얻을 수 있습니다. 
기본적으로 count, mean, standard deviation, min, max, percentile 등을 한눈에 볼수 있는 좋은 함수입니다. 



```python
# Load Data
import pandas as pd
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv('pima-indians-diabetes.data', names=names)

# Describe Data
print(data.describe())

```

### Visualize Data

Graph를 이용하면, 보다 쉽게 속성간의 관계를 볼수 있습니다. 
가장 간단한 예는 boxplot 입니다. 

```python

import matplotlib.pyplot as plt
data.boxplot()

```
각 속성간 분포도를 확인하려면 historam이 아주 유용합니다. 

```python
data.hist()

```

Class별로 그룹화해서 관계를 보고싶을경우 groupby()함수를 사용하면 됩니다. 

```python

print(data.groupby('calss').hist())

# 특정 column기준(class 0, 1 둘다 표시됨)
print(data.groupby('class').plas.hist(alpha=0.4)

```

Feature들 간의 관계성을 보기 위해서는 scatter_matrix를 사용해서 상호 작용 분포를 볼수 있습니다. 

```python

from pandas.plotting import scatter_matrix
scatter_matrix(data, alpha=0.2, figsize=(6,6), diagonal='kde')

```

변수들간에 correlation을 보기 위해서 corr() 함수를 사용합니다. 

```python

# Correction Matrix Plot
import matplotlib.pyplot as plt
import pandas
import numpy
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
correlations = data.corr()
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()

```











