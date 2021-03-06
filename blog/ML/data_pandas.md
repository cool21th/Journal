[Prepare Data for Machine Learning in Python with Pandas](https://machinelearningmastery.com/prepare-data-for-machine-learning-in-python-with-pandas/)

### Overview

Python 기반으로 머신러닝을 개발하기 위해서, 데이터를 분석하거나 조작할때 가장 적합한 Library가 Pandas 입니다. 
Python SciPy Stack은 Scientific computing에서 가장 널리 쓰입니다. 
배열 데이터활용(Numpy 등)이라던지, 그래프를 그린다던지(matplotlib 등), Missing data를 핸들링하는 등의 다양하게 활용가능 합니다. 

실 데이터는 머신러닝에 바로 쓸 수 있는 형태로 들어오지 않습니다. 
데이터를 활용한 프로젝트를 할 때 중요한 부분이 데이터 분석과 데이터 전처리(Data Munging / Wrangling 등) 입니다. 

- Data analysis: 데이터를 이해를 바탕으로 문제 상황을 파악하기 위해 통계학적 툴이나 시각화를 사용하는 방법
- Data Munging: raw 데이터를 Data analysis or Machine Learning 에 적합하게 변환시키는 방법


### Pandas

앞서 이야기한 것 처럼 Pandas는 데이터 분석과 조작에 최적화된 라이브러리 입니다. 
SciPy framework에서 데이터 핸들링에서 빠진 부분을 보충해줍니다. 

Pandas는 데이터를 Python을 활용하여 메모리에 로드한 후 작업이 진행됩니다. 
관계형 Database 나 spreadsheet 와 같은 테이블 형태의 데이터 작업을 할 수 있습니다. 

##### Scipy

SciPy는 수학과 과학에 최적화된 Python Library 입니다. SciPy를 구성하는 ecosystem은 다음과 같습니다. 

- NumPy: A foundation for SciPy that allows to efficiently work with data in arrays
- Matplotlib: Allows you to create 2D charts and plots from data
- pandas: tools and data structures to organize and analyze your data


### Pandas Features

Pandas는 SciPy Stack의 표준 라이브러리 위에 구축됩니다. Numpy를 활용해 빠른 배열처리를 하고, 
StatsModels을 활용한 통계작업들과 matplotlib를 통해 차트작성이 가능하도록 Wrapper를 제공합니다. 

Financial domain에 적합한 시계열데이터, 일반적인 그리드 데이터등 빠르게 처리할 수 있도록 지원해줍니다. 
이와 같은 데이터 처리에 관련된 feature들은 다음과 같습니다.

- Manipulation: Moving columns, slicing, reshaping, merging, joining, filtering 등등
- Time-series Handling: date/times 처리, resampling, moving windows, auto-alignment of datasets 등
- Missing Data Handling: auto-exclude, drop, replace, interpolate miising values
- Hierarchical Indexing: data structure level, column기반 효율적인 구성 등
- Summary Statistics: Fast and powerful summary statistics of data
- Visualization: histogram, box plots,scatter matrix 등 데이터 구조 기반 시각화 


### [Data Loading (python, numpy, pandas)](https://machinelearningmastery.com/load-machine-learning-data-python/)

기본적으로 Python에서 loading 할 data format은 csv 파일을 주로 씁니다. 
csv에 대해 고려할 것은 다음과 같습니다. 

- CSV File Header : 보통 1번째 row를 column 명으로 인식한다. (원하지 않은 경우 header =None)
- Comments: hash('#')로 시작하는 문장을 주석여부에 대해 정의해야 합니다.
- Delimiter: field 단위를 나누는 기준입니다. 보통 ','를 쓰지만 다른 delimiter인 경우 따로 고려 해야 합니다. 
- Quotes: 기본 인용부호는 큰따옴표인 "\"" 입니다. 다른 문자를 사용할 경우 지정해줘야 합니다.


##### Load csv with Python Standard Library

```python
# Load CSV (using python)
import csv
import numpy
filename = 'pima-indians-diabetes.data.csv'
raw_data = open(filename, 'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
data = numpy.array(x).astype('float')
print(data.shape)
```

##### Load CSV file with NumPy

```python

# 1. Load CSV
import numpy
filename = 'pima-indians-diabetes.data.csv'
raw_data = open(filename, 'rt')
data = numpy.loadtxt(raw_data, delimiter=",")
print(data.shape)


# 2. Load CSV from URL using NumPy
from numpy import loadtxt
from urllib.request import urlopen
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
raw_data = urlopen(url)
dataset = loadtxt(raw_data, delimiter=",")
print(dataset.shape)

```


##### Load CSV file with Pandas

```python

# Load CSV using Pandas
import pandas
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(filename, names=names)
print(data.shape)

# Load CSV using Pandas from URL
import pandas
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
print(data.shape)

```



참고자료:\
[pandas homepage](https://pandas.pydata.org/)



