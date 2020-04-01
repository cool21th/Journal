Boosting 계열의 ensemble 모델에서 2018~19년에 가장 핫했던, XGBoost와 LightGBM 중 XGBoost 에 대한 소개를 하고자 합니다. 

[공식문서](https://xgboost.readthedocs.io/en/latest/)

## Introduction XGBoost

[A Gentle Introduction to XGBoost for Applied Machine Learning](https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/)

XGBoost 모델은 속도와 성능 두가지를 충족시킨, gradient boosted decision trees 알고리즘입니다.
물론 현재(2020년) 속도 측면에서 LightGBM이 훨씬 강력한 모델인 것은 맞습니다. 
XGBoost와 LightGBM은 그 시작이 같으므로 XGBoost를 정확하게 이해하면 그 뒤 Boosting 계열의 알고리즘을 쉽게 받아들일 것입니다. 

### XGBoost 란 무엇인가?

XGBoost 는 e**X**treme **G**radient **Boost**ing 의 약자입니다.

기존의 Gradient Boosting 계열보다 over fitting에 적극적으로 제어하면서 더 나은 성능을 보여줍니다. 
이에 대한 내용은 차츰 설명으로 풀어가도록 하겠습니다.

XGBoost는 여러 Interface를 제공하고 있어서, 다양한 곳에서 사용이 가능합니다. 그 내용은 다음과 같습니다. 

- Command Line Interface (CLI)
- C++ (the language in which the library is written)
- Python interface as well as a model in scikit-learn
- R interface as well as model in the caret package
- Julia
- Java and JVM languages like Scala and platforms like Hadoop

### XGBoost Features

XGBoost 는 컴퓨팅 속도와 모델성능을 고려해 여러가지 Feature들로 그 기능을 제공합니다. 

#### Model Features

기본적인 모델은 Gradient Boosting 계열을 지원하며,
3가지의 Boosting 개념을 지원해줍니다.

> Boosting이란 약한 분류기를 결합하여 강한 분류기를 만드는 과정입니다. 
> 여러개 tree 모델 a, b, c 가 각각의 성능이 0.4라고 가정했을때, 합쳐서 0.8의 성능이 나오는 과정이 Ensemble 모델 결합입니다.
> Ensemble 모델에서 boosting은 a, b, c를 순차적으로 예측 오류를 줄이면서 분류기를 만드는 방식입니다.
> (참고로, Bagging 계열은 독립적으로 모델을 생성해서 결과를 예측)

- Gradient Boosting: Boosting의 가장 일반적인 알고리즘입니다
- Stochastic Gradient Boosting: 행,열 및 하위 레벨에 대한 서브샘플링 기능 지원
- Regularized Gradient Boosting: L1, L2 Norm 지원

#### System Features

컴퓨팅 자원을 활용할 수 있는 기능을 지원합니다. 

- Parallelization: 훈련시 모든 CPU 기반의 Tree 모델 생성 가능
- Distributed Computing: computing cluster를 활용한 규모가 큰 모델 구축 지원
- Out-of-Core Computing: 메모리에만 의존하지 않은 대량의 데이터 셋 처리
- Cache Optimization: Hard ware를 고려하여 최적의 알고리즘 및 데이터구조 사용


#### Algorithm Features


### 
