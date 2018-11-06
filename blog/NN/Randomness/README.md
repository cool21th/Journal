## [Embrace Randomness in Machine Learning](https://machinelearningmastery.com/randomness-in-machine-learning/)

Machine Learning에서 randomness의 역활을 정확히 하는 것이 중요하다
이를 통해서 algorithm, hyperparameter 튜닝등을 정할 수 있기 때문이다

#### Machine Learning Algorithms Use Random Numbers

1. Randomness in Data Collection

    다른 데이터로 훈련된 알고리즘은 다른 모델을 구성하게 됩니다. 
    알고리즘에 따라 다르고, 모델이 얼마나 다른지를 보는 기준은 model variance(편차) 입니다.
    따라서 데이터 그자체가 Randomness의 소스가 됩니다. 

2. Randomness in Observation Order

    데이터의 순서를 통해 영향을 받는데, 이를 극복하기 위한 것이 무작위로 데이터를 섞는 방법입니다. 
    

3. Randomness in the Algorithm

    알고리즘내에서 random state로 초기화를 하면서 deterministic method를 찾아낸다.

4. Randomness in Sampling

    너무 많은 데이터에서 random으로 트레이닝할 데이터를 뽑아낸다.

5. Randomness in Resampling

    k-fold와 같이 데이터 교차검증 할때 사용한다
    

#### Random Seeds and Reproducible Results

동일한 데이터에서 동일한 모델을 생성하려면 code, data 그리고 sequence of random numbers가 정확히 같아야 한다

난수를 생성하는 함수는 deterministic하기 때문이다. 


#### Machine Learning Algorithms are Stochastic

1. Tactics to Reproduce the Uncertainty

    K-fold를 통해 검증한다면, k를 증가시키거나, 훈련 반복 횟수를 증가 시키는 방법이 있습니다. 
    
2. Tactics to Report the Uncertainty

    보통 Gaussian 분포를 이루기 때문에 성능의 표준편차와 평균으로 시작하는 것이 좋습니다
    
    * Choosing between algorithms
    * Choosing between configurations for one algorithm
    

최종 모델을 선택할 때는 Accuracy가 높은 모델을 선택하는 것이 답이 아니다. 그것은 overfitting 때문이다.

글에서 추천하는 방법은 상황에 따른 앙상블 모델을 만드는 것을 추천한다. 


