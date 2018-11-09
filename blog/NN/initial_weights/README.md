## [Why Initialize a Neural Network with Random Weights?](https://machinelearningmastery.com/why-initialize-a-neural-network-with-random-weights/)

Neural Network에서 가중치는 small random number로 되어야 한다. Stochastic 최적화 알고르즘의 기대치 이기 때문이다. 

1. Deterministic and Non-Deterministic Algorithms
  
    기존의 알고리즘은  Deterministic이다.(sort 정렬 등) Deterministic은 아주 우수하지만, 모든 문제를 해결할 수는 없다
    
    Non-Deterministic  Alogrithm은 알고리즘 실행중에 Randomness를 이용하여 결정합니다. 그러나 실행 속도나, 결과에 대한 만족도를 보장할수는 없습니다.  
    
    
2. Stochastic Search Algorithm

    Search Problem은 어렵기 때문에 non-deterministic 을 사용합니다. 
    그러나 바운딩 된 random을 사용한 stochastic algorithm을 사용하여 아주 임의적으로 수행되지 않습니다.
    (genetic algorithm, simulated annealing, and stochastic gradient descent 등)
    
    Randomness 는 초기화, 검색 진행에서 사용 합니다. 
    
    Search 공간의 구조를 모르는 상태에서 시작하기 때문에 bias를 없애기 위해 random으로 지정된 위치에서 시작하게 됩니다. 
    
    [Clever Algorithms: Nature-Inspired Programming Recipes](http://cleveralgorithms.com/nature-inspired/index.html)
     
    
3. Random initialization in Neural Networks
    
    최적의 가중치를 찾기위해 randomness 이용한 알고리즘은 input data 와 output 데이터에 종속적으로
    특정 데이터에서는 learning이 진행할 때마다 다른 형태의 모델을 가진 네트웤을 가지게 됩니다.
    
    Stochastic optimization algorithm은 시작점 선택에서 randomness를 사용합니다. 
    그리고 각 epoch별로 shuffling된 training dataset을 사용하기 때문에 각 배치별로 다른 gradient값이 추정됩니다
    
    여러번 수행함으로써 최적의 configuration값을 찾을 수 있게 된다
    
    일반적으로 training 할 때 동일한 Random number를 사용하지 않는다. 왜냐하면 network의 weight값의 변경이 어려울 수 있기 때문이다
    
    동일한 random number가 필요한 경우는 운영환경에서 모델을 사용하여 동일한 weight값을 찾을 때 유용하다.


4. Initialization Methods

    Keras에서 사용가능한 weights를 random number로 초기화하는 리스트는 다음과 같다
    
        Zeros: 생성되는 tensor들을 0으로 초기화
        Ones: 생성되는 tensor들을 1로 초기화
        Constant: 생성되는 tensor들을 상수로 초기화
        RandomNormal: 생성되는 tensor들을 정규분포로 초기화
        RandomUniform : 생성되는 tensor들을 균등분포로 초기화
        TruncatedNormal : 생성되는 tensor들을 Truncated normal distribution으로 초기화
        VarianceScaling : weight에 scaling을 적용 (keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
        Orthogonal : random orthogonal matrix를 생성 (keras.initializers.Orthogonal(gain=1.0, seed=None))
        Identity: 2D matrices 에서만 사용가능하며 identity matrix를 생성
        lecun_uniform : LeCun Uniform initializer
        glorot_normal : Xavier normal initializer
        glorot_uniform : Xavier uniform initializer
        he_normal : He normal initializer
        lecun_normal : LeCun normal initializer
        he_uniform: He uniform variance scaling initializer
        
  일반적으로 사용하는 초기화방법은 다음과 같습니다
  
  * Dense(e.g. MLP): glorot_uniform
  * LSTM : glorot_uniform
  * CNN : glorot_uniform

