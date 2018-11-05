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



4. Initialization Methods


