## A Gentle Introduction to Mini-Batch Gradient Descent and How to Configure Batch Size

Stochastic gradient descent는 deep learning 모델을 트레이닝 하는데 가장 주된 방법입니다. 

1. What is Gradient Decent?

    Gradient descent는 neural network, logistic regression에서  
    weights or coefficients를 찾는데 자주 사용되는 최적화 알고리즘입니다.
    
    모델이 훈련데이터를 통해 예층하고 에러를 줄이기위해 에러를 사용하여 모델을 업데이트 합니다.
    모델의 오류 기울기가 최소가되는 값을 찾는 방법입니다. 
    
        model = initialization(...)
        n_epochs = ...
        train_data = ...
        for i in n_epochs:
          train_data = shuffle(train_data)
          X, y = split(train_data)
          predictions = predict(X, model)
          error = calculate_error(y, predictions)
          model = update_model(model, error)
          

2. Contrasting the 3 Types of Gradient Descent

    * Stochastic Gradient Descent
    
        SGD라고 불리기도 하는 gradient descent algorithm으로 dataset으로 
        트레이닝을 하는 동안 발생하는 에러를 계산하여 모델을 업데이트 하는 방법이다
        
        * 장점
            즉각적인 업데이트로 모델의 성능 개선 속도 등에 대한 insight를 바로 볼수 있다.
            초보자들이 구현하고 이해하기 쉬운 방법이다
        
    * Batch Gradient Descent
    
    * Mini-Batch Gradient Descent
    
    
