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
          
