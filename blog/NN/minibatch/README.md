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
              일부 문제들에 대해 빠르게 학습하여 결과를 낼 수 있다
              일부분의 최소점을 피할 수 있다
        
        * 단점
        
              모델을 자주 업데이트 하거나 대규모 데이터셋에서 학습하면 시간이 오래 걸린다
              빈번한 업데이트는 노이즈를 가져오고 그 결과 training epochs을 넘어선 분산값을 가짐으로 모델 에러를 유발할 수 있다
              Noise signal learning 인해 모델의 최소 오류값을 가져오는데 어려움이 생긴다
                
        
    * Batch Gradient Descent
    
        SGD는 Training 예제 마다 모델을 변경하지만, 
        Batch Gradient Descent는 모든 학습이 평가되고 나서 모델을 없데이트 합니다.
        
        * 장점
            
              모델의 update 횟수가 적어지면서 SGD보다 gradient descent 변화를 계산하는데 효율적입니다. 
              Update빈도가 감소하면서 error gradient가 안정적으로 유지되며 결과적으로 수렴하는 결과를 가져옵니다
              모델 에러 예측 계산과 모델의 업데이트를 분리함으로써 병렬처리할 수 있는 기반을 마련해 줍니다. 
              
        * 단점
            
              안정적인 error gradient로 인해 최적이 아닌 parameter값에 수렴할 수 있다
              모든 training set에서 error를 예측을 누적하는 complexity를 추가해야 합니다
              일반적으로 메모리에 전체 training set이 필요하며 동시에 algorithm을 사용할 수 있어야 합니다.
              대규모 dataset의 경우 모델의 업데이트와 training 속도가 느려질 수 있습니다
              
    
    * Mini-Batch Gradient Descent
    
        Mini-Batch Gradient Descent 는 gradient descent의 변형된 형태로 training set를 분할해서 
        모델 error를 계산하고 coefficients를 업데이트 하는 방법입니다.
        
        Mini-batch를 통해 gradient를 합산하거나, 분산값을 감소시키는 값의 평균으로 선택할 수 있습니다.
        
        * 장점
        
              Gradient descent보다 모델의 업데이트 주기가 빨라서 일부 최소점을 피할 수 있습니다. 
              SGD보다 효율적인 프로세스로 계산을 합니다. 
              메모리에 모든 데이터를 올리지 않아도 되고, 알고리즘 구현하는 수행을 하지 않아도 됩니다. 

        * 단점
            
              Mini-batch 사이즈를 추가한 hyperparameter 구성이 요구됩니다
              Training Example의 오류에 대한 정보를 mini-batch 전체에 누적시켜야 합니다(Batch Gradient와 동일)
              
3. How to Cinfigure Mini-batch Gradient Descent

    Mini-batch size는 computational architecture측면에 맞춰 조정이 되며 
    GPU, CPU 의 하드웨어 메모리 요구사항에 맞춥니다. (32, 52, 128, 256 , 2^n 등)
    
    
