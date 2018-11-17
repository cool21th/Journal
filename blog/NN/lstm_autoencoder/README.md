## [A Gentle Introduction to LSTM Autoencoders](https://machinelearningmastery.com/lstm-autoencoders/)

Encoder 파트는 data visualization 또는 feature vector 로 표현되는 seqeunce data로 압축되거나 인코딩하는데 사용됩니다


1. What Are Autoencoder

   Autoencoder는 input의 압축된 표현을 배우는 neural network model입니다
   
   Self-supervised learning 방법을 통해 input을 재현하는 시도를 하는 모델입니다.
   
   일반적으로 학습 또는 automatic feature extraction model로 사용합니다.
   
   bottleneck 지점에서 input data는 고정된 길이의 압축된 representation으로 표현되며,
   output은 visualization 위한 feature vector 또는 차원 감소에 사용된다
   
   
2. A Problem with Sequence

   Sequence 예측은 입력 시퀀스의 길이가 다르기 때문에 어렵습니다.
   
   * machine learning algorithm이나 neural network는 고정된 길이의 입력을 사용하도록 설계되어 있기 때문입니다.
   
   * Supervised learning 의 input 으로 사용하기 위해 관측의 순서로 추출된 feature들에 대해 도메인이나 신호처리 분야 등의 전문지식이 필요합니다.
   
   * Sequence model 예측 자체가 sequence 이기 때문입니다 (seq2seq)
   

3. Encoder-Decoder LSTM Models

   LSTM은 입력 sequence의 시간 순서내에서의 복잡한 역학을 학습할 수 있을 뿐 아니라, 
   긴 입력 sequence에서 정보를 기억하거나 사용하기 위해 내부 메모리를 사용할 수 있습니다.
   
   LSTM 네트워크는 가변길이 입력 seqeunce를 지원하고, 
   가변길이 출력 sequence를 예측 또는 출력하는 모델을 사용할 수 있는 Encoder-Decoder LSTM 아키텍처로 구성될 수 있습니다
   
   이러한 아키텍쳐는 Speech recognition, text translation과 같은 시퀀스 예측문제에서 사용했습니다.
   
   
4. What is an LSTM Autoencoder

   Encoder-Decoder LSTM 아키텍처를 사용하여 sequence data용 autoencoder를 구현한 것입니다. 
   
   

   
   
