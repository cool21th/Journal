## [Making Predictions with Sequences](https://machinelearningmastery.com/sequence-prediction/)

Sequence model은 훈련할 때, 예측할 때 관찰된 순서대로 진행됩니다.

1. Sequence

   일반적인 set은 순서가 중요하지 않지만, sequence에서는 순서가 중요합니다.
   
2. Sequence Prediction
   
   Sequence prediction은 주어진 값을 통해 다음 값을 예측하는 것입니다
   

   * Weather Forecasting : 시간 경과에 따른 날씨에 대한 관측을 통해 내일 예상되는 날씨 예측
   
   * Stock Market Prediction : 시간 경과에 따른 주식 변동에 따른 다음 주식의 움직임 예측
   
   * Priduct Recommendation : 고객의 이전 구매이력을 통해 다음 고객의 구매 예측
   
3. Sequence Classification

   Sequence Classification은 주어진 입력 시퀀스에 대한 class label을 예측하는 작업입니다.
   
   * DNA Sequence Classification : ACGT값의 DNA sequence가 주어지면, coding 또는 non-coding region을 예측
   
   * Anomaly Detection : 일련의 값들이 주어지면, 그 값이 정상인지 비정상인지 예측
   
   * Sentiment Analysis : 리뷰나 트윗 같은 텍스트들이 주어지면, 감정이 긍정적인지 부정적인지 예측
   

4. Sequence Generation

   Sequence generation에서는 corpus의 다른 sequence와 동일한 일반적인 특성을 갖는 새로은 출력 sequence 생성합니다.
   
   * Text Generation : 셰익스피어의 작품과 같은 텍스트의 corpus가 주어지면 셰익스피어와 같은 새로운 문장을 생성
   
   * Handwriting Prediction : Handwriting 예제의 corpus가 주어지면, 동일한 속성을 가진 새로운 문구에 대한 handwriting 생성
   
   * Music Generation : 음악에 대한 corpus가 주어지면, 동일한 속성을 가진 음악 작품을 생성
   
   * Image Caption Generation : 이미지가 주어지면, 설명할 수 있는 단어 생성
   
   
5. Sequence-to-Sequence Prediction

   입력 seqeunce가 주어지면, 출력 sequence를 예측하는 것입니다
   
   * Multi-Step Time Series Forecasting : 주어진 시계열 값으로 미래의 시간 간격 범위에 대한 관측 예측
   
   * Text Summarization : 텍스트 문서가 주어지면, 문서의 중요한 부분을 설명 하는 짧은 텍스트 시퀀스를 예측
   
   * Program Execution : 텍스트 설명 프로그램 또는 수학 방정식이 주어지면 올바른 출력을 나타내는 문자 sequence를 예측
