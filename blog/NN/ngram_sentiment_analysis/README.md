## [How to Develop an N-gram Multichannel Convolutional Neural Network for Sentiment Analysis](https://machinelearningmastery.com/develop-n-gram-multichannel-convolutional-neural-network-sentiment-analysis/)

기본적인 텍스트 분류와 감정분석은 word embedding layer와 1D CNN을 사용합니다.

서로 다른 kenel size를 통하여 multiple parallel CNN으로 확장이 가능합니다.(다른 N-gram size 이용)

1. Data Preparation

   [Movie Review Polarity Dataset](https://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz)
   
    1. Separation of data into training and test sets
    2. Loading and cleaning the data to remove punctuation and numbers
    3. Prepare all reviews and save to file
   
2. Develop Multichannel Model

    1. Encode Data
    2. Define Model
    3. Complete Example
    
