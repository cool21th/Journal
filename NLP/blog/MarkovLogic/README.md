## [Can Markov Logic Take Machine Learning to the Next Level?](https://www.datanami.com/2018/07/03/can-markov-logic-take-machine-learning-to-the-next-level/)

학계나 과학계에서는 Markov Logic의 등장으로 인해 머신러닝을 구현하기 위한 Probabilistic Programming이 좀더 쉬워질 것으로 예상합니다.
Markov Logic은 Pedro Domingos and Matthew Richardson(University of Washington’s Department of Computer Science and Engineering)의 
seminal 2006 paper “Markov Logic Networks.”에서 발표한 논문에서 따왔습니다.

Markov Logic은 Markov Random Field의 개념을 구현합니다(미래의 상태는 현재상태에만 의존하는 확률적 과정)

전형적인 Deep Learning과 Statistical learning에서 모든 객체는 하나의 행이고, 모든 변수는 하나의 열, 즉 하나의 테이블에서 나옵니다
그러나 실제적으로 모든 데이터 과학에서 의미있는 통찰력은 서로다른 데이터셋에서 상관 관게및 변칙이 발견될때 나옵니다. 이러한 과정이 현실적으로 어렵습니다

Markov Logic은 데이터 집합에 존재하는 상위 레벨의 구조를 이용하여 패턴을 감지하고, 추론을 수행하여 기존의 머신러닝보다 좋습니다.
기본적으로 feature를 정의하거나 1차 논리수식으로 데이터의 feature들을 학습하는 것으로 구성되어있습니다.

기존의 머신러닝과 다른점은 feature들에게 weight가 있다는 것입니다. 자연적으로 1차로직이지만 여러 테이블을 정의할 수 있게 됩니다.

확률론적 측면과 통계적 측면에서 대부분의 머신러닝 모델을 가정하기 때문에 매우 강력한 로직입니다.

머신러닝이 자동화됨으로 데이터 과학자와 개발자가 모델 개발과 지식개발에 집중할 수 있게 됩니다. 

