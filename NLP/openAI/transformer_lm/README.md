## [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)

서문

Natural Language Understanding은 텍스트 함의(한문장의 의미가 논리적으로 시사하는것), QA, 의미론적으로 유사성, 
문서 분류등으로 구성됩니다. 언어 데이터가 많아도, 레이블된 데이터가 상대적으로 부족하기 때문에 언어모델을 만드는 것이 어렵습니다.
이번 논문에서 레이블이없는 텍스트 데아터로 만든 언어모델을 약간의 조정을 통해 많은 이득을 볼수 있는 것을 보여줍니다.
특히, 상식추론(commonsense reasoning)8.9%, QA 5.7%, 텍스트 함의 1.5% 부분에서 성과를 보여줍니다.

1. 소개

    자연어 처리는 supervised learning 의 의존도가 높습니다. 특히 주석이 많이 필요한 도메인에서는 더욱더 제한적입니다
    Unlabeled된 자료를 이용해서 모델을 만든다면, 시간과 비용측면에서 효과적으로 절약할 수 있습니다.
    현재까지 자연어 처리 범위에서 알려진 가장 강력한 방법은 사전 학습된 word embedding을 사용하는 것입니다

    Unlabeled 된 텍스트에서 추출한 단어 레벨의 정보를 활용하는데는 두가지 어려움이 있습니다. 
    첫 번째는 변환에 사용할 learning text representation 최적화 목표가 불명확하다는 것입니다.  
    Language Modeling, Machine Translation, Discourse coherence 와 같은 다양한 방법들을 최근에 조사한 결과로 
    각각 다른 목표 테스크에서 뛰어난 성과를 달성했습니다.
    두 번째는 가장 효과적인 방법에 대한 합의가 없다는 것입니다. 기존 모델 [43](https://arxiv.org/pdf/1705.00108.pdf) [44](https://arxiv.org/pdf/1802.05365.pdf)은 작업별 변경, 복잡ㅂ한 학습계획, 보조적인 학습목표 추가 등을 조합합니다 
    띠라서 semi-supervised Learning에 효과적인 개발을 진행하는데 어려움이 따릅니다. 

    




