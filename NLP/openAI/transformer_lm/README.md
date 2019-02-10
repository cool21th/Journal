## [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)

  서문

  Natural Language Understanding은 텍스트 함의(한문장의 의미가 논리적으로 시사하는것), QA, 의미론적으로 유사성, 
  문서 분류등으로 구성됩니다. 언어 데이터가 많아도, 레이블된 데이터가 상대적으로 부족하기 때문에 언어모델을 만드는 것이 어렵습니다.
  이번 논문에서 레이블이없는 텍스트 데아터로 만든 언어모델을 약간의 조정을 통해 많은 이득을 볼수 있는 것을 보여줍니다.
  특히, 상식추론(commonsense reasoning)8.9%, QA 5.7%, 텍스트 함의 1.5% 부분에서 성과를 보여줍니다.

1. 소개
  자연어 처리는 supervised learning 의 의존도가 높습니다. 특히 주석이 많이 필요한 도메인에서는 더욱더 제한적입니다
  Unlabeled된 자료를 이용해서 모델을 만든다면, 시간과 비용측면에서 효과적으로 절약할 수 있습니다.




