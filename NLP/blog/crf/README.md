## [Complete tutorial on Text Classification using Conditional Random Fields Model(in Python)](https://www.analyticsvidhya.com/blog/2018/08/nlp-guide-conditional-random-fields-text-classification/)

뉴스 사이트 및 기타 온라인 미디어만으로도 매시간마다 수많은 텍스트 콘텐츠가 생성됩니다. 
해당 콘텐츠의 패턴을 분석하기 위해 CRF라고 불리는 엔티티 인식을 사용에 대해 다룰 것입니다

* Entity Recognition이란 무엇인가?

  일반적으로 엔티티는 데이터 사이언티스트 또는 비즈니스들이 관심을 갖는 텍스트의 부분으로 정의할 수 있습니다. 
  예) 사람이름, 주소, 계좌번호, 위치 등
  이 외에도 문제해결을 위해 개인이 판단하여 자체 엔티티를 제시할 수 있습니다.
  
  * Case Study Objective & Understanding Different Approaches
  
    * 대상: 보험회사
    * 현상: 고객으로 부터 수천통의 이메일로 클레임을 받음
    * 패턴 식별 방법:
      
      1. Regular expression: 정규표현식을 통해 이메일, 전화번호 등을 인식 -> 학습이 아닌 무차별적인 접근방법
      2. HMM(Hidden Markov Model): 패턴을 학습하고 식별하는 시퀀스 모델링 알고리즘(패턴 학습을 위해 엔티티 주변 예측을 고려해야 하지만, Feature가 서로 독립적으이라고 가정합니다) 엔티티 인식에 가장 적합한 모델은 아닙니다.
      3. MaxEnt Markov Model(MEMM): 시퀀스 모델링 알고리즘(엔티티 주변 예측을 고려하지 않고 기능이 독립적이라고 가정하지 않습니다). 엔티티관계를 식별하는 가장 좋은 방법이 아닙니다. 
      4. Conditional Random Fileds(CRF): 시퀀스 모델링 알고리즘(Feature가 서로 의존한다고 가정할 뿐아니라 주변 예측또한 고려합니다: HMM과 MEMM의 장점 결합) 엔티티 인식을 하는 가장 좋은 방법으로 알려져 있습니다.

* CRF에 사용할 Training Data curating(수집/ 선별 및 새로운 가치 부여)

  1. Annotating training Data: XML 형태로 Annotating 작업 
  
    Email received:
      “Hi,
      I am writing this email to claim my insurance amount. My id is abc123 and I claimed it on 1st January 2018. I did not receive any acknowledgement. Please help.
      Thanks,
      randomperson”

    Annotated Email:
      “<document>Hi, I am writing this email to claim my insurance amount. My id is <claim_number>abc123</claim_number> and I claimed on 1st January 2018. I did not receive any acknowledgement. Please help. Thanks, <claimant>randomperson</claimant></document>”
      
  2. Annotation using [GATE(General Architecture for Text Engineering)](https://gate.ac.uk/download/#latest)


* Building and Training a CRF Module in python

  pycrf module을 설치한다 *pip install python-crfsuite* or *conda install -c conda-forge python-crfsuite*
  



참고자료: [Training Conditional Random Field](https://www.lewuathe.com/machine%20learning/crf/conditional-random-field.html)
