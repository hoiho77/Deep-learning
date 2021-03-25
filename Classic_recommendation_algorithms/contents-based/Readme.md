# Contents-based Sytem
- multi-regression
  - item 속성을 분석한 뒤, 각 유저 별로 item 선호도를 속성으로 입력 x -> y는 선호도 예측
  
- tf-idf 
  - tf : 단어 w가 문서 d에 등장한 빈도수 -> w의 수 / 모든단어
  - df : 단어 w가 등장한 문서 d의 수
  - idf : log(전체문서 수/ 단어 w가 포함된 문서의 수)
    * 단어 w가 포함된 문서 수가 많을 수록 df 증가, idf 감소, 정보력 하락
    * df가 큰 단어는 정보력이 적고, idf 가 클수록 정보력 큼.
  - tf-idf : tf * idf