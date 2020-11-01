---
title: Pretrained Model Visualization
category: tmp
---

우리가 어떤 인공지능 모델을 만들었고 이를 활용하면서 마주치는 문제에는 **'왜 이 모델이 이렇게 판단했는가?'**를 이야기해야 한다는 점입니다.
딥러닝이 발전하며 높은 성능을 보인다고 하지만 왜 그렇게 성능을 낼 수 있는지 모델이 설명하진 못합니다.
물론 안되면 되도록 하는 여러가지 방법들이 제시되겠죠.
본 포스트에서는 제안된 여러가지 방법론들에 대해 이야기해보고자 합니다.

## 데이터 공간으로 투영시키기(Projection to data space)

딥러닝 모델은 이미지, 텍스트, 동영상, 음성과 같은 비정형데이터에 적용되어 좋은 성능을 보여왔습니다.
그리고 이 모델이 풀 수 있는 여러가지 문제가 있겠지만 단순하게 분류문제를 풀었다고 해보겠습니다.
입력 데이터가 고양이 사진일 경우 '왜 이것을 고양이 사진으로 분류했는가?', 입력 데이터가 영화 리뷰일 경우 '왜 이 리뷰를 긍정 리뷰로 분류했는가?'와 같은 질문이 우리가 요구하는 설명력입니다.

![](/public/img/pretrained_model_visualization_figure1.JPG "Guided Backpropagation result")

해당 사진은 [Guided Backpropagation 알고리즘](https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec)을 적용한 결과입니다.
Imagenet 데이터로 학습시킨 분류기에 고양이 사진을 넣었을 경우 전파된 그래디언트를 시각화한 것입니다.
이 부분으
