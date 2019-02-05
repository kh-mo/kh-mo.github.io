---
layout: post
title: neurlIPS conversational challenge(NIPS ConvAI)
category: Non-Category
---

본 포스트는 2018 NeurlIPS에서 열린 [the conversational intelligence challenge 2(ConvAI2)](http://convai.io/)에 관한 포스트입니다.
다룰 내용은 대회에 대한 간략한 소개와 데이터셋을 소개하는 논문, 베이스라인 모델 Key-Value Memory Network 논문입니다.
잘못된 해석이나 이해가 포함될 수 있으니 첨언과 조언은 언제나 환영합니다.
포스트에 사용된 그림은 논문의 그림을 참고했습니다.

ConvAI는 2017년에 처음 열린 대회로 2018년은 두번째 대회입니다.
대회 주제는 대화형 시스템 개발로 흔히 말하는 챗봇을 만드는 것이 목적입니다.
주최측은 비목적지향 대화형 시스템(non-goal-oriented dialogue system)을 개발하기 위한 데이터셋과 평가기준을 마련했고 2018 NeuralIPS workshop에서 최종 수상자가 발표되었습니다.
수상작에 대한 자료는 [여기](https://github.com/atselousov/transformer_chatbot/blob/agent/docs/slides.pdf)서 확인할 수 있습니다.

## Dataset 

ConvAI 대회에서 사용된 데이터셋에 대한 설명은 ACL 2018에 억셉된 논문 "[Personalizing Dialogue Agents: I have a dog, do you have pets too?](https://arxiv.org/abs/1801.07243)"에 자세히 설명되어 있습니다.
대화 시스템은 크게 목적 지향과 비목적 지향 방식으로 구분할 수 있습니다.
이 중 칫챗(chit-chat) 모델은 그 이름처럼 잡담을 주로하는 비목적 지향 대화 모델입니다.
비행기 티켓 예약, 재무 상담 등 특정한 목적을 가지고 대화하는 목적 지향 시스템들에 비해 가볍게 나누는 대화 시스템이라고도 정의할 수 있습니다.

최근 딥러닝이 여러 NLP 분야의 성능을 끌어올리면서 대화 시스템에서도 이를 적용하고자 하는 연구가 있었습니다.
그러나 아직 일반적인 대화형 시스템은 잠시만 대화를 해보면 몇가지 약점을 드러냅니다.
일관성이 부족하거나 최근 발화를 기준으로 대화를 생성하거나 "난 잘 몰라"와 같은 구체적이지 않은 대화를 생성하는 것이 그 예시입니다.

대부분의 대화 모델은 주어진 데이터셋에서 나온 대화를 **흉내**내도록 만들어집니다.
즉, 위와같은 대화 시스템의 문제는 잘 정제된 데이터셋이 있을 경우 어느정도 해결할 수 있는 문제로 볼 수 있습니다.
본 논문은 일관성 있는 대화를 유지하기 위해 말하는 화자에게 성격을 부여한 새로운 데이터셋을 제시합니다.
대화 주체들은 자신들의 성격 profile을 가지고 서로 대화하게 됩니다.
이를 논문에서는 PERSONA-CHAT dataset이라 부릅니다.
아래 그림은 PERSONA-CHAT dataset의 예시입니다.

![](/public/img/personalizing_dialogue_agents_figure1.JPG "Figure1 of personalizing_dialogue_agents_figure")

Persona 1과 Persona 2는 자신들의 성격을 5가지 문장으로 정의하고 있습니다.
데이터셋에는 총 1155개 persona가 있으며 각 persona는 5개 문장으로 그 성격이 정의됩니다. 
또한 데이터셋을 만드는 과정에서 사람들이 무의식적으로 성격을 표현하는 어휘를 이용해 유사한 대화를 하는 것을 방지하기 위해 profile 정보를 유사한 문장으로 수정합니다(revised persona).
그 성격에 기반한 대화가 PERSONA-CHAT dataset에서 제공하는 데이터입니다. 
Original persona와 revised persona 사이의 예시는 아래와 같습니다.

![](/public/img/personalizing_dialogue_agents_figure2.JPG "Figure2 of personalizing_dialogue_agents_figure")

이렇게 만들어진 데이터셋을 바탕으로 저자들은 총 6가지 베이스라인 모델을 제시합니다.

1. Information Retrieval Model(IR baseline) 
2. Supervised Embedding Model(Starspace)
3. Ranking Profile Memory Network(Profile Memory)
4. Key-Value Profile Memory Network(KV Profile Memory)
5. Seq2Seq
6. Generative Profile Memory Network

1,2,3,4번 모델은 ranking model이고 5,6은 generative model입니다.
대화 시스템은 어떤 문장이 주어졌을 때, 그 문장에 대한 답변을 해야합니다.
그렇기 때문에 대화 시스템은 생성하거나 예측한 답변을 바탕으로 모델을 평가합니다.
Ranking model은 이미 대답할 수 있는 답변 후보군이 존재하고 그 중 적절한 답변을 선택하는 모델들입니다.
Sentence Classification 문제로도 볼 수 있습니다.
이 모델군은 이미 대답가능한 답변 형태가 정해져있기 때문에 새로운 형태의 답변을 줄 순 없습니다.
Generative model은 단어를 하나하나 선택하여 답변을 만들어내는 모델입니다.
자유로운 형태로 대답을 줄 순 있으나 그만큼 정확하게 문장을 생성하긴 어려운 모델입니다.

모델들이 답변을 생성해내면 과연 그것이 얼마나 잘 만들어진 문장인지 평가해야합니다.
성능지표를 어떻게 측정할지 고민이 많이 필요한 이유는 아직 명확한 지표가 없기 때문이기도 하지만 올바른 평가 방법이 곧 더 나은 대화 시스템을 만들 수 있는 지침이 되기 때문입니다. 
이 논문에서 제시하는 정량적 평가 지표는 다음과 같습니다.

1. perplexity
2. hits@1

Perplexity는 생성 모델이 다음 단어를 생성할 때 혼란스러워 하는 정도를 의미합니다.
총 단어 집합이 n이라 가정하고 perplexity가 n이 나왔다면 이는 uniform distribution을 이루고있는 것입니다.
평균적으로 n개의 후보 중에서 선택할 수 있다는 얘기이기 때문에 이 값은 낮을수록 좋습니다.
더 자세한 내용이 궁금하신 분은 [이 페이지](https://kh-kim.gitbook.io/natural-language-processing-with-pytorch/00-cover-8/03-perpexity)를 참고하시기 바랍니다.
Hits@1은 후보 정답 집합군 중 정답으로 맞는 갯수를 1 나머지를 0으로 놓고 계산하는 방식으로 accuracy와 유사한 개념으로 볼 수 있습니다.

아래 그림은 정량적 평가지표를 바탕으로 베이스라인 모델들의 성능을 평가한 표입니다.
![](/public/img/personalizing_dialogue_agents_figure3.JPG "Figure3 of personalizing_dialogue_agents_figure")

표를 살펴보면 성격을 나타내는 persona 정보를 더 사용했을 시 generative model과 ranking model의 성능이 모두 향상됨을 확인할 수 있습니다.
Revised persona는 original persona보다 성격을 나타내는 표현 사이에 단어 오버랩이 줄어든 등 훨씬 어려운 작업입니다.

정량적인 평가 이외에도 사람이 직접 평가하는 정성적 평가도 진행되었습니다.
아래 그림은 정성적인 평가가 수행된 표입니다.
![](/public/img/personalizing_dialogue_agents_figure4.JPG "Figure4 of personalizing_dialogue_agents_figure")

이 표는 사람들이 학습된 모델과 대화를 나누면서 fluency, engagingness, consistency에 대해 1~5점 사이 점수를 매긴 표입니다.
가장 상단은 사람과 사람이 대화를 나눠 받은 점수로 다른 모델들이 목표로 삼을 human level 기준점으로 볼 수 있습니다.
많은 연구가 진행되어 이 지점에 도달하기를 기대합니다.

## Key-Value Memory Network
