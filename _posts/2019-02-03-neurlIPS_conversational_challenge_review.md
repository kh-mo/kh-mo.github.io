---
layout: post
title: neurlIPS conversational challenge(Nips ConvAI)
category: Non-Category
---

본 포스트는 2018 NeurlIPS에서 열린 [the conversational intelligence challenge 2(ConvAI2)](http://convai.io/)에 관한 포스트입니다.
ConvAI는 대화형 시스템을 개발하는 것을 목적으로 하는 대회로 흔히 말하는 챗봇을 만드는 대회입니다.
포스트에서 다룰 내용은 대회에 대한 간략한 소개와 데이터셋을 소개하는 논문, 베이스라인 모델 중 Key-Value Memory Network 논문입니다.
잘못된 해석이나 이해가 포함될 수 있으니 첨언과 조언은 언제나 환영합니다.
포스트에 사용된 그림은 논문의 그림을 참고했습니다.

ConvAI는 2017년에 처음 열린 대회로 2018년은 두번째 대회입니다.
주최측은 비목적지향 대화형 시스템(non-goal-oriented dialogue system)을 개발하기 위한 데이터셋과 평가기준을 마련했고 2018 NeuralIPS workshop에서 최종 수상자가 발표되었습니다.
수상작에 대한 자료는 [여기](https://github.com/atselousov/transformer_chatbot/blob/agent/docs/slides.pdf)서 확인할 수 있습니다.

## Dataset 

ConvAI 대회에서 사용된 데이터셋은 ACL 2018에 억셉된 논문 "[Personalizing Dialogue Agents: I have a dog, do you have pets too?](https://arxiv.org/abs/1801.07243)"에서 자세히 설명하고 있습니다.
칫챗(chit-chat) 모델은 그 이름처럼 잡담을 주로하는 대화 모델입니다.
비행기 티켓 예약, 재무 상담 등 특정한 목적을 가지고 대화하는 시스템들에 비해 가볍게 나누는 대화 시스템이라고도 정의할 수 있습니다.

대화 시스템은 크게 목적 지향과 비목적 지향 방식으로 구분하여 만들어집니다.
딥러닝 기반의 대화형 시스템은 잠시만 대화를 해보면 몇가지 약점을 드러냅니다.
일관성이 부족하거나 최근 발화를 기준으로 대화를 생성하거나 "난 잘 몰라"와 같은 구체적이지 않은 대화를 생성하는 것이 그 예시입니다.

그래서 대화를 생성하는 기준을 주고자 만들어진 데이터셋이 본 논문에서 소개하려는 데이터셋입니다.
대화를 하는 주체들은 이미 자신들의 대화 성격, profile을 가지게됩니다.
그리고 그 성격을 기반으로 대화를 하는 데이터가 생성되었습니다.
이를 논문에서는 PERSONA-CHAT dataset이라 부릅니다.
아래 그림은 PERSONA-CHAT dataset 예시입니다.

![](/public/img/personalizing_dialogue_agents_figure1.JPG "Figure1 of personalizing_dialogue_agents_figure1")

이 데이터셋이 만들어지는 과정은 여러 절차가 있지만 본 포스트에서는 생략하겠습니다.
궁금하신 분들은 해당 논문의 3절을 참고하시기 바랍니다.

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
Sentence Classification 문제로 볼 수 있습니다.
이 모델군은 이미 대답가능한 답변 형태가 정해져있기 때문에 새로운 형태의 답변을 줄 순 없습니다.
Generative model은 단어를 하나하나 선택하여 답변을 만들어내는 모델입니다.
자유로운 형태로 대답을 줄 순 있으나 그만큼 정확하게 문장을 생성하긴 어려운 모델입니다.

