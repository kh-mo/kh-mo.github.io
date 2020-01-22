---
layout: post
title: Phonetics
category: etc
---

본 포스트는 speech and language processing 교재의 7장 phonetics를 정리한 것입니다.
공부한 내용을 정리하는 것이기에 잘못된 점이 있을 수 있습니다.
해당 내용에 대한 교정, 첨언, 조언은 언제나 환영입니다.

## Intro

소리기반의 문자(sound-based writing system)에 내제된 아이디어는 말할 수 있는 모든 단어(spoken word)는 작은 요소(units of speech, phones)로 구성되어 있다는 것입니다.
이를 설명하는 Ur-theory는 현대의 모든 음운론(phonology)들의 근간이 됩니다.
Phonetics란 언어학 소리에 대한 연구로 크게 3가지 분야를 연구합니다.

>
> How they are produced by the articulators of the human vocal track.<br>
> 어떻게 소리가 사람의 vocal track의 조음기관으로 생성되는지
>
> How they are realized acoustically.<br>
> 어떻게 음향적(청각적)으로 인지되는지
>
> How this acoustic realization can be digitized and processed.<br>
> 어떻게 청각적 인지가 디지털화 되고 처리될 수 있는지
>

Phonetics 연구의 핵심 요소는 Phones 입니다.
Phones은 단어가 어떻게 발음되는지는 설명하는 요소이자 개별적인 speech unit 입니다.
STT나 TTS같은 현대 알고리즘을 제작하려면 먼저 **'발음을 인지할 수 있어야 하고 쓸 수 있어야'**하는데 이 때 phones이 중요한 역할을 하게 됩니다.
이 phones을 설명하기 위한 체계가 phonetic alphabets 입니다.
더 나아가 입의 조음기관으로 생성되는 소리에 대한 연구가 articulatory phonetics 이고, 생성된 소리에 대한 음향학적 연구가 acoustic phonetics 입니다.

