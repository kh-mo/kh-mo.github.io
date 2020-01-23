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


## 언어음과 음성 표기(Speech sounds and phonetic transcription)

Phonetics의 세부분야에는 단어의 발음에 대한 연구가 있습니다.
연구자들은 phones으로 표현되는 symbol로 단어의 발음을 모델링하게 됩니다.
즉, phones이 중요한 요소가 되는 것인데 영어에서 phones을 나타내는 큰 2가지 체계가 있습니다.

첫번째 체계는 IPA(International Phonetic Alphabet) 입니다.
1,888년 국제음성협회(International Phonetic Association)에서 처음 제정되어 발전하고 있는 체계로 모든 인간의 언어 소리를 기록하는 것을 목적으로 합니다.
IPA는 일종의 알파벳이자 기록하는 방법의 집합체 또는 원리라고 볼 수도 있습니다.
이 원리를 따르면 동일한 발화도 주어진 상황에 따라 다른 방식으로 기록될 수 있습니다.

두번째 체계는 ARPAbet 입니다.
ASCII symbols을 사용하는 본 체계는 american english에 특화 설계되었습니다.
IPA의 american english subset의 ASCII 표현 버전으로 여기기도 해서 ASCII 폰트 사용이 용이한 어플리케이션에서 많이 사용된다고 합니다.
IPA보다 더 보편적으로 사용되기도 합니다.

IPA나 ARPAbet 기호의 대부분은 로마 문자와 유사하기도 하지만 phones와 알파벳 사이에 매핑에 어려운 부분이 존재하기도 합니다.
때문에 해당 부분의 차이를 인지하고 공부, 연구하는 것이 중요합니다.

