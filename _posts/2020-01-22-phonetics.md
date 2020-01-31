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
> 어떻게 사람의 성도 조음기관으로 소리가 생성되는지
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
더 나아가 조음기관으로 생성되는 소리에 대한 연구가 articulatory phonetics 이고, 생성된 소리에 대한 음향학적 연구가 acoustic phonetics 입니다.


## 언어음과 음성 표기(Speech sounds and phonetic transcription)

Phonetics의 세부분야에는 단어 발음에 대한 연구가 있습니다.
연구자들은 phones을 표현는 symbol로 단어의 발음을 모델링하게 됩니다.
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


## 조음음성학(Articulatory Phonetics)

어떻게 phones이 생성되는지 이해하는 것은 매우 중요합니다.
조음음성학은 폐에서 나온 공기가 입, 코, 목을 통해 어떻게 소리를 생성하는지에 관한 연구로 여기서부터 우리는 어떻게 phones이 생성되는지 살펴볼 수 있습니다.

### 발성 기관(The Vocal Organs)

소리는 공기의 빠른 움직임입니다.
사람이 말하는 언어는 폐에서 나온 공기가 호흡기관(trachea)을 통해 코나 입으로 나오며 생성됩니다.
호흡기관을 거치며 공기는 후두(larynx)를 통과하며 성대(vocal folds, vocal cords)를 거치게 됩니다.
성대의 두 주름 사이의 공간을 성문(glottis)라고 하는데 성문이 작은 상태이면(성대 주름이 가까운 상태이면) 공기는 이곳을 통과하며 진동을 일으킵니다.
이 성대의 떨림과 함께 만들어지는 소리를 유성음(voiced)이라고 하며 성대의 떨림이 없는 소리를 무성음(unvoiced, voiceless)라고 합니다.
호흡기관 위의 영역은 구성도(oral tract)와 비성도(nasal tract)로 이루어진 성도(vocal tract)입니다.
대부분의 소리가 입을 통해 만들어지고 몇몇 소리는 코를 통합니다.
이 때 코를 통해 만들어지는 소리를 비음(nasal sound)라고 부릅니다.

Phones은 크게 자음(consonant)과 모음(vowel)으로 나누어집니다.
자음은 어디선가 공기의 흐름이 방해받아 만들어지고 유성음, 무성음이 혼재합니다.
모음은 자음에 비해서 공기의 흐름 방해가 덜하고 대부분 유성음이며 소리가 크고 길게 발음되는 특징이 있습니다.

### 자음 : 조음위치(Place of Articulation)

자음은 공기흐름을 제약받으면서 생기기 때문에 어느 위치에서 제약을 받냐에 따라 구분지을 수 있습니다.
제약을 최대로 받는 포인트를 조음위치라고 부릅니다.
조음위치는 동일한 범주로 phones을 묶어줄 수 있기 때문에 음성인식에서 유용하게 사용됩니다.

- 순음, 입술소리(labial) : 두 입술이 주된 제약.
- 치음(dental) : 이에 혀를 놓고 만들어지는 소리.
- 치경음(alveolar) : 윗니 바로 뒤, 입의 천장 부분.
- 구개음(palatal) : 치경음을 만드는 치조융선(alveolar ridge) 뒤로 급격히 일어나는 입의 천장.
- 연구개음(velar) : 입천장 가장 뒤의 근육 덮개(muscular flap), 연구개라고도 부른다.
- 성문(glottal) : 성대.

