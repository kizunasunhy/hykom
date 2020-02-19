# hykom
[English](/README.md) | Korean

khaiii의 태깅 방법을 활용해서 BiLSTM 기반으로 개발한 띄어쓰기 자동 수정 기능이 있는 형태소분석기입니다.
## 실행방법
### 필요한 패키지
```
PyTorch 0.4 or higher
tensorboardX
tqdm
```
Conda 가상 환경에서 실행하기를 권장합니다. 그리고 PyTorch 0.4 같은 경우에는,  torch0.4 + cuda9.2만 됩니다.
아니면 "RuntimeError: CuDNN error: CUDNN_STATUS_SUCCESS"라는 에러가 뜹니다.
```
conda install pytorch=0.4.1 cuda92 -c pytorch
```
### 데모
우선 pre-trained 모델을 다운 받으세요. https://drive.google.com/open?id=1uJkbM3vT0kURxzIV7x8VHDkn9USkv2Ou
`model.state`와 `optim.state`를 `/logdir/corpus.cut2.sdo0.0.emb100.lr0.001.lrd0.6.bs250`에 넣으세요.

PYTHONPATH를 export하세요.
```
export PYTHONPATH=/path/to/hykom/src
```
이 데모로 형태소 분석 결과를 쉽게 볼 수 있습니다.
python demo.py
문장을 입력하면 결과를 볼 수 있습니다. 또한, 띄어쓰기 잘 못한 경우에도 자동으로 수정하고 결과를 제대로 줍니다.
예를 들어서 "경 복궁 야간 개장 언제 어 디서 해 요? 시 간도 알고 싶어요."를 입력하면은
```
경복궁 야간개장 언제 어디서 해요? 시간도 알고 싶어요.

경복궁  경복궁/NNP
야간개장        야간개장/NNG
언제    언제/MAG
어디서  어디/NP + 서/JKB
해요?   하/VV + 여요/EF + ?/SF
시간도  시간/NNG + 도/JX
알고    알/VV + 고/EC
싶어요. 싶/VX + 어요/EF + ./SF
```
이 기능은 음성인식이나 챗봇 같은 상황에서 중요한 기능입니다.
## Corpus
세종 코퍼스는 아래와 같은 포맷을 가집니다.
```
<text>
<group>
<text>
<body>
<source>
<date>
BTAA0001-00000001       1993/06/08      1993/SN + //SP + 06/SN + //SP + 08/SN
</date>
<page>
BTAA0001-00000002       19      19/SN
</page>
</source>
<head>
BTAA0001-00000003       엠마누엘        엠마누엘/NNP
BTAA0001-00000004       웅가로  웅가로/NNP
BTAA0001-00000005       /       //SP
BTAA0001-00000006       의상서  의상/NNG + 서/JKB
BTAA0001-00000007       실내    실내/NNG
BTAA0001-00000008       장식품으로…     장식품/NNG + 으로/JKB + …/SE
BTAA0001-00000009       디자인  디자인/NNG
BTAA0001-00000010       세계    세계/NNG
BTAA0001-00000011       넓혀    넓히/VV + 어/EC
</head>
<p>
BTAA0001-00000012       프랑스의        프랑스/NNP + 의/JKG
BTAA0001-00000013       세계적인        세계/NNG + 적/XSN + 이/VCP + ㄴ/ETM
BTAA0001-00000014       의상    의상/NNG
BTAA0001-00000015       디자이너        디자이너/NNG
BTAA0001-00000016       엠마누엘        엠마누엘/NNP
BTAA0001-00000017       웅가로가        웅가로/NNP + 가/JKS
BTAA0001-00000018       실내    실내/NNG
BTAA0001-00000019       장식용  장식/NNG + 용/XSN
BTAA0001-00000020       직물    직물/NNG
BTAA0001-00000021       디자이너로      디자이너/NNG + 로/JKB
BTAA0001-00000022       나섰다. 나서/VV + 었/EP + 다/EF + ./SF
</p>
```
아래 명령을 통해 어절의 원문과 형태소 분석 결과를 음절 단위로 정렬합니다.  동시에 원형복원사전이 생깁니다.
```
python map_char_to_tag.py -c corpus --output corpus.txt --restore-dic restore.dic
```
최종의 corpus는 아래 형식과 같습니다:
```
엠마누엘    I-NNP I-NNP I-NNP I-NNP
웅가로    I-NNP I-NNP I-NNP
/    I-SP
의상서    I-NNG I-NNG I-JKB
실내    I-NNG I-NNG
장식품으로…    I-NNG I-NNG I-NNG I-JKB I-JKB I-SE
디자인    I-NNG I-NNG I-NNG
세계    I-NNG I-NNG
넓혀    I-VV I-VV:I-EC:0

프랑스의    I-NNP I-NNP I-NNP I-JKG
세계적인    I-NNG I-NNG I-XSN I-VCP:I-ETM:0
의상    I-NNG I-NNG
디자이너    I-NNG I-NNG I-NNG I-NNG
엠마누엘    I-NNP I-NNP I-NNP I-NNP
웅가로가    I-NNP I-NNP I-NNP I-JKS
실내    I-NNG I-NNG
장식용    I-NNG I-NNG I-XSN
직물    I-NNG I-NNG
디자이너로    I-NNG I-NNG I-NNG I-NNG I-JKB
나섰다.    I-VV I-VV:I-EP:0 I-EF I-SF
```
```
혀/I-VV:I-EC:0    히/I-VV 어/I-EC
혀/I-VV:I-EC:1    히/I-VV 여/I-EC
혀/I-VV:I-EC:2    허/I-VV 어/I-EC
혀/I-VV:I-EC:3    하/I-VV 여/I-EC
혀/I-VV:I-EC:4    혀/I-VV 어/I-EC
혀/I-VV:I-EC:5    치/I-VV 어/I-EC
혀/I-VV:I-EC:6    히/I-VV 아/I-EC
인/I-VCP:I-ETM:0    이/I-VCP ㄴ/I-ETM
인/I-VCP:I-ETM:1    이/I-VCP 은/I-ETM
섰/I-VV:I-EP:0    서/I-VV 었/I-EP
섰/I-VV:I-EP:1    시/I-VV 었/I-EP
섰/I-VV:I-EP:2    스/I-VV 었/I-EP
```
## Training
### 준비
```
python split_corpus.py --input corpus.txt -o corpus
```
```
python make_vocab.py --input corpus.train
```
vocab.in:
```
齒  25
齡  8
龍  300
龕  8
龜  16
가  499305
각  58237
간  77133
갇  478
갈  15383
```
vocab.out:
```
I-XSN
I-XSV
I-ZN
I-ZV
I-ZZ
B-EP:I-EC:0
B-EP:I-EF:0
B-EP:I-ETM:0
B-JKB:I-JKG:0
B-JKB:I-JX:0
```
### 주의!
### 시작
```
export PYTHONPATH=/path/to/hykom/src
```
```
python train.py -i corpus
``` 
## 사전 
```
복지택시	복지택시/NNG
로그인	로그인/NNG
재부팅	재부팅/NNG
가즈아	가/VV + 즈아/EC
```
## RESTful API
