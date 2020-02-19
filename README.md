# hykom
English | [Korean](/README_KOR.md)

A Korean morpheme analyzer.

hykom is a Korean Morpheme analyzer based on BiLSTM architecture. It can parse the morpheme of Korean sentences and correct the spacing at the same time. We also used the tagging method referring from khaiii of Kakao.
## Usage
### Requirement
```
PyTorch 0.4 or higher
tensorboardX
tqdm
```
We highly recommned the conda virtual environment. And for PyTorch 0.4, we've tested that only torch0.4 + cuda9.2 can work. Otherwise you will get a "RuntimeError: CuDNN error: CUDNN_STATUS_SUCCESS" error.
```
conda install pytorch=0.4.1 cuda92 -c pytorch
```
### Inference Demo
First, please download our pretrained model.
https://drive.google.com/open?id=1uJkbM3vT0kURxzIV7x8VHDkn9USkv2Ou
And put the `model.state` and `optim.state` in `/logdir/corpus.cut2.sdo0.0.emb100.lr0.001.lrd0.6.bs250`

And then export the python path.
```
export PYTHONPATH=/path/to/hykom/src
```
We provided a trained model and it's very easy to see the result from a simple demo.
```
python demo.py
```
When you input a sentence, you can see the morpheme analyzed result. And even if you make mistakes on spacing,
you can still get the correct result. For example, if you input "경 복궁 야간 개장 언제 어 디서 해 요? 시 간도 알고 싶어요.",
of which the spacing is wrong, you can get:
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
This is a significant function, which can be applied in many situations, for example, Korean speech recognition.
## Corpus
In this project we used Sejong Corpus for training. Sejong Corpus has the format as follow.
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
https://github.com/kakao/khaiii/wiki/%EB%AC%B8%EC%A2%85-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8
Referring to khaiii's site to correct the format error of Sejong Corpus. 
And then run this code to get the corpus.txt for training. 
```
python map_char_to_tag.py -c corpus --output corpus.txt --restore-dic restore.dic
```
-c corpus is the sejong corpus directory. At the same time we got the restore.dic to restore morpheme. 
Put the `restore.dic` under `./rsc` directory (not necessary, cause I've done it for you ^^).

corpus.txt has this format, It has the morphemes for every word (actually it's called 어절 in Korean, 
which means a unit):
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
The restore dictionary has this kind of format:
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
### Preperation
Refer to the above to get a corpus.txt with correct format. Put it in the root folder.

Run this code to seperate the corpus into train/dev/test corpus.
```
python split_corpus.py --input corpus.txt -o corpus
```
Run this code to make the vocab file.
```
python make_vocab.py --input corpus.train
```
vocab.in
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
vocab.out
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
### Careful!
Because vocab.in is extracted from corpus.train, and when sepperating the corpus to train/dev/test,
actually it's random. So every time you do this process, you will get different training corpus and vocab.in.
Therefore the model for demo in this project is not applicable anymore if you construct a new vocab.in. 
But still, you can train your own model and use it for inference.
### Start training
First we should export the python path.
```
export PYTHONPATH=/path/to/hykom/src
```
And then we start training.
```
python train.py -i corpus
```
We highly recommend using NVIDIA's Automatic Mixed Precision (AMP) for acceleration.
https://github.com/NVIDIA/apex
Install the apex first and then turn on the "-fp16" option.
## Users Dictionary
Add necessary items to preanal.manual unber ./rsc directory to make your own users dictionary.
It has the follow format:
```
복지택시	복지택시/NNG
로그인	로그인/NNG
재부팅	재부팅/NNG
가즈아	가/VV + 즈아/EC
```
## RESRTful API
We've made a demo and a RESTful api for inference using gunicorn. Please check inference_gunicorn.py
The usage is simple. You can choose to output only nouns, or nouns + verbs etc.
