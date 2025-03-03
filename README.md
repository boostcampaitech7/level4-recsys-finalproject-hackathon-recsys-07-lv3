# 🎞️ Cold Start Problem on Your RecSys
> RecSys 데이터셋을 선정해 콜드 스타트 문제를 정의 및 관측하고, 해결하는 솔루션을 제시하는 프로젝트입니다.

## Team
|곽정무|박준하|박태지|배현우|신경호|이효준
|:-:|:-:|:-:|:-:|:-:|:-:
|<img  width="100" height="100" src = 'https://avatars.githubusercontent.com/u/20788198?v=4'>|<img  width="100" height="100" src = 'https://avatars.githubusercontent.com/u/81938013?v=4'>|<img  width="100" height="100" src = 'https://avatars.githubusercontent.com/u/112858891?v=4'>|<img  width="100" height="100" src = 'https://avatars.githubusercontent.com/u/179800298?v=4'>|<img  width="100" height="100" src = 'https://avatars.githubusercontent.com/u/103016689?s=64&v=4'>|<img  width="100" height="100" src = 'https://avatars.githubusercontent.com/u/176903280?v=4'>|
|<a href = 'https://github.com/jkwag'><img src = 'https://img.shields.io/badge/github%20pages-121013?style=for-the-badge&logo=github&logoColor=white'> </a>|<a href = 'https://github.com/joshua5301'><img src = 'https://img.shields.io/badge/github%20pages-121013?style=for-the-badge&logo=github&logoColor=white'> </a>|<a href = 'https://github.com/spsp4755'><img src = 'https://img.shields.io/badge/github%20pages-121013?style=for-the-badge&logo=github&logoColor=white'> </a>|<a href = 'https://github.com/hwbae42'><img src = 'https://img.shields.io/badge/github%20pages-121013?style=for-the-badge&logo=github&logoColor=white'> </a>|<a href = 'https://github.com/Human3321'><img src = 'https://img.shields.io/badge/github%20pages-121013?style=for-the-badge&logo=github&logoColor=white'> </a>|<a href = 'https://github.com/Jun9096'><img src = 'https://img.shields.io/badge/github%20pages-121013?style=for-the-badge&logo=github&logoColor=white'> </a>|

## 프로젝트 구조
```
📦 level4-recsys-finalproject-hackathon-recsys-07-lv3
├── 📜 README.md
├── 📂 cold_emb_retrain
│   ├── 📂 data
│   │   ├── 📂 ml-1m
│   │   │   ├── 📜 train.csv
│   │   │   └── 📜 val.csv
│   │   └── 📂 ml-20m
│   │       ├── 📜 train1.csv
│   │       ├── 📜 train2.csv
│   │       └── 📜 val.csv
│   ├── 📜 main.py
│   ├── 📜 readme.md
│   ├── 📜 requirements.txt
│   ├── 📂 saved
│   │   └── 📜 ml-1m_best_model.pt
│   └── 📂 src
│       ├── 📜 __init__.py
│       ├── 📜 dataset.py
│       ├── 📜 model.py
│       ├── 📜 sampler.py
│       ├── 📜 sampler_cpp.cpp
│       ├── 📜 trainer.py
│       └── 📜 utils.py
├── 📂 docs
│   ├── 📜 TVING_RecSys_팀 리포트(07조).pdf
│   └── 📜 Tving_RecSys_발표자료.pdf
└── 📂 neg_sampling_finetune
    ├── 📜 main.py
    ├── 📂 notebooks
    │   ├── 📜 cold_baseline.ipynb
    │   └── 📜 setup.ipynb
    ├── 📜 readme.md
    ├── 📜 requirements.txt
    └── 📂 src
        ├── 📂 data
        │   ├── 📜 __init__.py
        │   ├── 📜 dataloader.py
        │   └── 📜 dataset.py
        ├── 📂 models
        │   ├── 📜 NCF.py
        │   └── 📜 __init__.py
        ├── 📂 train
        │   ├── 📜 __init__.py
        │   ├── 📜 loss.py
        │   ├── 📜 metrics.py
        │   └── 📜 trainer.py
        └── 📂 utils
            ├── 📜 __init__.py
            └── 📜 setting.py
```


## 개발환경 
- python 3.10.15

 ## 기술스택
<img src = 'https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54'> <img src = 'https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white'> <img src= 'https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white'> <img src ='https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white'> 

### 협업툴
<img src ='https://img.shields.io/badge/jira-%230A0FFF.svg?style=for-the-badge&logo=jira&logoColor=white'> <img src = 'https://img.shields.io/badge/confluence-%23172BF4.svg?style=for-the-badge&logo=confluence&logoColor=white'>

## 라이브러리 설치
```shell
$ pip install -r requirement.txt
```

## 기능 및 예시
- Cold Embedding Retraining

cold_emb_retrain 폴더로 이동하면 자세한 안내사항을 확인하실 수 있습니다.

<br/>

- Hybrid Negative Sampling

먼저 Uniform Random Negative Sampling 을 통해 모델이 Cold User의 성능이 최고 성능을 달성할 때까지 학습합니다. 이후 저장된 모델을 불러와 상위 30% 인기도 영화를 샘플링하는
Popularity 기반 Hard Negative Sampling을 통해 모델을 추가적으로 학습합니다. 경로를 neg_sampling_finetune으로 설정한 뒤 다음의 명령어를 실행하면 됩니다.

```shell
$ python main.py 
```
