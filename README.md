# MARAN

MARAN: Supporting Awareness of Users’ Routines and Preferences for next POI Recommendation Based on Spatial Aggregation

## Requirements

```
python==3.9.12
torch==1.11.0
pandas==1.4.2
torch_geometric==2.1.0
numpy==1.22.3
matplotlib==3.5.1
tqdm==4.64.0
pygeohash==1.2.0
```

You can install the requirements using `pip install -r requirements.txt`

## Datasets

Put [MARAN_dataset.zip](https://drive.google.com/file/d/1zup1aP2JvkrpdYNXQnqlvzsQvZcgA7QN/view?usp=share_link) into `./Datasets/`.

## Pretrained Model

The checkpoint file can be found [here](https://drive.google.com/file/d/1uqTdjJaMnxJhPIRpQRlSH5Wxqj5TDybr/view?usp=share_link) and put them into `./Model/`.

## Training

```bash
# Gowalla
python main.py --mode=train --dataset=Gowalla --gpu=1

# Foursquare
python main.py --mode=train --dataset=Foursquare --gpu=1
```

## Evaluation

```bash
python main.py --mode=test --dataset=Gowalla  --gpu=1 --model_path='./Model/model_gowalla.pkl'
```