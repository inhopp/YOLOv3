# YOLOv3
> YOLOv3 from scartch (pytorch)

## Repository Directory 

``` python 
├── YOLO_pytorch
        ├── datasets
        │     └── PASCAL_VOC
        ├── data
        │     ├── __init__.py
        │     └── dataset.py
        ├── utils
        │      ├── bbox_tools.py
        │      ├── IoU.py
        │      ├── mAP.py
        │      └── nms.py      
        ├── option.py
        ├── model.py
        ├── loss.py
        ├── train.py
        └── inference.py
```

- `datasets/PASCAL_VOC` : subset of PASCAL_VOC ( train/val/test = 800/100/100 )
- `data/__init__.py` : dataset loader
- `data/dataset.py` : data preprocess & get annotations
- `utils` : utils for models and pre/post processing
- `option.py` : Environment setting


<br>


## Tutoral

### Clone repo and install depenency

``` python
# Clone this repo and install dependency
!git clone https://github.com/inhopp/YOLOv3.git
```

<br>


### train
``` python
python3 train.py
    --device {}(defautl: cpu) \
    --data_name {}(default: PASCAL_VOC) \
    --lr {}(default: 1e-5) \
    --n_epoch {}(default: 10) \
    --num_workers {}(default: 4) \
    --batch_size {}(default: 16) \
    --eval_batch_size {}(default: 16) \
    --conf_threshold {}(default: 0.05) \
    --map_iou_threshold {}(default: 0.5) \
    --nms_iou_threshold {}(default: 0.45) \
```

### inference
```python
python3 inference.py
    --device {}(defautl: cpu) \
    --data_name {}(default: PASCAL_VOC) \
    --num_workers {}(default: 4) \
    --conf_threshold {}(default: 0.05) \
    --map_iou_threshold {}(default: 0.5) \
    --nms_iou_threshold {}(default: 0.45) \
```

<br>

#### Main Reference
https://github.com/aladdinpersson/Machine-Learning-Collection