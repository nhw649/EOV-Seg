# EOV-Seg: Efficient Open-Vocabulary Panoptic Segmentation
## 📋 Table of content
    1. [🛠️ Install](#1)
    2. [✏️ Usage](#2)
           1. [Prepare Datasets](https://github.com/bytedance/fc-clip/blob/main/datasets/README.md)
           2. [Training](#2)
           3. [Inference](#2)
    3. [❤️ Acknowledgement](#3)


## 🛠️ Install <a name="1"></a> 
```bash
conda create --name eov-seg python=3.8 -y
conda activate eov-seg
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

pip install -U opencv-python
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git

git clone https://github.com/nhw649/EOV-Seg.git
cd EOV-Seg
pip install -r requirements.txt
```

## ✏️ Usage <a name="2"></a> 
1、Please follow [this](https://github.com/bytedance/fc-clip/blob/main/datasets/README.md) to prepare datasets for training.

2、To train a model, use

```bash
python train_net.py --num-gpus 4 --config-file configs/EOV-Seg-convnext-l.yaml
```

3、To evaluate a model's performance, use

```bash
python train_net.py --config-file configs/EOV-Seg-convnext-l.yaml --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```

## License

EOV-Seg is released under the [Apache 2.0 license](LICENSE).

## ❤️ Acknowledgement <a name="3"></a> 
-   [FC-CLIP](https://github.com/bytedance/fc-clip)
-   [detectron2](https://github.com/facebookresearch/detectron2)