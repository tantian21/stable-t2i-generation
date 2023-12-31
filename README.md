# Bridging the Domain Gap Arising from Text Description Differences for Stable Text-to-Image Generation(ICASSP 2024)

Codes and pre-trained models for 'BRIDGING THE DOMAIN GAP ARISING FROM TEXT DESCRIPTION DIFFERENCES FOR STABLE TEXT-TO-IMAGE GENERATION'.
Official Pytorch implementation for the paper Bridging the Domain Gap Arising from Text Description Differences for Stable Text-to-Image Generation by Tian Tan, Weimin Tan, Xuhao Jiang, Yueming Jiang, Bo Yan.

# Environment
python        -3.9.13

numpy         -1.21.5

pytorch-cuda  -11.6

1×16GB NVIDIA GPU

# Datasets
1.Download the preprocessed metadata for [birds](https://drive.google.com/file/d/1I6ybkR7L64K8hZOraEZDuHh0cCJw5OUj/view?usp=sharing) [coco](https://drive.google.com/file/d/15Fw-gErCEArOFykW3YTnLKpRcPgI_3AB/view?usp=sharing) and extract them to data/

2.Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) image data. Extract them to data/birds/

3.Download [coco2014](http://cocodataset.org/#download) dataset and extract the images to data/coco/images/

# Evaluation
Download Pretrained Model

For bird. Download and save it to ./code/saved_models/bird/

For coco. Download and save it to ./code/saved_models/coco/

# Performance
|Model|CUB|Coco|
|---|---|---|
|FID↓|10.14|14.59|
|IS↑|5.20|20.09|
