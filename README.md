# LightNet++
This repository contains the code (PyTorch-1.0+, **W.I.P.**) for: "**LightNet++: Boosted Light-weighted Networks for Real-time Semantic Segmentation**" by Huijun Liu.  
**LightNet++** is an advanced version of **[LightNet](https://github.com/ansleliu/LightNet)**,  which purpose to get more concise model design, 
smaller models, and better performance.

- **MobileNetV2Plus**: Modified MobileNetV2 (backbone)<sup>[[1,8]](#references)</sup> + DSASPPInPlaceABNBlock<sup>[[2,3]](#references)</sup> + 
Parallel Bottleneck Channel-Spatial Attention Block (PBCSABlock)<sup>[[6]](#references)</sup> + UnSharp Masking (USM) + Encoder-Decoder Arch.<sup>[[3]](#references)</sup> + 
InplaceABN<sup>[[4]](#references)</sup>.

- **ShuffleNetV2Plus**: Modified ShuffleNetV2 (backbone)<sup>[[1,8]](#references)</sup> + DSASPPInPlaceABNBlock<sup>[[2,3]](#references)</sup> + 
Parallel Bottleneck Channel-Spatial Attention Block (PBCSABlock)<sup>[[6]](#references)</sup>+ UnSharp Masking (USM)  + Encoder-Decoder Arch.<sup>[[3]](#references)</sup> + 
InplaceABN<sup>[[4]](#references)</sup>.
 
More about **USM(Unsharp Mask)-Operator Block** see Repo: [**SharpPeleeNet**](https://github.com/ansleliu/SharpPeleeNet)

## Dependencies

- [Python3.6](https://www.python.org/downloads/)  
- [PyTorch(1.0.1+)](http://pytorch.org)  
- [inplace_abn](https://github.com/mapillary/inplace_abn)  
- [apex](https://github.com/NVIDIA/apex): Tools for easy mixed precision and distributed training in Pytorch  
- [tensorboard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard)  
- [tensorboardX](https://github.com/lanpa/tensorboard-pytorch)  
- [tqdm](https://github.com/tqdm/tqdm)  

### Datasets for Autonomous Driving
- [Cityscapes](https://www.cityscapes-dataset.com/)  
- [Mapillary Vistas Dataset](https://www.mapillary.com/dataset/vistas)  
- [Berkeley Deep Drive (BDD100K)](https://bdd-data.berkeley.edu/)  
- [ApolloScape](http://apolloscape.auto/index.html#)  


## Results

### Results on Cityscapes (Pixel-level/Semantic Segmentation)

| Model | mIoU (S.S* Mixed Precision) |Model Weight|
|---|---|---|
|**MobileNetV2Plus X1.0**|[71.5314 (**WIP**)](https://github.com/ansleliu/LightNetPlusPlus/blob/master/checkpoint/MobileNetv2Plus.csv)|[cityscapes_mobilenetv2plus_x1.0.pkl (14.3 MB)](https://github.com/ansleliu/LightNetPlusPlus/blob/master/checkpoint/cityscapes_mobilenetv2plus_x1.0.pkl)|
|**ShuffleNetV2Plus X1.0**|[69.0885-72.5255 (**WIP**)](https://github.com/ansleliu/LightNetPlusPlus/blob/master/checkpoint/ShuffleNetV2PlusX1.0.csv)|[cityscapes_shufflenetv2plus_x1.0.pkl (8.59 MB)](https://github.com/ansleliu/LightNetPlusPlus/blob/master/checkpoint/cityscapes_shufflenetv2plus_x1.0.pkl)|

* S.S.: Single Scale (1024x2048)

## Feature Visualization

<p align="center">
<img src="https://github.com/ansleliu/LightNetPlusPlus/blob/master/netviz/feat_viz/Figure_1.png" />
<img src="https://github.com/ansleliu/LightNetPlusPlus/blob/master/netviz/feat_viz/Figure_3.png" />
<img src="https://github.com/ansleliu/LightNetPlusPlus/blob/master/netviz/feat_viz/Figure_5.png" />
<img src="https://github.com/ansleliu/LightNetPlusPlus/blob/master/netviz/feat_viz/Figure_8.png" />
</p>
