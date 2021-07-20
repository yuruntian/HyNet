# HyNet: Learning Local Descriptor with Hybrid Similarity Measure and Triplet Loss

This a Pytorch implementation of our paper:

**"HyNet: Learning Local Descriptor with Hybrid Similarity Measure and Triplet Loss"**   
Yurun Tian, Axel Barroso-Laguna, Tony Ng, Vassileios Balntas, Krystian Mikolajczyk. NeurIPS 2020. [[arXiv](https://arxiv.org/abs/2006.10202)]

## Requirements
- Python 3
- [PyTorch](https://pytorch.org/get-started/locally/) tested on 1.4.0-1.9.0
- [opencv-python (cv2)](https://pypi.org/project/opencv-python/) tested on 3.3.0.10
- numpy
- PIL
- [tqdm](https://github.com/tqdm/tqdm)

## Training
- We provide codes for training on the [UBC](http://matthewalunbrown.com/patchdata/patchdata.html) data set and the [HPatches](https://github.com/hpatches/hpatches-dataset) data set. 
The downloaded data should be organised as the following folder structure:
>data_root
>> -- liberty
>
>> -- notredame
>
>> -- yosemite
>
>> -- hpatches-benchmark-master

Specify the training data path and path saving path for the code:
```
python train.py --data_root=data_root --network_root= save_root
```

- To accelerate the training, all the data needed will be generated and saved at the fist run. 



## Citation
If you use this repository in your work, please cite our paper:
```bibtex
@inproceedings{hynet2020,
 author = {Tian, Yurun and Barroso Laguna, Axel and Ng, Tony and Balntas, Vassileios and Mikolajczyk, Krystian},
 title = {HyNet: Learning Local Descriptor with Hybrid Similarity Measure and Triplet Loss},
 booktitle = {NeurIPS},
 year      = {2020}
}
```