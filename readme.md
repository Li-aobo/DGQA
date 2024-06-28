# DGQA
This is the official pytorch implementation of CVPR2024 ["Bridging the Synthetic-to-Authentic Gap: Distortion-Guided Unsupervised Domain Adaptation for Blind Image Quality Assessment"]([https://arxiv.org/abs/2003.08932](https://arxiv.org/abs/2405.04167)).

## Requirement
+ Python 3.6
+ pytorch 1.4.0
+ torchvision 0.5.0

## Usage
1. Download the pretrained model and put them into ```model/```. 

    Google Drive: https://drive.google.com/file/d/12BZ0Xts5xTj0ppqyZRA4wFuIt6s3ONG2/view?usp=drive_link
    
    百度网盘：链接：https://pan.baidu.com/s/1Z9KjI6wXj6Kr1vy4YgyWew 提取码：9do9

2. To selecting the appropriate distortions for the target dataset, you need to run:
    ```
    python sel_dist.py --dataset target_dataset --root target_dataset_root
    ```

## Results
The distortion types selected by DGQA on several common datasets are shown in the following table.
Selecting only images with these distortion types in KADID-10K for training can significantly improve generalization performance and stability.

|           | #1  | #2  | #3  | #4  | #7  | #9  | #10 | #11 | #12 | #16 | #17 | #18 | #19 | #21 | #22 | #23 | #25 |
|-----------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| LIVEC     | √   |     | √   |     |     |     | √   |     |     |     | √   | √   |     |     |     |     | √   |
| Koniq-10k | √   |     | √   |     |     |     |     |     |     | √   | √   | √   |     |     |     |     | √   |
| BID       | √   |     | √   |     |     | √   |     |     |     |     |     |     |     | √   |     | √   | √   |
| PIPAL-1   | √   | √   | √   |     |     | √   |     |     |     |     | √   | √   |     |     |     |     | √   |
| PIPAL-2   | √   | √   | √   |     |     | √   |     |     |     |     | √   |     |     |     |     | √   |     |
| PIPAL-3   | √   |     | √   |     |     |     |     |     |     | √   | √   | √   | √   |     |     |     | √   |
| PIPAL-4   | √   | √   | √   |     | √   | √   | √   |     |     |     | √   | √   |     |     |     |     | √   |
| PIPAL-5   | √   | √   | √   | √   |     | √   |     |     |     |     |     |     |     |     |     |     |     |
| PIPAL-6   | √   | √   | √   |     |     | √   | √   | √   | √   | √   | √   | √   | √   | √   | √   |     | √   |


For LIVEC and Koniq-10k, these distortion types are more recommanded, which are selected based on a greedy selection strategy.

|           | #1  | #2  | #3  | #9  | #10 | #17 | #18 | #20 | #25 | 
|-----------|-----|-----|-----|-----|-----|-----|-----|-----|-----| 
| LIVEC     | √   |     | √   | √   |     |     | √   | √   | √   | 
| Koniq-10k | √   | √   |     |     | √   | √   |     | √   | √   | 



## Citation
If you find our code helpful for your research, please consider citing:

```
@inproceedings{li2024bridging,
  title={Bridging the Synthetic-to-Authentic Gap: Distortion-Guided Unsupervised Domain Adaptation for Blind Image Quality Assessment},
  author={Li, Aobo and Wu, Jinjian and Liu, Yongxu and Li, Leida},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={28422--28431},
  year={2024}
}
```