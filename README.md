### Enhancing Cross-Modal Person Re-Identification via Frequency-Spatial-Channel Collaborative Networks

Official PyTorch implementation of the paper Enhancing Cross-Modal Person Re-Identification via Frequency-Spatial-Channel Collaborative Networks. 

Frequency-Spatial-Channel Collaborative Networks(FSC-Net)
Pytorch Code of FSC-Net method for Cross-Modality Person Re-Identification (Visible-Infrared Re-ID) on RegDB dataset [3], SYSU-MM01 dataset [4] and LLCM dataset [5]. This code is based on [mangye16](https://github.com/mangye16/Cross-Modal-Re-ID-baseline) and [DEEN](https://github.com/ZYK100/LLCM) [1, 2, 5].


### 1. Prepare the datasets.

- (1) RegDB Dataset [3]: The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html) by submitting a copyright form.

    - (Named: "Dongguk Body-based Person Recognition Database (DBPerson-Recog-DB1)" on their website). 
  
- (2) SYSU-MM01 Dataset [4]: The SYSU-MM01 dataset can be downloaded from this [website](http://isee.sysu.edu.cn/project/RGBIRReID.htm).

   - run `python pre_process_sysu.py` to pepare the dataset, the training data will be stored in ".npy" format.
 
- (3) LLCM Dataset [5]: The LLCM dataset can be downloaded by sending a signed [dataset release agreement](https://github.com/ZYK100/LLCM/blob/main/Agreement/LLCM%20DATASET%20RELEASE%20AGREEMENT.pdf) copy to zhangyk@stu.xmu.edu.cn. 


### 2. Training.
Train a model by:
```
python train_FSC.py --dataset llcm --gpu 1
```
--dataset: which dataset "llcm", "sysu" or "regdb".

--gpu: which gpu to run.

You may need mannully define the data path first.

Parameters: More parameters can be found in the script.

### 3. Testing.
Test a model on LLCM, SYSU-MM01 or RegDB dataset by
```
python test.py --mode all --mode_regdb/llcm 1 True --resume 'model_path' --gpu 1 --dataset llcm
```
--dataset: which dataset "llcm", "sysu" or "regdb".

--mode_regdb/llcm: "0" or "1" for Infrared to Visible or Visible to Infrared (For regdb and llcm dataset).

--mode: "all" or "indoor" all search or indoor search (only for sysu dataset).

--resume: the saved model path.

--gpu: which gpu to run.



###  5. References.

[1] M. Ye, J. Shen, G. Lin, T. Xiang, L. Shao, and S. C., Hoi. 	Deep learning for person re-identification: A survey and outlook. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2020.

[2] M. Ye, X. Lan, Z. Wang, and P. C. Yuen. Bi-directional Center-Constrained Top-Ranking for Visible Thermal Person Re-Identification. IEEE Transactions on Information Forensics and Security (TIFS), 2019.

[3] D. T. Nguyen, H. G. Hong, K. W. Kim, and K. R. Park. Person recognition system based on a combination of body images from visible light and thermal cameras. Sensors, 17(3):605, 2017.

[4] A. Wu, W.-s. Zheng, H.-X. Yu, S. Gong, and J. Lai. Rgb-infrared crossmodality person re-identification. In IEEE International Conference on Computer Vision (ICCV), pages 5380–5389, 2017.

[5] Zhang Y, Wang H. Diverse Embedding Expansion Network and Low-Light Cross-Modality Benchmark for Visible-Infrared Person Re-identification[J]. arXiv preprint arXiv:2303.14481, 2023.

### 6. Contact

If you have any question, please feel free to contact us. 201909018@gznu.edu.cn.
