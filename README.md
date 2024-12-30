# CVML-Pose: Convolutional VAE based Multi-Level Network for Object 3D Pose Estimation <br> <span style="float: right"><sub><sup>IEEE ACCESS 2023</sub></sup></span>

<img width="1618" alt="new ga" src="https://github.com/user-attachments/assets/41a11257-84ca-4c19-b189-5ee2ba84fbc5">

Authors: [Jianyu Zhao](https://scholar.google.com/citations?user=b6qSMLwAAAAJ&hl=en), [Edward Sanderson](https://scholar.google.com/citations?user=ea4c7r0AAAAJ&hl=en&oi=ao), [Bogdan J. Matuszewski](https://scholar.google.co.uk/citations?user=QlUO_oAAAAAJ&hl=en)

Paper link: https://ieeexplore.ieee.org/document/10040668

If you find this work useful to your research, please consider citing:

```
@article{zhao2023cvml,
  title={CVML-Pose: Convolutional VAE Based Multi-Level Network for Object 3D Pose Estimation},
  author={Zhao, Jianyu and Sanderson, Edward and Matuszewski, Bogdan J},
  journal={IEEE Access},
  volume={11},
  pages={13830--13845},
  year={2023},
  publisher={IEEE}
}
```

An extended multi-object version ([CVAM-Pose](https://github.com/JZhao12/CVAM-Pose)) has been accepted at BMVC 2024, you may consider citing:
```
@inproceedings{zhao2024cvam,
  title={CVAM-Pose: Conditional Variational Autoencoder for Multi-Object Monocular Pose Estimation},
  author={Zhao, Jianyu and Quan, Wei and Matuszewski, Bogdan J},
  booktitle={The 35th British Machine Vision Conference 2024, {BMVC} 2024, Glasgow, UK, November 25-28, 2024},
  publisher={BMVA},
  year={2024},
  url={https://bmva-archive.org.uk/bmvc/2024/papers/Paper_967/paper.pdf}
}
```

## 1. Usage

### 1.1 Download data

Download the repository, navigate to the CVML-Pose directory, download and unzip the following datasets from the [BOP challenge](https://bop.felk.cvut.cz/datasets/).

```
cd CVML-Pose
mkdir original\ data                  # make the "original data" folder
cd original\ data

export SRC=https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main
wget $SRC/lm/lm_base.zip              # Linemod base archive
wget $SRC/lm/lm_models.zip            # Linemod 3D object's model
wget $SRC/lm/lm_test_all.zip          # Linemod real images
wget $SRC/lm/lm_train_pbr.zip         # Linemod PBR images
wget $SRC/lmo/lmo_base.zip            # Linemod-Occluded base archive
wget $SRC/lmo/lmo_test_bop19.zip      # The BOP version of the Linemod-Occluded test images

unzip lm_base.zip                     # Contains folder "lm"
unzip lm_models.zip -d lm             # Unpacks to "lm"
unzip lm_test_all.zip -d lm           # Unpacks to "lm"
unzip lm_train_pbr.zip -d lm          # Unpacks to "lm"
unzip lmo_base.zip                    # Contains folder "lmo"
unzip lmo_test_bop19.zip -d lmo       # Unpacks to "lmo"
```

Since many approaches (BB8, SingleShotPose, PVNet, etc) divide the original Linemod real images into train/test set, and evaluate on the test set, we download the test list from [SingleShotPose](https://github.com/microsoft/singleshotpose) to make a fair comparison.
```
wget -O LINEMOD.tar --no-check-certificate "https://onedrive.live.com/download?cid=05750EBEE1537631&resid=5750EBEE1537631%21135&authkey=AJRHFmZbcjXxTmI"
tar xf LINEMOD.tar
```

**Important update on 2024/05/08**: For whatever reason this Linemod dataset cannot be downloaded from the SingleShotPose repository, so I have uploaded it through [Google Drive](https://drive.google.com/file/d/107xiwRzM5dIvpb93CaAU79zstuuJ4Bj0/view?usp=drive_link)

Our method does not include object detection/segmentation, we use the Mask-R-CNN detector pretrained by [CosyPose](https://github.com/ylabbe/cosypose). The CosyPose environment can be created using the following commands:
```
cd CVML-Pose
git clone --recurse-submodules https://github.com/ylabbe/cosypose.git

cd cosypose
conda env create -n cosypose --file environment.yaml
conda activate cosypose
git lfs pull
python setup.py install
```

Download the pretrained Mask-R-CNN detector:
```
conda activate cosypose
cd /home/jianyu/CVML-Pose/cosypose    # example directory, replace with yours
mkdir local_data

python -m cosypose.scripts.download --model=detector-bop-lmo-pbr--517542
```


### 1.2 CVML-Pose environment

To set up the CVML-Pose environment, run:
```
conda create -n CVML-Pose python=3.9.7
conda activate CVML-Pose

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install matplotlib pandas
conda install -c conda-forge scikit-learn
pip install pyrender pypng opencv-python
```

### 1.3 Data preprocessing

As described in the paper, each target object is cropped into a square shape from the scene based on the bounding box. You can run the following scripts to get training/test images.
```
cd CVML-Pose
conda activate CVML-Pose

python scripts/pbr.py                   # extract input training images
python scripts/recon.py --object 1      # generate ground truth reconstruction images
python scripts/test_gt.py               # extract test images with ground truth bounding box
```

To use the pretrained Mask-R-CNN detector, go to this script ```scripts/test_mask.py```, change the code ```sys.path.append('/home/jianyu/CVML-Pose/cosypose/')``` to your CosyPose path, then run the script:
```
python scripts/test_mask.py             # extract test images with the detector bounding box
```
### 1.4 Train CVML-Pose

To train the CVML-Pose, there are 3 models and 13 objects available:
+ model choice: CVML_base, CVML_18, CVML_34; 
+ object choice: 1-15 except 3&7;
```
cd CVML-Pose
conda activate CVML-Pose

python scripts/cvml_ae.py --model CVML_18 --object 1      # train the CVML_AE models
python scripts/latent.py --model CVML_18 --object 1       # get latent variables
python scripts/mlp_r.py --model CVML_18 --object 1        # train the rotation MLP
python scripts/mlp_t.py --model CVML_18 --object 1        # train the translation MLP and KNN
python scripts/topology.py                                # train multiple objects and visualize the latent space
```

### 1.5 Evaluation

To evaluate the CVML-Pose, we use ADD/MSSD/MSPD which have been implemented in the [BOP Toolkit](https://github.com/thodan/bop_toolkit).
```
cd CVML-Pose
git clone https://github.com/thodan/bop_toolkit.git
```

To use the BOP Toolkit, go to this script ```scripts/evaluate.py```, change the code ```sys.path.append('/home/jianyu/CVML-Pose/bop_toolkit/')``` to your BOP Toolkit path, then evaluate the estimated pose:
+ model choice: CVML_base, CVML_18, CVML_34; 
+ data choice: lm, lmo;
+ type choice: gt, mask;
```
cd CVML-Pose
conda activate CVML-Pose

python scripts/evaluate.py --model CVML_18 --data lm --type gt
```

### 1.6 Real-time pose estimation

To implement real-time pose estimation with CVML-Pose, go to the two scripts ```scripts/realtime.py``` and ```scripts/realtime_no_detect.py```, change the code ```sys.path.append('/home/jianyu/CVML-Pose/cosypose/')``` to your CosyPose path, then run:
+ object choice: 1-15 except 3&7;
```
cd CVML-Pose
conda activate CVML-Pose

python scripts/realtime.py --object 1             # estimate object 3D pose in real-time with the pretrained Mask-R-CNN detector
python scripts/realtime_no_detect.py --object 1   # estimate object 3D pose in real-time in a fixed location(no detector)
```

### 1.7 Trained models and results

In terms of the train models (weights), although the data we used are publicly available, we cannot guarantee whether we can distribute the data including the weights trained on the data. But you can always follow the guidance to train the models, and any questions about training are welcome. We are very sorry for any inconvenience.

With respect to the results, we have opened the results of our method on Linemod and the BOP version of the Linemod-Occluded data. Please feel free to download them through [Google Drive](https://drive.google.com/drive/folders/1uJJ1YXLyggwvFicBZVgze_lbKXoVIB5H?usp=share_link).

## 2. License

This repository is released under the Apache 2.0 license as described in the [LICENSE](https://github.com/JZhao12/CVML-Pose/blob/main/LICENSE).

## 3. Commercial use

We allow commercial use of this work, as permitted by the [LICENSE](https://github.com/JZhao12/CVML-Pose/blob/main/LICENSE). However, where possible, please inform us of this use for the facilitation of our impact case studies.

## 4. Acknowledgements

Contributions to this paper by B.J. Matuszewski were in part supported by the Engineering and Physical Sciences Research Council [grant number EP/K019368/1].

This work makes use of multiple existing datasets which are openly available at:

+ Hodaň, T., Sundermeyer, M., Drost, B., Labbé, Y., Brachmann, E., Michel, F., Rother, C. and Matas, J., 2020, August. BOP challenge 2020 on 6D object localization. In European Conference on Computer Vision (pp. 577-594). Springer, Cham. [link](https://bop.felk.cvut.cz/datasets/)

+ Hinterstoisser et al.: Model based training, detection and pose estimation of texture-less 3d objects in heavily cluttered scenes, ACCV 2012. [link](https://campar.in.tum.de/Main/StefanHinterstoisser)

+ Tekin, B., Sinha, S.N. and Fua, P., 2018. Real-time seamless single shot 6d object pose prediction. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 292-301). [link](https://github.com/microsoft/singleshotpose)

+ Brachmann et al.: Learning 6d object pose estimation using 3d object coordinates, ECCV 2014. [link](https://heidata.uni-heidelberg.de/dataset.xhtml?persistentId=doi:10.11588/data/V4MUMX)

This work makes use of multiple existing code which are openly available at:
+ [BOP Toolkit](https://github.com/thodan/bop_toolkit)
+ [CosyPose](https://github.com/ylabbe/cosypose)
+ [Pyrender](https://github.com/mmatl/pyrender)
+ [PyTorch3D](https://github.com/facebookresearch/pytorch3d)
+ [PyTorch VAE](https://github.com/AntixK/PyTorch-VAE)
+ [Dive into Deep Learning](https://d2l.ai/index.html)

This work makes use of an existing object detection model which is openly available at:
+ [CosyPose](https://github.com/ylabbe/cosypose)

## 5. Additional information

[Student Profile](https://www.uclan.ac.uk/articles/research/jianyu-zhao)

[UCLan Computer Vision and Machine Learning (CVML) Group](https://www.uclan.ac.uk/research/activity/cvml)

Contact: jzhao12@uclan.ac.uk
