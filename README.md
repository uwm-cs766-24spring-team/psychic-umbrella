# Real-time Crowd Detection for Shared Facilities using Surveillance Cameras
by Mondo Jiang, William Sun, and Lao Chang

## Introduction
In this main repository, we present all methodologies for real-time detection for CS766 Computer Vision course at UW-Madison. Our focus is using CV on facilities such as restaurants, libraries, gyms, etc., where occupancy and congestion status can be provided to individuals. This can help save time and lower wait status for such facilities.

We were particularly interested in crowd counting through computer vision due to the lackluster real-time occupancy reports currently provided by UW-Madison's gyms.

Our objective and goals are the following:
1. Develop and evaluate a baseline for direct head-counting given images  
2. Develop and evaluate a baseline for indirect people-counting given images  
3. Develop and evaluate deep learning methods for crowd counting given images  
4. Apply the above techniques on a custom, privacy-ensured dataset collected from the gym 

## [Dataset](https://drive.google.com/drive/u/2/folders/1Kvuk0fiKKyZUmWqEIRx6_jgfvQuHyO7t):
1) ShanghaiTech
2) PETS2009
3) Fine-Grained

## Run Methods:
1. Download above dataset, unzip, and place into top directory
2. Run Baseline 1 directly with 'methods/direct/DirectCV.py'
3. Run Baseline 2 directly by opening 'methods/baseline-indirect/gaussian_segmentation.ipynb'
4. Run MCNN directly with 'methods/deep-learning-mcnn/test_new_data.py'

Command for generating video from frames: ffmpeg -framerate 5 -pattern_type glob -i '*.png' -pix_fmt yuv420p out.mp4

