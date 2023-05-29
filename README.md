# EEE 197z Project 1 - Zero Shot Object Detection
use SAM and OpenCLIP to perform zero-shot object detection using COCO 2017 val split. 

*Author: Sean Red Mendoza | 2020-01751 | scmendoza5@up.edu.ph*

## Tools/ References
- [SegmentAnything](https://github.com/facebookresearch/segment-anything)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [Coco 2017 Validation Dataset](https://cocodataset.org/#home)
- [roatienza/mlops](https://github.com/roatienza/mlops)
- [roatienza/Deep-Learning-Experiments](https://github.com/roatienza/Deep-Learning-Experiments)
- [Google Cloud G2 GPU VM (2x Nvidia L4)](https://cloud.google.com/blog/products/compute/introducing-g2-vms-with-nvidia-l4-gpus)

## Goals
- [x] Integrate SAM with OpenCLIP
- [x] Compare CoCo Ground Truth vs Model Prediction vs Yolov8 Prediction
- [x] Support random image pick from CoCo 2017 validation split OR manual image upload (link)
- [x] Show Summary Statistics

## Approach
![Project Approach](images/project_approach.png)

1. Automatically generate object masks for source image using [SAM AutomaticMaskGenerator](https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb)
2. Filter masks using predicted_iou, stability score, and mask image area to garner better masking results
3. Crop source image based on generated masks
4. Run each crop under [OpenCLIP](https://github.com/mlfoundations/open_clip#pretrained-model-interface) (trained with [CoCo paper labels](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/))
5. Filter labels for most probable labels based on generated text probability
6. Evaluate mask bounding boxes using [mAPs score](https://github.com/tylin/coco-caption/blob/master/cocoEvalCapDemo.ipynb) and label accuracy (Top-1 vs Top-5) from ground truth labels
7. Tune SAM Model parameters to achieve better mAPs performance
8. Select appropriate OpenCLIP [pretrained model](https://github.com/mlfoundations/open_clip#pretrained-model-interface) to achieve better mAPs and label accuracy

## Notes
- It is recommended to use a CUDA-powered GPU with at least 40 GB VRAM, such as 2x L4s (current implementation), A100 40GB, or anything similar
    - if hardware resources are limited, it is recommended to use lower `points_per_batch` setting in SAM as well as to use a lighter pretrained model in OpenCLIP
- It is recommended to clone the repository for easier use, so you don't have to manually download any required files
- Due to hardware limitations, I am running the repo in a Google Cloud VM Instance. You may also consider leveraging [Credits](https://cloud.google.com/billing/docs/how-to/edu-grants) to make high-level computing accessible
- Ultimately, the performance of the system is limited because pretrained models are used instead of using the CoCo 2017 training dataset. This includes fine calibration of the SAM AutomaticMaskGenerator in comparison to the performance of Yolo v8. Similarly, the performance of OpenCLIP is bottlenecked by the quality of chosen labels and the pretrained model used.

## Usage 
1. Duplicate this repository on a working directory
```bash
git clone https://github.com/reddiedev/197z-zshot-objdetection
cd 197z-zshot-objdetection
```

2. Install the necessary packages
```bash
pip install -r requirements.txt
```
or alternatively, install them manually
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib segment_anything opencv-contrib-python-headless
pip install open_clip_torch scikit-image validators

```

3. Run the jupyter notebook 
You may use the test images
```
../images/dog_car.jpg
https://djl.ai/examples/src/test/resources/dog_bike_car.jpg

```

4. To view complete logs and information, set `VERBOSE_LOGGING` to `TRUE`

# Acknowledgements
[Professor, Rowel Atienza](https://github.com/roatienza)


