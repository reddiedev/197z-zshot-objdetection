# obtain link to image
from pycocotools.coco import COCO
import numpy as np

print("[0]: loading coco annotations...")

dataDir='../coco'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
# initialize COCO api for instance annotations
coco = COCO(annFile)
# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
# print('COCO categories: \n{}\n'.format(' '.join(nms)))
nms = set([cat['supercategory'] for cat in cats])
# print('COCO supercategories: \n{}'.format(' '.join(nms)))

# get all category IDs
catIds = coco.getCatIds()
# choose random category ID
catId = catIds[np.random.randint(0,len(catIds))]
# choose random image from category
imgIds = coco.getImgIds(catIds=[catId] )
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
imgUrl = img['coco_url']

print(f"[1]: generating image masks for {imgUrl}")

# automatically generate masks using SAM
import torch
import torchvision
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())


# crop source image to each mask
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

from skimage import io
import matplotlib.pyplot as plt

image = io.imread(imgUrl)

plt.figure(figsize=(20,20))
plt.imshow(image)
plt.axis('off')
print(f"[1]: saving source image")
plt.savefig("../output/source.jpg")

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os

sam_checkpoint = os.path.join("../checkpoints", "sam_vit_b_01ec64.pth")
model_type = "vit_b"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)
print(f"[1]: generating masks")
masks = mask_generator.generate(image)
print(len(masks))
print(masks[0].keys())

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
print(f"[1]: saving masks image")
plt.savefig("../output/generated-masks.jpg")

mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)
print(f"[1]: generating masks-2")
masks2 = mask_generator_2.generate(image)
len(masks2)
print(masks2[0].keys())
plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks2)
plt.axis('off')
print(f"[1]: saving masks-2 image")
plt.savefig("../output/generated-masks-2.jpg")


# for each mask image, annotate using open-clip


# evaluate generated annotations