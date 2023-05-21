# obtain link to image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import torch
import torchvision
from skimage import io
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
import open_clip
from PIL import Image
import os
import glob
from pprint import pprint
import json


print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

## clean output directory
files = glob.glob('../output/*.jpg')
for f in files:
    os.remove(f)

print("[0]: loading coco annotations...")

dataDir='../coco'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
# initialize COCO api for instance annotations + COCO ground truth api
cocoGt = COCO(annFile)

# get all category IDs
catIds = cocoGt.getCatIds()
# choose random category ID
catID = catIds[np.random.randint(0,len(catIds))]
# choose random image from category
imgIds = cocoGt.getImgIds(catIds=[catID] )
imgID = imgIds[np.random.randint(0,len(imgIds))]
img = cocoGt.loadImgs(imgID)[0]
imgUrl = img['coco_url']
print("ORIGINAL IMAGE")
pprint(img)

print(f"[0]: generating labels for source image...")
annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
coco_caps=COCO(annFile)
annIds = coco_caps.getAnnIds(imgIds=imgID)
anns = coco_caps.loadAnns(annIds)
# print("GROUND TRUTH ANNOTATIONS")
# pprint(anns)
ground_truth_labels = []
ground_truth = []
for ann in anns:
    ground_truth_labels.append(ann['caption'])
    words = ann['caption'].split()
    for word in words:
        ground_truth.append(word.lower())

ground_truth_values = list(set(ground_truth))

print(f"[1]: generating image masks for {imgUrl}")

# automatically generate masks using SAM

# crop source image to each mask
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))


image = io.imread(imgUrl)

print(f"[1]: saving source image")
plt.figure(figsize=(20,20))
plt.imshow(image)
plt.axis('off')
plt.savefig("../output/source.jpg",bbox_inches='tight',pad_inches = 0)
plt.close()

print("[1]: loading sam model")
sam_checkpoint = os.path.join("../checkpoints", "sam_vit_h_4b8939.pth") # sam_vit_b_01ec64 | sam_vit_h_4b8939
model_type = "vit_h"
device = torch.device   ('cuda' if torch.cuda.is_available() else 'cpu')
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
print(f"[1]: generating masks")
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    points_per_batch=128,
    pred_iou_thresh=0.95,
    stability_score_thresh=0.95,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)
masks = mask_generator.generate(image)
print(f"[1]: generated {len(masks)} masks for source image")
plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.savefig("../output/generated-masks.jpg",bbox_inches='tight',pad_inches = 0)
plt.close()


# for each mask image, annotate using open-clip


print("[2]: creating open clip model...")
model, _, transform = open_clip.create_model_and_transforms(
  model_name="coca_ViT-L-14",
  pretrained="mscoco_finetuned_laion2B-s13B-b90k"
)
model.to(device)

results = []

def generate_labels(anns):
    if len(anns) == 0:
        return
    length = len(anns)
    labels = []
    for i in range(length):
        mask = anns[i]
        x,y,w,h = mask['bbox']
        x,y,w,h = int(x), int(y), int(w), int(h)
        im = image[y:y+h, x:x+w]
        plt.figure(figsize=(20,20))
        plt.imshow(im)
        plt.axis('off')
        plt.savefig(f"../output/mask-{i}.jpg", bbox_inches='tight', pad_inches = 0)
        plt.close()
        im = Image.open(f"../output/mask-{i}.jpg").convert("RGB")
        im = transform(im).unsqueeze(0)
        im = im.to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            generated = model.generate(im)

        label = open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")
        print(f"[{i}]:",mask['predicted_iou'],mask['stability_score'], label)
        labels.append(label)
        result = {'image_id':imgID,'category_id':catID,"bbox":mask['bbox'], "score": mask['stability_score']}
        results.append(result)
    return labels
        
generated_labels = generate_labels(masks)
print("GENERATED LABELS")
pprint(generated_labels)
print("GROUND TRUTH LABELS")
pprint(ground_truth_labels)

# evaluate generated annotations


# running evaluation
# Serializing json
json_object = json.dumps(results, indent=4)
with open("../output/results.json", "w") as outfile:
    outfile.write(json_object)
cocoDt=cocoGt.loadRes("../output/results.json")
cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
cocoEval.params.imgIds  = [imgID]
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()