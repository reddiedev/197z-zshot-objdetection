# obtain link to image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import torch
import torchvision
import sys
from skimage import io
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
import open_clip
from PIL import Image
from pprint import pprint
import json
import shutil
import re

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

print("[setup]: determining CUDA support...")
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
print("--------------------------------------")

imageCount = 1
if len(sys.argv) > 1:
    imageCount = int(sys.argv[1])
print(f'[initialize]: running object identification for {imageCount} images')


# clean output directory
if not os.path.exists("../output"):
    os.mkdir(f"../output")
shutil.rmtree("../output/")
os.mkdir(f"../output/")


print("[0]: loading coco annotations and captions...")
dataDir = '../coco'
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
# initialize COCO api for instance annotations + COCO ground truth api
cocoGt = COCO(annFile)
annFile = '{}/annotations/captions_{}.json'.format(dataDir, dataType)
coco_caps = COCO(annFile)
print("--------------------------------------")

# get all category IDs
catIDs = cocoGt.getCatIds()
cats = cocoGt.loadCats(catIDs)
cocoCategories = [cat['name'] for cat in cats]
imgIDs = []

# get all captions
annIds = coco_caps.getAnnIds(imgIds=[], catIds=[])
anns = coco_caps.loadAnns(annIds)
coco_labels_words = []
for ann in anns:
    words = ann['caption'].split()
    for word in words:
        clean_word = re.sub(r'[\W_]', '', word.lower())
        coco_labels_words.append(clean_word)

coco_labels_words_values = list(set(coco_labels_words))

print("[1]: loading sam model")
# sam_vit_b_01ec64 | sam_vit_h_4b8939
sam_checkpoint = os.path.join("../checkpoints", "sam_vit_h_4b8939.pth")
model_type = "vit_h"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    points_per_batch=150,
    pred_iou_thresh=0.8,
    box_nms_thresh=0.5,
    stability_score_thresh=0.9,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=300,  # Requires open-cv to run post-processing
)

# pred_iou_thresh (Prediction IoU Threshold):
# This parameter determines the threshold for the Intersection over Union (IoU) score between predicted bounding boxes and ground truth boxes. It controls how well the predicted boxes need to overlap with the ground truth boxes to be considered valid. Higher values result in stricter overlap requirements.

# box_nms_thresh (Box Non-Maximum Suppression Threshold):
# This parameter controls the threshold for Non-Maximum Suppression (NMS) during post-processing. NMS is applied to remove redundant bounding boxes that cover the same object. A higher threshold allows more overlapping boxes to remain, while a lower threshold removes more duplicates.

# stability_score_thresh (Stability Score Threshold):
# This parameter is specific to Segment Anything's automatic mask generator and determines the threshold for the stability score. The stability score measures the consistency of a pixel being assigned to a particular object across multiple frames or segments. Higher values indicate greater stability, and lower values allow more dynamic changes.

# min_mask_region_area (Minimum Mask Region Area):
# The purpose of setting a minimum mask region area is to remove small, potentially noisy or irrelevant segments that may not be meaningful or useful for the specific segmentation task. By filtering out these smaller regions, you can focus on larger and more substantial objects in the scene, improving the overall quality and relevance of the generated masks.

print("--------------------------------------")

print("[2]: creating open clip model...")
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print("[2]: loading coco categories as labels...")
text = tokenizer(coco_labels_words_values)
text = text.to(device)

print("--------------------------------------")

print("[3]: generating results for images...")

coco_results = []
label_results = []

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
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))


def generate_labels(anns, imgID, catID):
    if len(anns) == 0:
        return
    length = len(anns)
    values = []
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    for i in range(length):
        mask = sorted_anns[i]
        x, y, w, h = mask['bbox']
        x, y, w, h = int(x), int(y), int(w), int(h)
        im = image[y:y+h, x:x+w]
        plt.figure(figsize=(20, 20))
        plt.imshow(im)
        plt.axis('off')
        plt.savefig(f"../output/{imgID}/mask-{i+1}.jpg",
                    bbox_inches='tight', pad_inches=0)
        plt.close()
        im = Image.open(f"../output/{imgID}/mask-{i+1}.jpg").convert("RGB")
        img = preprocess(im).unsqueeze(0)
        img = img.to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(img)
            text_features = model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @
                          text_features.T).softmax(dim=-1)

        text_prob = np.max(text_probs.cpu().numpy())
        index = np.argmax(text_probs.cpu().numpy())
        label = coco_labels_words_values[index]

        print(f"[{i+1}/{length}]:",  label, f"({text_prob*100:.2f}%)",)
        values.append(
            {"label": label, "area": mask["area"], "prob": text_prob})
        result = {'image_id': imgID, 'category_id': catID,
                  "bbox": mask['bbox'], "score": mask['predicted_iou']}
        coco_results.append(result)

    # generate top 5 labels according to label_accuracy and mask_area
    sorted_values = sorted(values, key=lambda x: x['area'] * x['prob'])
    payload = sorted_values[:5]
    labels = list(map(lambda d: d['label'], payload))
    return labels


for i in range(imageCount):
    # choose random category ID
    catID = catIDs[np.random.randint(0, len(catIDs))]
    coco_image_ids = cocoGt.getImgIds(catIds=[catID])
    imgID = coco_image_ids[np.random.randint(0, len(coco_image_ids))]
    imgIDs.append(imgID)
    print("--------------------------------------")
    img = cocoGt.loadImgs(imgID)[0]
    imgUrl = img['coco_url']
    print(f"[3.5]: ({i+1}/{imageCount}) processing {imgID} | {imgUrl}...")

    os.mkdir(f"../output/{imgID}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"({imgID}): loading captions...")
    annIds = coco_caps.getAnnIds(imgIds=imgID)
    anns = coco_caps.loadAnns(annIds)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    ground_truth_labels = []
    ground_truth = []
    for ann in anns:
        ground_truth_labels.append(ann['caption'])
        words = ann['caption'].split()
        for word in words:
            ground_truth.append(word.lower())
    ground_truth_values = list(set(ground_truth))

    print(f"({imgID}): saving source images...")
    image = io.imread(imgUrl)
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(f"../output/{imgID}/source.jpg",
                bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"({imgID}): generating masks...")
    # automatically generate masks using SAM
    masks = mask_generator.generate(image)
    print(f"({imgID}): generated {len(masks)} masks...")

    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.savefig(f"../output/{imgID}/generated-masks.jpg",
                bbox_inches='tight', pad_inches=0)
    plt.close()

    # for each mask image, annotate using open-clip
    print(f"({imgID}): generating labels...")
    generated_labels = generate_labels(masks, imgID, catID)
    print("GENERATED LABELS")
    pprint(generated_labels)
    print("GROUND TRUTH LABELS")
    pprint(ground_truth_labels)


print("[4]: evaluating image results...")
print("[4] mAPS SCORE")
json_object = json.dumps(coco_results, indent=4)
with open("../output/results.json", "w") as outfile:
    outfile.write(json_object)
cocoDt = cocoGt.loadRes("../output/results.json")
cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
cocoEval.params.imgIds = imgIDs
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

print("[4] Top-1 Accuracy SCORE")
print("[4] Top-5 Accuracy SCORE")
