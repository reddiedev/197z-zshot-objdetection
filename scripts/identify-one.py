# obtain link to image
from pycocotools.coco import COCO
import numpy as np
import torch
import torchvision
from skimage import io
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
import open_clip
import urllib 
from PIL import Image
import os
import glob

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
# initialize COCO api for instance annotations
coco = COCO(annFile)
# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
# print('COCO categories: \n{}\n'.format(' '.join(nms)))
# nms = set([cat['supercategory'] for cat in cats])
# print('COCO supercategories: \n{}'.format(' '.join(nms)))

# get all category IDs
catIds = coco.getCatIds()
# choose random category ID
catId = catIds[np.random.randint(0,len(catIds))]
# choose random image from category
imgIds = coco.getImgIds(catIds=[catId] )
imgID = imgIds[np.random.randint(0,len(imgIds))]
img = coco.loadImgs(imgID)[0]
imgUrl = img['coco_url']


print(f"[0]: generating labels for source image...")
annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
coco_caps=COCO(annFile)
annIds = coco_caps.getAnnIds(imgIds=imgID)
anns = coco_caps.loadAnns(annIds)
ground_truth = []
for ann in anns:
    words = ann['caption'].split()
    for word in words:
        ground_truth.append(word.lower())

ground_truth = list(set(ground_truth))

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
sam_checkpoint = os.path.join("../checkpoints", "sam_vit_b_01ec64.pth") # sam_vit_b_01ec64 | sam_vit_h_4b8939
model_type = "vit_b"
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
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')
tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print("[2]: loading imagenet 1k labels")
# load imagenet 1k labels and tokenize them
filename = "imagenet1000_labels.txt"
url = "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"

# Download the file if it does not exist
if not os.path.isfile(filename):
    urllib.request.urlretrieve(url, filename)

with open(filename) as f:
    idx2label = eval(f.read())

imagenet_labels = list(idx2label.values())
text = tokenizer(imagenet_labels)
text = text.to(device)

def generate_labels(anns):
    if len(anns) == 0:
        return
    length = len(anns)
    labels = []
    for i in range(length):
        mask = anns[i]
        x,y,w,h = mask['bbox']
        im = image[y:y+h, x:x+w]
        plt.figure(figsize=(20,20))
        plt.imshow(im)
        plt.axis('off')
        plt.savefig(f"../output/mask-{i}.jpg", bbox_inches='tight', pad_inches = 0)
        plt.close()
        ima = Image.open(f"../output/mask-{i}.jpg")
        processedImage = preprocess(ima).unsqueeze(0)
        processedImage = processedImage.to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(processedImage)
            text_features = model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)


        index = np.argmax(text_probs.cpu().numpy())
        label = imagenet_labels[index]
        if (label in nms):
            labels.append(label)
        print(f"[label{i}]: {label} = {label in nms}") 
    return labels
        
labels = generate_labels(masks)
print(labels)
print(ground_truth)

# evaluate generated annotations