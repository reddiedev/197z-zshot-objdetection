from pycocotools.coco import COCO
import numpy as np

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
print(img)
