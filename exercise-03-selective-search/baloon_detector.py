import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.io import imread
from sklearn.svm import SVC
from sklearn.utils import shuffle
from tqdm import tqdm
import torch
from torchvision import models, transforms
from PIL import Image
import albumentations as A

from selective_search import selective_search

class FeatureExtractor:
    def __init__(self):
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model = torch.nn.Sequential(*list(model.children())[:-1])
        self.model.eval()
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def extract(self, crop):
        if crop.shape[0] == 0 or crop.shape[1] == 0: return None
        crop_pil = Image.fromarray(crop).resize((224, 224))
        input_tensor = self.transform(crop_pil).unsqueeze(0)
        with torch.no_grad(): features = self.model(input_tensor).squeeze().numpy()
        return features.flatten()

def non_max_suppression(boxes, scores, iou_threshold):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
    areas, order = (x2 - x1) * (y2 - y1), scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        w, h = np.maximum(0.0, xx2 - xx1), np.maximum(0.0, yy2 - yy1)
        iou = (w * h) / (areas[i] + areas[order[1:]] - (w * h) + 1e-7)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[0] + boxA[2], boxB[0] + boxB[2]), min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea, boxBArea = boxA[2] * boxA[3], boxB[2] * boxB[3]
    unionArea = float(boxAArea + boxBArea - interArea)
    return interArea / unionArea if unionArea > 0 else 0

def generate_proposals(image_dir, save_path):
    proposals = {}
    for fname in tqdm(os.listdir(image_dir), desc=f"Proposals for {os.path.basename(image_dir)}"):
        if not fname.lower().endswith(('.jpg', '.jpeg')): continue
        fpath = os.path.join(image_dir, fname)
        img = imread(fpath)
        if img.shape[2] == 4: img = img[:,:,:3]
        _, props = selective_search(img)
        proposals[fpath] = [p['rect'] for p in props]
    with open(save_path, 'wb') as f: pickle.dump(proposals, f)
    return proposals

def run_pipeline():
    # --- 0. FINAL PARAMETERS ---
    iou_positive_threshold = 0.6
    confidence_threshold = 0.6
    nms_iou_threshold = 0.4

    script_dir = os.path.dirname(__file__) if '__file__' in locals() else '.'
    data_dir = os.path.join(script_dir, "data", "balloon_dataset")
    train_dir, valid_dir, test_dir = [os.path.join(data_dir, s) for s in ['train', 'valid', 'test']]
    train_ann, valid_ann = [os.path.join(d, "_annotations.coco.json") for d in [train_dir, valid_dir]]
    
    model_path = "svm_model_final.pkl"
    combined_props_path = 'train_valid_proposals.pkl'
    test_props_path = 'test_proposals.pkl'
    
    aug_pipeline = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Rotate(limit=15, p=0.3),
        A.RandomScale(scale_limit=0.1, p=0.3)
    ])
    extractor = FeatureExtractor()

    # --- 1. TRAIN FINAL MODEL (on TRAIN + VALID) ---
    if os.path.exists(model_path):
        print("✅ Loading final SVM model...")
        with open(model_path, 'rb') as f: clf = pickle.load(f)
    else:
        if os.path.exists(combined_props_path):
            print("✅ Loading existing combined training/validation proposals...")
            with open(combined_props_path, 'rb') as f: train_props = pickle.load(f)
        else:
            print("⏳ Generating training proposals...")
            train_part = generate_proposals(train_dir, 'train_props_part.pkl')
            print("⏳ Generating validation proposals...")
            valid_part = generate_proposals(valid_dir, 'valid_props_part.pkl')
            train_props = {**train_part, **valid_part}
            with open(combined_props_path, 'wb') as f: pickle.dump(train_props, f)
            print("✅ Combined and saved training/validation proposals.")

        print("⏳ Training final SVM model on combined train/valid data...")
        with open(train_ann) as f: train_anns_json = json.load(f)
        with open(valid_ann) as f: valid_anns_json = json.load(f)
        gt_boxes = {}
        def process_annotations(anns_json, data_dir_path):
            img_map = {img["id"]: img["file_name"] for img in anns_json["images"]}
            for a in anns_json["annotations"]:
                img_path = os.path.join(data_dir_path, img_map[a["image_id"]])
                gt_boxes.setdefault(img_path, []).append([int(v) for v in a["bbox"]])
        process_annotations(train_anns_json, train_dir)
        process_annotations(valid_anns_json, valid_dir)
        
        pos, neg = [], []
        for img_path, boxes in train_props.items():
            gts = gt_boxes.get(img_path, [])
            for box in boxes:
                iou = max([compute_iou(box, gt) for gt in gts] or [0])
                if iou > iou_positive_threshold: pos.append((img_path, box))
                elif iou < 0.2: neg.append((img_path, box))
        print(f"   [INFO] Final training data: {len(pos)} positives, {len(neg)} negatives")
        
        if len(pos) == 0:
            print("\n❌ ERROR: No positive training samples found.")
            return

        X, y = [], []
        for path, box in tqdm(pos + neg, desc="   Extracting features"):
            img = imread(path)
            x, y_coord, w, h = box
            crop = img[y_coord:y_coord+h, x:x+w]
            if (path, box) in pos: crop = aug_pipeline(image=crop)['image']
            feat = extractor.extract(crop)
            if feat is not None: X.append(feat); y.append(1 if (path, box) in pos else 0)
        
        X, y = shuffle(X, y)
        clf = SVC(probability=True, kernel='rbf', C=1.0, class_weight='balanced')
        clf.fit(X, y)
        with open(model_path, 'wb') as f: pickle.dump(clf, f)
        print("✅ Final model trained and saved.")

    # --- 2. FINAL INFERENCE (on TEST set) ---
    if os.path.exists(test_props_path):
        print("\n✅ Loading existing test proposals...")
        with open(test_props_path, 'rb') as f: test_props = pickle.load(f)
    else:
        print("\n⏳ Generating and saving test proposals...")
        test_props = generate_proposals(test_dir, test_props_path)
    
    print(" Starting final inference...")
    output_dir = "output_final"
    os.makedirs(output_dir, exist_ok=True)
    for test_img_path, boxes in test_props.items():
        img = imread(test_img_path)
        if img.shape[2] == 4: img = img[:,:,:3]
        
        high_conf_boxes, scores = [], []
        for box in boxes:
            feat = extractor.extract(img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]])
            if feat is not None:
                prob = clf.predict_proba([feat])[0][1]
                if prob > confidence_threshold: high_conf_boxes.append(box); scores.append(prob)
        
        final_boxes = []
        if high_conf_boxes:
            boxes_np, scores_np = np.array(high_conf_boxes), np.array(scores)
            keep_indices = non_max_suppression(boxes_np, scores_np, nms_iou_threshold)
            final_boxes = boxes_np[keep_indices]
        
        fig, ax = plt.subplots(1); ax.imshow(img); ax.axis('off')
        for box in final_boxes: ax.add_patch(patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=2, edgecolor='lime', facecolor='none'))
        output_path = os.path.join(output_dir, os.path.basename(test_img_path))
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=200); plt.show(); plt.close(fig)
        print(f"   [RESULT] Found {len(final_boxes)} balloons in {os.path.basename(test_img_path)}. Saved to '{output_dir}'.")

if __name__ == "__main__":
    run_pipeline()