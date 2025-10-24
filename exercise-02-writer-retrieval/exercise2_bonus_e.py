# exercise2_bonus_e.py
import os
import gzip
import pickle
import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.metrics import average_precision_score, pairwise_distances
from sklearn.preprocessing import normalize
from tqdm import tqdm
import parmap

# -------------------------- Basic Loading --------------------------
def loadLabels(fname):
    with open(fname, 'r') as f:
        return {line.split()[0]: int(line.split()[1]) for line in f}

def loadFeaturePkl(fname):
    with gzip.open(fname, 'rb') as f:
        return pickle.load(f, encoding='latin1')

# -------------------------- RootSIFT Feature Extraction --------------------------
def computeDescs(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((0, 128), dtype=np.float32)

    sift = cv2.SIFT_create()
    keypoints = sift.detect(img, None)
    for kp in keypoints:
        kp.angle = 0  # Set all angles to zero
    keypoints, descriptors = sift.compute(img, keypoints)

    if descriptors is None:
        return np.zeros((0, 128), dtype=np.float32)

    # RootSIFT (Hellinger normalization)
    descriptors /= (np.linalg.norm(descriptors, ord=1, axis=1, keepdims=True) + 1e-7)
    descriptors = np.sign(descriptors) * np.sqrt(np.abs(descriptors))
    return descriptors.astype(np.float32)

# -------------------------- VLAD --------------------------
def vlad(descriptors, codebook):
    k = codebook.n_clusters
    d = descriptors.shape[1]
    vlad_vec = np.zeros((k, d), dtype=np.float32)
    matcher = cv2.BFMatcher()
    matches = matcher.match(descriptors.astype(np.float32), codebook.cluster_centers_.astype(np.float32))
    for i, match in enumerate(matches):
        cid = match.trainIdx
        residual = descriptors[i] - codebook.cluster_centers_[cid]
        vlad_vec[cid] += residual
    return vlad_vec.flatten()

def vlad_normalize(v):
    v = np.sign(v) * np.sqrt(np.abs(v))
    return normalize(v.reshape(1, -1), norm='l2')[0]

# -------------------------- Encode Wrapper for Multiprocessing --------------------------
def encode_wrapper(args):
    key, codebook, mode, split = args
    if mode == 'precomputed':
        fname = os.path.join("icdar17_local_features", split, f"{key}_SIFT_patch_pr.pkl.gz")
        descriptors = loadFeaturePkl(fname)
    else:
        fname = os.path.join("images", split, f"{key}.jpg")
        descriptors = computeDescs(fname)
    if descriptors.shape[0] == 0:
        return np.zeros((codebook.n_clusters * 128,), dtype=np.float32)
    return vlad_normalize(vlad(descriptors, codebook))

# -------------------------- SVM Wrapper for Multiprocessing --------------------------
def train_svm_wrapper(test_vec, train_enc):
    X = np.vstack([test_vec, train_enc])
    y = np.array([1] + [0]*len(train_enc))
    clf = LinearSVC(C=1000, class_weight='balanced')
    clf.fit(X, y)
    return normalize(clf.coef_.reshape(1, -1), norm='l2')[0]

# -------------------------- Pipelines --------------------------
def computeEncodings(keys, codebook, mode='precomputed', split='train'):
    args = [(k, codebook, mode, split) for k in keys]
    return np.array(parmap.map(encode_wrapper, args, pm_pbar=True))

def loadRandomDescriptors(keys, mode='precomputed', split='train', sample_size=500_000):
    all_descs = []
    for key in tqdm(keys):
        if mode == 'precomputed':
            fname = os.path.join("icdar17_local_features", split, f"{key}_SIFT_patch_pr.pkl.gz")
            desc = loadFeaturePkl(fname)
        else:
            fname = os.path.join("images", split, f"{key}.jpg")
            desc = computeDescs(fname)
        if desc.shape[0] > 0:
            all_descs.append(desc)
    all_descs = np.vstack(all_descs)
    idx = np.random.choice(all_descs.shape[0], sample_size, replace=False)
    return all_descs[idx]

def exemplarSVM(train_enc, test_enc):
    args = [(vec, train_enc) for vec in test_enc]
    return np.array(parmap.starmap(train_svm_wrapper, args, pm_pbar=True))

def evaluateMap(encodings, labels):
    dists = pairwise_distances(encodings, metric='euclidean')
    aps = []
    for i in range(len(labels)):
        true = (labels == labels[i]).astype(int)
        score = -dists[i]
        aps.append(average_precision_score(true, score))
    return np.mean(aps)

# -------------------------- Main --------------------------
def main():
    train_labels = loadLabels("icdar17_labels_train.txt")
    test_labels = loadLabels("icdar17_labels_test.txt")
    train_keys = list(train_labels.keys())
    test_keys = list(test_labels.keys())
    all_labels = np.array([train_labels[k] for k in train_keys] + [test_labels[k] for k in test_keys])

    for mode in ['precomputed', 'rootsift']:
        print(f"\n=== Running mode: {mode} ===")
        sampled_descs = loadRandomDescriptors(train_keys, mode=mode)
        codebook = MiniBatchKMeans(n_clusters=100, random_state=42).fit(sampled_descs)

        train_enc = computeEncodings(train_keys, codebook, mode=mode, split='train')
        test_enc = computeEncodings(test_keys, codebook, mode=mode, split='test')
        all_enc = np.vstack([train_enc, test_enc])

        print("Evaluating VLAD mAP...")
        map_vlad = evaluateMap(all_enc, all_labels)
        print(f"{mode.upper()} VLAD mAP: {map_vlad:.4f}")

        print("Training exemplar SVMs...")
        svm_test_enc = exemplarSVM(train_enc, test_enc)
        all_enc_svm = np.vstack([train_enc, svm_test_enc])
        map_svm = evaluateMap(all_enc_svm, all_labels)
        print(f"{mode.upper()} Exemplar SVM mAP: {map_svm:.4f}")

if __name__ == '__main__':
    main()
