import os
import gzip
import pickle
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize
from tqdm import tqdm
import cv2
import parmap

# --------------- Helper Functions ---------------
def loadLabels(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
    return {line.split()[0]: int(line.split()[1]) for line in lines}

def loadFeature(fname):
    with gzip.open(fname, 'rb') as f:
        return pickle.load(f, encoding='latin1')

def get_feature_path(image_id, split):
    return os.path.join("icdar17_local_features", split, f"{image_id}_SIFT_patch_pr.pkl.gz")

# --------------- Your Implementation Below ---------------
def loadRandomDescriptors(train_keys, sample_size=500_000):
    descriptors = []
    for key in tqdm(train_keys):
        path = get_feature_path(key, "train")
        desc = loadFeature(path)
        descriptors.append(desc)
    descriptors = np.vstack(descriptors)
    idx = np.random.choice(descriptors.shape[0], sample_size, replace=False)
    return descriptors[idx]

def computeCodebook(descriptors, num_clusters=100):
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(descriptors)
    return kmeans

def vlad(descriptors, codebook):
    k = codebook.n_clusters
    d = descriptors.shape[1]
    centers = codebook.cluster_centers_
    vlad_vec = np.zeros((k, d), dtype=np.float32)

    matcher = cv2.BFMatcher()
    matches = matcher.match(descriptors.astype(np.float32), centers.astype(np.float32))

    for i, match in enumerate(matches):
        cid = match.trainIdx
        residual = descriptors[i] - centers[cid]
        vlad_vec[cid] += residual

    return vlad_vec.flatten()

def vlad_normalize(vlad_vec):
    vlad_vec = np.sign(vlad_vec) * np.sqrt(np.abs(vlad_vec))
    vlad_vec = normalize(vlad_vec.reshape(1, -1), norm='l2')[0]
    return vlad_vec

def encode_single_image(key, codebook, split):
    desc = loadFeature(get_feature_path(key, split))
    v = vlad(desc, codebook)
    return vlad_normalize(v)

def encode_single_image_wrapper(args):
    key, codebook, split = args
    return encode_single_image(key, codebook, split)

def computeEncodings(keys, codebook, split):
    args = [(k, codebook, split) for k in keys]
    return np.array(parmap.map(encode_single_image_wrapper, args, pm_pbar=True))

def evaluateMap(encodings, labels):
    dists = pairwise_distances(encodings, metric='euclidean')
    labels = np.array(labels)
    mAPs = []
    for i in range(len(labels)):
        true = (labels == labels[i]).astype(int)
        score = -dists[i]
        mAPs.append(average_precision_score(true, score))
    return np.mean(mAPs)

def train_exemplar_svm(args):
    test_vec, train_encodings = args
    X = np.vstack([test_vec.reshape(1, -1), train_encodings])
    y = np.array([1] + [0] * len(train_encodings))
    clf = LinearSVC(C=1000, class_weight='balanced')
    clf.fit(X, y)
    w = clf.coef_[0]
    return normalize(w.reshape(1, -1), norm='l2')[0]

def exemplarSVM(train_encodings, test_encodings):
    args = [(v, train_encodings) for v in test_encodings]
    return np.array(parmap.map(train_exemplar_svm, args, pm_pbar=True))

# --------------- Main ---------------
def main():
    print("Loading labels...")
    train_labels = loadLabels("icdar17_labels_train.txt")
    test_labels = loadLabels("icdar17_labels_test.txt")
    train_keys = list(train_labels.keys())
    test_keys = list(test_labels.keys())

    print("Sampling descriptors...")
    sample_descs = loadRandomDescriptors(train_keys)

    for k in [32, 64, 100, 128]:
        print(f"\n---- Testing codebook size: k = {k} ----")
        codebook = computeCodebook(sample_descs, num_clusters=k)

        print("Computing VLAD encodings (train)...")
        train_encodings = computeEncodings(train_keys, codebook, "train")
        print("Computing VLAD encodings (test)...")
        test_encodings = computeEncodings(test_keys, codebook, "test")

        all_encodings = np.vstack([train_encodings, test_encodings])
        all_labels = np.array([train_labels[k] for k in train_keys] + [test_labels[k] for k in test_keys])

        print("Evaluating VLAD mAP...")
        vlad_map = evaluateMap(all_encodings, all_labels)
        print(f"VLAD mAP (k={k}): {vlad_map:.4f}")

        print("Running Exemplar SVM...")
        svm_test_encodings = exemplarSVM(train_encodings, test_encodings)
        svm_all_encodings = np.vstack([train_encodings, svm_test_encodings])
        svm_map = evaluateMap(svm_all_encodings, all_labels)
        print(f"Exemplar SVM mAP (k={k}): {svm_map:.4f}")

if __name__ == '__main__':
    main()
