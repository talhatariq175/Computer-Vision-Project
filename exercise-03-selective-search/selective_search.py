from __future__ import division
import numpy as np
from skimage.segmentation import felzenszwalb
from skimage.feature import local_binary_pattern
from skimage.color import rgb2hsv, rgb2lab

def generate_segments(im_orig, scale, sigma, min_size):
    mask = felzenszwalb(im_orig, scale=scale, sigma=sigma, min_size=min_size)
    im_masked = np.append(im_orig, mask[:, :, np.newaxis], axis=2)
    return im_masked

def calc_hsv_hist(pixels):
    BINS = 25
    hist = []
    for ch in range(3):
        h, _ = np.histogram(pixels[:, ch], bins=BINS, range=(0, 1), density=True)
        hist.extend(h)
    return np.array(hist)

def calc_lab_hist(pixels):
    BINS = 25
    hist = []
    h, _ = np.histogram(pixels[:, 0], bins=BINS, range=(0, 100), density=True)
    hist.extend(h)
    h, _ = np.histogram(pixels[:, 1], bins=BINS, range=(-128, 127), density=True)
    hist.extend(h)
    h, _ = np.histogram(pixels[:, 2], bins=BINS, range=(-128, 127), density=True)
    hist.extend(h)
    return np.array(hist)

def calc_texture_gradient(img):
    ret = np.zeros_like(img, dtype=float)
    for ch in range(3):
        ret[:, :, ch] = local_binary_pattern(img[:, :, ch], P=8, R=1, method='uniform')
    return ret

def calc_texture_hist(pixels):
    BINS = 10
    hist = []
    for ch in range(3):
        h, _ = np.histogram(pixels[:, ch], bins=BINS, range=(0, 255), density=True)
        hist.extend(h)
    if np.sum(hist) == 0: return hist
    return hist / np.sum(hist)

def sim_texture(r1, r2):
    return np.sum(np.minimum(r1['hist_t'], r2['hist_t']))

def sim_size(r1, r2, imsize):
    return 1.0 - (r1['size'] + r2['size']) / imsize

def sim_fill(r1, r2, imsize):
    bbsize = (max(r1['max_x'], r2['max_x']) - min(r1['min_x'], r2['min_x'])) * \
             (max(r1['max_y'], r2['max_y']) - min(r1['min_y'], r2['min_y']))
    return 1.0 - (bbsize - r1['size'] - r2['size']) / imsize

def calc_sim(r1, r2, imsize):
    sim_hsv = np.sum(np.minimum(r1['hist_c_hsv'], r2['hist_c_hsv']))
    sim_lab = np.sum(np.minimum(r1['hist_c_lab'], r2['hist_c_lab']))
    final_sim_color = (sim_hsv + sim_lab) / 2
    return (final_sim_color + sim_texture(r1, r2)
            + sim_size(r1, r2, imsize) + sim_fill(r1, r2, imsize))

def extract_regions(img):
    hsv = rgb2hsv(img[:, :, :3])
    lab = rgb2lab(img[:, :, :3])
    texture_grad = calc_texture_gradient(img[:, :, :3])
    labels = img[:, :, 3].astype(int)
    regions = {}
    for y in range(labels.shape[0]):
        for x in range(labels.shape[1]):
            l = labels[y, x]
            if l not in regions: regions[l] = {'min_x': x, 'min_y': y, 'max_x': x, 'max_y': y, 'size': 0}
            regions[l]['min_x'] = min(regions[l]['min_x'], x)
            regions[l]['min_y'] = min(regions[l]['min_y'], y)
            regions[l]['max_x'] = max(regions[l]['max_x'], x)
            regions[l]['max_y'] = max(regions[l]['max_y'], y)
            regions[l]['size'] += 1
    for k, r in regions.items():
        mask = (labels == k)
        regions[k]['hist_c_hsv'] = calc_hsv_hist(hsv[mask])
        regions[k]['hist_c_lab'] = calc_lab_hist(lab[mask])
        regions[k]['hist_t'] = calc_texture_hist(texture_grad[mask])
        regions[k]['rect'] = (r['min_x'], r['min_y'], r['max_x'] - r['min_x'], r['max_y'] - r['min_y'])
    return regions

def extract_neighbours(regions):
    def intersect(a, b):
        return not (a['max_x'] < b['min_x'] or b['max_x'] < a['min_x'] or
                    a['max_y'] < b['min_y'] or b['max_y'] < a['min_y'])
    keys = list(regions.keys())
    neighbours = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if intersect(regions[keys[i]], regions[keys[j]]):
                neighbours.append((keys[i], keys[j]))
    return neighbours

def merge_regions(r1, r2):
    new = {'min_x': min(r1['min_x'], r2['min_x']), 'min_y': min(r1['min_y'], r2['min_y']),
           'max_x': max(r1['max_x'], r2['max_x']), 'max_y': max(r1['max_y'], r2['max_y']),
           'size': r1['size'] + r2['size']}
    new['rect'] = (new['min_x'], new['min_y'], new['max_x'] - new['min_x'], new['max_y'] - new['min_y'])
    new['hist_c_hsv'] = (r1['hist_c_hsv'] * r1['size'] + r2['hist_c_hsv'] * r2['size']) / new['size']
    new['hist_c_lab'] = (r1['hist_c_lab'] * r1['size'] + r2['hist_c_lab'] * r2['size']) / new['size']
    new['hist_t'] = (r1['hist_t'] * r1['size'] + r2['hist_t'] * r2['size']) / new['size']
    return new

def selective_search(image_orig, scale=100, sigma=0.8, min_size=50, max_merges=2000):
    assert image_orig.shape[2] == 3, "Input image must be a 3-channel RGB image."
    imsize = image_orig.shape[0] * image_orig.shape[1]
    image_masked = generate_segments(image_orig, scale, sigma, min_size)
    R = extract_regions(image_masked)
    neighbours = extract_neighbours(R)
    S = {}
    for (i, j) in neighbours:
        S[tuple(sorted((i, j)))] = calc_sim(R[i], R[j], imsize)
    
    merge_count = 0
    while S and merge_count < max_merges:
        i, j = max(S, key=S.get)
        t = max(R.keys()) + 1
        R[t] = merge_regions(R[i], R[j])
        keys_to_delete = [key for key in S if i in key or j in key]
        for key in keys_to_delete:
            del S[key]
        for k in R:
            if k != t:
                if not (R[k]['max_x'] < R[t]['min_x'] or R[t]['max_x'] < R[k]['min_x'] or
                        R[k]['max_y'] < R[t]['min_y'] or R[t]['max_y'] < R[k]['min_y']):
                    S[tuple(sorted((k, t)))] = calc_sim(R[k], R[t], imsize)
        merge_count += 1
    regions = [{'rect': r['rect'], 'size': r['size']} for r in R.values()]
    return image_masked, regions