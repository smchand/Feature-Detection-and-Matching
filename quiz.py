import cv2 as cv
import matplotlib.pyplot as plt
import os

base_path = './Dataset/Data'

img = cv.imread('./Dataset/Target.png')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.equalizeHist(img)

dataset = []

for i in os.listdir(base_path):
    img_data = cv.imread(base_path + '/' + i)
    img_data = cv.cvtColor(img_data, cv.COLOR_BGR2GRAY)
    img_data = cv.equalizeHist(img_data)
    
    dataset.append(img_data)
    
sift = cv.SIFT_create()

target_kp, target_dsc = sift.detectAndCompute(img, None)

target_dsc = target_dsc.astype('f')

all_mask = []
total = 0
best_idx = -1
best_kp = None
best_match = None

for idx, i in enumerate(dataset):
    scene_kp, scene_dsc = sift.detectAndCompute(i, None)
    scene_dsc = scene_dsc.astype('f')
    
    index_param = dict(algorithm = 1)
    search_param = dict(check = 50)
    
    flann = cv.FlannBasedMatcher(index_param, search_param)
    matches = flann.knnMatch(scene_dsc, target_dsc, 2)
    
    scene_mask = [[0,0]] * len(matches)
    m_counter = 0

    for j, (m, n) in enumerate(matches):
        if m.distance < n.distance * 0.7:
            scene_mask[j] = [1,0]
            m_counter += 1
    
    all_mask.append(scene_mask)
    if m_counter >= total:
        total = m_counter
        best_idx = idx
        best_kp = scene_kp
        best_match = matches
        
        
res_img = cv.drawMatchesKnn(
        dataset[best_idx],
    best_kp,
    img,
    target_kp,
    best_match,
    None,
    matchColor = [0, 0, 255],
    singlePointColor = [255, 0, 0],
    matchesMask = all_mask[best_idx]
)

plt.imshow(res_img, cmap = 'gray')
plt.title('Best Match Result')
plt.show()
        
    
