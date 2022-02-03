import cv2
import numpy as np
import matplotlib.pyplot as plt

class Spectral_img:
    def __init__(self, paths, bands = ["G", "R", "IR", "NIR"]):

        self.min_threshold = 20
        self.offsets = [10,-10]

        self.bands = bands
        self.paths = paths
        if (len(paths) != len(bands)): 
            print ("Paths and bands with different sizes!")     
            return

        self.imgs = dict()
        self.warped_imgs = dict()
        self.croped_imgs = dict()
        self.draw_matches = []

        self.read_imgs()
        
        offset = self.imgs[self.bands[0]].shape[0]//3
        self.imgs[self.bands[0]] = self.imgs[self.bands[0]][offset:, ]
        self.imgs[self.bands[1]] = self.imgs[self.bands[1]][offset:,]
        self.imgs[self.bands[2]] = self.imgs[self.bands[2]][:-offset,]
        self.imgs[self.bands[3]] = self.imgs[self.bands[3]][:-offset]

        self.align()
        
        return
    
    def read_imgs(self):
        """ Reads the images and store in the specific band input order"""
        for index in range(len(self.bands)):
            new_img = cv2.imread(self.paths[index], 0)
            if new_img is not None:
                self.imgs[self.bands[index]] = new_img
        return

    def equalize_smooth(self, img):
        """ Histogram Normalization and gaussian blur of input image """
        img_blur = cv2.GaussianBlur(img, (3,3),0)
    #     img_blur = cv2.medianBlur(img, 5)
        img_norm = cv2.equalizeHist(img_blur)
        ret, img_norm = cv2.threshold(img_norm, self.min_threshold, 255, cv2.THRESH_TOZERO)
        return img

    def warp_img(self, img_original, base, draw_matches = False):
        #if (img_original.all() == base.all()): return img_original

        if (len(base.shape) > 2):
            gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        else:
            gray = base
            
        img =  self.equalize_smooth(img_original)
        gray = self.equalize_smooth(base)   

        # Seleção dos descriptors:
        # descriptor = cv2.SIFT.create()
        descriptor = cv2.xfeatures2d.SURF_create() #O SURF não é um algoritmo open source e só é incluido no opencv-contrib-python use com cautela.
        matcher = cv2.FlannBasedMatcher()

        # get features from images
        kps_img, desc_img = descriptor.detectAndCompute(img, mask=None)
        kps_base, desc_base = descriptor.detectAndCompute(gray, mask=None)

        # find the corresponding point pairs
        if (desc_img is not None and desc_base is not None and len(desc_img) >=2 and len(desc_base) >= 2):
            rawMatch = matcher.knnMatch(desc_base, desc_img, k=2)

        matches = []
        # ensure the distance is within a certain ratio of each other (i.e. Lowe's ratio test)
        ratio = 0.75
        for m in rawMatch:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
                # print("Train index: ", m[0].trainIdx, "\n Query index", m[0].queryIdx)
                # print()
                
        if draw_matches:
            # Apply ratio test
            good = []
            for m,n in rawMatch:
                if m.distance < 0.75*n.distance:
                    good.append([m])
            self.draw_matches.append(cv2.drawMatchesKnn(img,kps_img,gray,kps_base,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))
            

        # convert keypoints to points
        pts_img, pts_base = [], []
        for id_img, id_base in matches:
            pts_img.append(kps_img[id_img].pt)
            pts_base.append(kps_base[id_base].pt)
        pts_img = np.array(pts_img, dtype=np.float32)
        pts_base = np.array(pts_base, dtype=np.float32)

        # compute homography
        if len(matches) > 4:
    #         H, status = cv2.findHomography(pts_img, pts_base, cv2.RANSAC)
            H, _ = cv2.estimateAffine2D(pts_img, pts_base)
    #         print(H)
    #         H = np.vstack((H, [0, 0, 1]))
            # print("H = ", H)
        warped = cv2.warpAffine(img_original, H, (base.shape[1], base.shape[0]))
    #     warped = cv2.warpPerspective(img_original, H, (base.shape[1], base.shape[0]))
        return warped

    def align(self, base = None):
        if (base == None): base =  self.bands[3]
        self.draw_matches = [] #Reset the matches drawings

        for band in self.bands:
                self.warped_imgs[band] = self.warp_img(self.imgs[band], self.imgs[base], True)
                if (band == self.bands[0]):
                     x, y, w, h  = cv2.boundingRect(cv2.findNonZero(self.warped_imgs[self.bands[0]]))
                     x, y, w, h = x+self.offsets[0], y+self.offsets[0], w+self.offsets[1], h+self.offsets[1]
                self.croped_imgs[band] = self.warped_imgs[band][y:y+h, x:x+w]

        return

    def visualize(self):

        if len(self.draw_matches) == 4: rows = 4 
        else: rows = 3

        fig, axs = plt.subplots(4,rows, constrained_layout = True, figsize = (4*5, 4*5)) 
        for i in range(4):
            axs[0][i].imshow(s.imgs[self.bands[i]])
            axs[1][i].imshow(s.warped_imgs[self.bands[i]])
            axs[2][i].imshow(s.croped_imgs[self.bands[i]])
            if len(self.draw_matches) == 4: axs[3][i].imshow(s.draw_matches[i])
        [axi.set_axis_off() for axi in axs.ravel()]
        plt.show()
        return

    def false_color(self, channels):
        merged = cv2.merge(channels)
        plt.imshow(merged)
        plt.show()
        return 