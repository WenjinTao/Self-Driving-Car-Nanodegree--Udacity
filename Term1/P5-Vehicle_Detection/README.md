# Vehicle Detection
---

## 1. Introduction

The goal of this project is to write a software pipeline to detect vehicles in a video from a front-facing camera on a car. The detailed goals / steps of this project are the following:

1. Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
2. Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
3. Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
4. Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
5. Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
6. Estimate a bounding box for vehicles detected.

[//]: # "Image References"
[image1]: ./output_images/03.png
[image2]: ./output_images/02.png
[image3]: ./output_images/04.png
[image4]: ./output_images/05.png
[image5]: ./output_images/06.png
[image6]: ./output_images/01.png
[image7]: ./output_images/07.png
## 2. Histogram of Oriented Gradients (HOG)

#### 2.1. Extracting HOG features from the training images

The code for this step is contained in the 4th code cell of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is a few examples of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=6`, `pixels_per_cell=(8, 8)` , `cells_per_block=(2, 2)`, `hog_channel=0`, `spatial_size=(16, 16)` and `hist_bins=16`:

![alt text][image2]

![alt text][image3]

#### 2.2 Choosing parameters.

I tried various combinations of parameters and choose them as follows:

```python
global color_space, orient, pix_per_cell, cell_per_block, hog_channel
global spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat, y_start_stop

### Setting parameters
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
```



####2.3 Training a classifier using SVM

The data was spited into training and testing data using `train_test_split( )`.  

```python
# Split up data into randomized training and test sets
# rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=89)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
```

Then a SVM model was implemented using `LinearSVC( )`.

```python
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
```

After the model was trained, the model parameters were save to disk using the code as follows.

```python
from sklearn.externals import joblib
joblib.dump(svc, 'model_svc.pkl')
print('Saved model to disk')
```



##3. Sliding Window Search

####3.1 Search windows

I chose two scales (1.5 and 2) to resize the original image and window size of 64x64, to implement the sliding window search. The two kinds of scales and corresponding windows are shown in the following two images.

![alt text][image4]

![alt text][image5]

####3.2 Test the pipeline on a single image

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here is an example image:


![alt text][image7]

---

## 4. Video Implementation

####4.1 False positives filtering & Bounding boxes overlapping 

```python
# Define a class to record the heatmap of each frame
class Heatmaps():
    def __init__(self):
        self.heatmaps = []
```

From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions on each frame. I used the above `Heatmaps()` class to record the heatmap of each frame for further averaging. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here are six frames and their detected results, the blue rectangles show the positive detections, the green rectangles show the assuming vehicle positions. Their corresponding heatmaps are shown below as well.

![alt text][image6]



####4.2 Test the pipeline on videos

The pipeline was implemented on 'test_video' and 'project_video'. It performed reasonably well on both of the videos. The bounding boxes are relatively stable in most of the video frames. One false postive occurs at the beginning of 'project_video'. 

The processed video can be found here:

|     Videos      |                  Links                   |
| :-------------: | :--------------------------------------: |
|  'test_video'   | [![](https://img.youtube.com/vi/RASZZ41Okfc/0.jpg)](https://youtu.be/RASZZ41Okfc) |
| 'project_video' | [![](https://img.youtube.com/vi/qlcGfB6M48Y/0.jpg)](https://youtu.be/qlcGfB6M48Y) |



---

##5. Discussion

During the implementation of this project, there are several aspects that I'm still considering how to improve.

- **Computational Efficiency**: HOG features extraction is the most time consuming part in the whole pipeline, which might be the bottleneck if we deploy the pipeline for real-time vehicle detection. So how to improve the computational efficiency is my further direction.
- **False Positive** :  How to reject false positive is another critical problem. I find that most of the false positive detections occur at the lane's region. So my idea is building a color filter for lane's region. Before running the car detection routine, filtering the lane's pixels away, which might be helpful for rejecting false positive. It also can be used to accurately define the ROI for searching windows, which will further improve the searching efficiency.