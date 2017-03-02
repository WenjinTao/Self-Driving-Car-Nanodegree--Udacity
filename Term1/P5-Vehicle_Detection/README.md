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
[image8]: ./output_images/car_histograms.png
[image9]: ./output_images/notcar_histograms.png
[image10]: ./output_images/color_space_rgb.png
[image11]: ./output_images/color_space_ycrcb.png
[image12]: ./output_images/image0000.png
[image13]: ./output_images/extra1.png
[image14]: ./output_images/color_space_rgb_notcar.png
[image15]: ./output_images/color_space_ycrcb_notcar.png
[image16]: ./output_images/hog_03.png
[image17]: ./output_images/hog_04.png

## 2. Feature Extraction

***Color spatial features***, ***histogram features*** and ***Histogram of  HOG features*** were used in this project, they were combined together to generate the feature vector for the classifier. 
I started by reading in all the `vehicle` and `non-vehicle` images.  Here is a few examples of the `vehicle` and `non-vehicle` classes:

![alt text][image1]


#### 2.1. Color Histogram Features 

Color histogram features extraction is looking at histograms of pixel intensity at each channel (color histograms) as features. The features from three channels were concatenated to form a feature vector. The following two images show the color histogram features of a car image and a background image, respectively.

![alt text][image8]

![alt text][image9]

#### 2.2. Color Spaces

It may be easier to locate clusters of colors that correspond to the cars/notcars in other color spaces rather than the original 'RGB' space. Here shows an example of plotting the pixels in different color space, from where we can notice the different clustering effects.

| Car/not (64x64) |     RGB      |    YCrCb     |
| :-------------: | :----------: | :----------: |
|  ![][image12]   | ![][image10] | ![][image11] |
|  ![][image13]   | ![][image14] | ![][image15] |

#### 2.3. Spatial Binning of Color

Spatial binning can be performed to reduce the complexity and retain enough information at the same time, which will improve the efficiency in searching for cars. `cv2.resize()` can be used to scale down the resolution of an image and then `ravel()` can be used to create the feature vector.

#### 2.4. Extracting HOG features from the training images

The code for this step is contained in the 4th code cell of the IPython notebook.  


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is several examples using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` , `cells_per_block=(2, 2)`, `hog_channel=0`, `spatial_size=(16, 16)` and `hist_bins=16`:

![alt text][image2]

![alt text][image3]

![][image16]

![][image17]

#### 2.5 Choosing parameters

After some trial and error, I chose the parameters as follows because of they can achieve good classifying results and acceptable training efficiency:

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



##3. Training a classifier using SVM

Data normalization was implemented before I fed the SVM classifier. This was done by using:

```python
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)
```

Then the data was spited into training and testing data using `train_test_split( )`.  20% of the data was reserved for testing.

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

Finally, the test accuracy of this SVM =  **0.9938**, which I think is a good starting point for implementing the vehicle detection pipeline.



##4. Sliding Window Search

####4.1 Search windows

I chose two scales (1.5 and 2) to resize the original image and window size of 64x64, to implement the sliding window search (i.e. window size of 96x96 and 128x128). The two kinds of scales and corresponding windows are shown in the following two images.

![alt text][image4]

![alt text][image5]

####4.2 Test the pipeline on a single image

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here is an example image:


![alt text][image7]

---

## 5. Video Implementation

####5.1 False positives filtering & Bounding boxes overlapping 

```python
# Define a class to record the heatmap of each frame
class Heatmaps():
    def __init__(self):
        self.heatmaps = []
```

From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions on each frame. I used the above `Heatmaps()` class to record the heatmap of each frame for further averaging. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here are six frames and their detected results, the blue rectangles show the positive detections, the green rectangles show the assuming vehicle positions. Their corresponding heatmaps are shown below as well.

![alt text][image6]



####5.2 Test the pipeline on videos

The pipeline was implemented on 'test_video' and 'project_video'. It performed reasonably well on both of the videos. The bounding boxes are relatively stable in most of the video frames. One false postive occurs at the beginning of 'project_video'. 

The processed video can be found here:

|     Videos      |                  Links                   |
| :-------------: | :--------------------------------------: |
|  'test_video'   | [![](https://img.youtube.com/vi/RASZZ41Okfc/0.jpg)](https://youtu.be/RASZZ41Okfc) |
| 'project_video' | [![](https://img.youtube.com/vi/qlcGfB6M48Y/0.jpg)](https://youtu.be/qlcGfB6M48Y) |



---

##6. Discussion

During the implementation of this project, there are several aspects that I'm still considering how to improve.

- **Computational Efficiency**: HOG features extraction is the most time consuming part in the whole pipeline, which might be the bottleneck if we deploy the pipeline for real-time vehicle detection. So how to improve the computational efficiency is my further direction.
- **False Positive** :  How to reject false positive is another critical problem. I find that most of the false positive detections occur at the lane's region. So my idea is building a color filter for lane's region. Before running the car detection routine, filtering the lane's pixels away, which might be helpful for rejecting false positive. It also can be used to accurately define the ROI for searching windows, which will further improve the searching efficiency.

----

## 7. References

| Topic                          | Links                                    |
| ------------------------------ | ---------------------------------------- |
| Histogram of oriented gradient | [1](http://www.learnopencv.com/histogram-of-oriented-gradients/), [2](http://www.pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/) |
| Classifiers                    | [Comparison](https://www.quora.com/What-are-the-advantages-of-different-classification-algorithms), [SVM](https://pdfs.semanticscholar.org/fb6b/a3944cf1e534f665bc86075e0af2d2337eb9.pdf) |
| Confusion matrix               | [1](https://en.wikipedia.org/wiki/Confusion_matrix), [2](http://notmatthancock.github.io/2015/10/28/confusion-matrix.html), [3: in sklearn](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) |
| Sliding window algorithm       | [1](http://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/) |

