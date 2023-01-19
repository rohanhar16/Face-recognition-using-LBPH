# Face-recognition-using-LBPH

# Summary

1. [Introduction to LBP](#introduction)
- [Step-by-Step](#step-by-step)  
- [Comparing Histograms](#comparing-histograms)  
- [Important Notes](#important-notes)
- [I/O](#io)  
- [Input](#input)  
- [Output](#output)
2. [Face recognition using LBPH](#face-recognition)
- [Dataset](#dataset)
- [Generate Labels for images](#labels)
- [Train model](#train)
- [Saving model](#save)
- [Load model, and test with Web-camera](#load)
3. [References](#references)

# Introduction

Local Binary Patterns (LBP) is a type of visual descriptor used for classification in computer vision. LBP was first described in 1994 and has since been found to be a powerful feature for texture classification. It has further been determined that when LBP is combined with the Histogram of oriented gradients (HOG) descriptor, it improves the detection performance considerably on some datasets.

As LBP is a visual descriptor it can also be used for face recognition tasks, as can be seen in the following Step-by-Step explanation.

## Step-by-Step

In this section, it is shown a step-by-step explanation of the LBPH algorithm:

1. First of all, we need to define the parameters (`radius`, `neighbors`, `grid x` and `grid y`) using the `Parameters` structure from the `lbph` package. Then we need to call the `Init` function passing the structure with the parameters. If we not set the parameters, it will use the default parameters as explained in the [Parameters](#parameters) section.
2. Secondly, we need to train the algorithm. To do that we just need to call the `Train` function passing a slice of images and a slice of labels by parameter. All images must have the same size. The labels are used as IDs for the images, so if you have more than one image of the same texture/subject, the labels should be the same.
3. The `Train` function will first check if all images have the same size. If at least one image has not the same size, the `Train` function will return an error and the algorithm will not be trained.
4. Then, the `Train` function will apply the basic LBP operation by changing each pixel based on its neighbors using a default radius defined by the user. The basic LBP operation can be seen in the following image (using `8` neighbors and radius equal to `1`):

![LBP operation](http://i.imgur.com/G4PqJPe.png)

5. After applying the LBP operation we extract the histograms of each image based on the number of grids (X and Y) passed by parameter. After extracting the histogram of each region, we concatenate all histograms and create a new one which will be used to represent the image.

![Histograms](http://i.imgur.com/3BGk130.png)

6. The images, labels, and histograms are stored in a data structure so we can compare all of it to a new image in the `Predict` function.
7. Now, the algorithm is already trained and we can Predict a new image.
8. To predict a new image we just need to call the `Predict` function passing the image as parameter. The `Predict` function will extract the histogram from the new image, compare it to the histograms stored in the data structure and return the label and distance corresponding to the closest histogram if no error has occurred. **Note**: It uses the [euclidean distance](#comparing-histograms) metric as the default metric to compare the histograms. The closer to zero is the distance, the greater is the confidence.

### Comparing Histograms

The LBPH package provides the following metrics to compare the histograms:

**Chi-Square** :

![Chi-Square](http://i.imgur.com/6CyngL9.gif)

**Euclidean Distance** :

![Euclidean Distance](http://i.imgur.com/6ll6hDU.gif)

**Normalized Euclidean Distance** :

![Normalized Euclidean Distance](http://i.imgur.com/6Wj2keg.gif)

**Absolute Value** :

![Absolute Value](http://i.imgur.com/27jXZ4V.gif)

The comparison metric can be chosen as explained in the [metrics](#metrics) section.

### Important Notes

The current LBPH implementation uses a fixed `radius` of `1` and a fixed number of `neighbors` equal to `8`. We still need to implement the usage of these parameters in the LBP package (feel free to contribute here). Related to the [issue 1](https://github.com/kelvins/lbph/issues/1).

## I/O

In this section, you will find a brief explanation about the input and output data of the algorithm.

### Input

All input images (for training and testing) must have the same size. Different of OpenCV, the images don't need to be in grayscale, because each pixel is automatically converted to grayscale in the [GetPixels](https://github.com/kelvins/lbph/blob/master/lbp/lbp.go#L55) function using the following [formula](https://en.wikipedia.org/wiki/Grayscale#Luma_coding_in_video_systems):

### Output

The `Predict` function returns 3 values:

* **label**: The label corresponding to the predicted image.
* **distance**: The distance between the histograms from the input test image and the matched image (from the training set).
* **err**: Some error that has occurred in the Predict step. If no error occurs it will return nil.

Using the label you can check if the algorithm has correctly predicted the image. In a real world application, it is not feasible to manually verify all images, so we can use the distance to infer if the algorithm has predicted the image correctly.

# Face recognition using LBPH Step by Step guide

In this section we will create a face recognition system using LBPH (local binary pattern histogram). This guide is didvide into multiple steps including creating own dataset, generate labels for images, train the model using created dataset, save the model, and testing the model with web-camera.

## Dataset [Dataset](#dataset)

We have created a dataset of sample size = 200 using openCV by extracting the 200 frames, and will convert into gray scale images. Images will be saved in <name>.<name_id>.<frame_num>.jpg

## Creating labels for images

### Define function to create images and labels for training
Below function creates the labels for the images.
```
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # create empth face list
    faceSamples = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image

        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces = detector.detectMultiScale(imageNp)
        # If a face is there then append that in the list as well as Id of it
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y + h, x:x + w])
            Ids.append(Id)
    return faceSamples, Ids
  
 ```
  
## Training model.[Train model](#train)

### Below code detects the face from the data, and trains the model.
```
model = cv2.face.LBPHFaceRecognizer_create()

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

faces, Id = getImagesAndLabels(image_path)

# Training the model
model.train(faces, np.array(Id))
```
  
## Save model.

```
model.save("./trained_model2.yml")
```
  
## Testing the model with web-camera
 
```
  #loading the model

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('model/trained_model2.yml')
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0

# add the list of names of your dataset here
names = ['rohan','none'] 


cam = cv2.VideoCapture(-1)

while True:
    ret, img =cam.read()
    #img = cv2.flip(img, -1) # Flip vertically
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5
       )
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        
        # If confidence is less them 100 ==> "0" : perfect match 
        if (confidence < 75):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(
                    img, 
                    str(id), 
                    (x+5,y-5), 
                    font, 
                    1, 
                    (255,255,255), 
                    2
                   )
         
    
    cv2.imshow('camera',img) 
    if cv2.waitKey(10) & 0xff == 'q':# Press 'ESC' for exiting video
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
```

## Clove the Repository and run **faceRecogition.ipynb** for steo byb step code execution.
                             
# References

* Ahonen, Timo, Abdenour Hadid, and Matti Pietikäinen. "Face recognition with local binary patterns." Computer vision-eccv 2004 (2004): 469-481. Link: https://link.springer.com/chapter/10.1007/978-3-540-24670-1_36

* Face Recognizer module. Open Source Computer Vision Library (OpenCV) Documentation. Version 3.0. Link: http://docs.opencv.org/3.0-beta/modules/face/doc/facerec/facerec_api.html

* Local binary patterns. Wikipedia. Link: https://en.wikipedia.org/wiki/Local_binary_patterns

* OpenCV Histogram Comparison. http://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_comparison/histogram_comparison.html

* Local Binary Patterns by Philipp Wagner. Link: http://bytefish.de/blog/local_binary_patterns/
                        
* Face recognition app by Arohi Singala . link https://www.youtube.com/watch?v=VTcMFtaIhag&lc=UgxbVdKzSQlvsEw9QLN4AaABAg
