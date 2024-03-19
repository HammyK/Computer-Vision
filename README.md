## Overview
This repository contains code and resources related to a computer vision project focusing on facial expression recognition (FER) using various techniques. The project includes the implementation of Convolutional Neural Networks (CNN), Histogram of Oriented Gradients (HOG) with Support Vector Machines (SVM), and Scale-Invariant Feature Transform (SIFT) with SVM for FER. Additionally, a test function for evaluating model performance and a video recognition demo are provided.

### Files Included
- **FER_using_CNN.ipynb**: Jupyter notebook containing the implementation of FER using CNN
- **FER_using_Hog_with_SVM.ipynb**: Jupyter notebook containing the implementation of FER using HOG with SVM
- **FER_using_SIFT_with_SVM.ipynb**: Jupyter notebook containing the implementation of FER using SIFT with SVM
- **Test_function.ipynb**: Jupyter notebook containing a test function for evaluating model performance
- **Video Recognition Results.mp4**: Video file demonstrating the results of facial expression recognition

### Usage
1. **FER_using_CNN.ipynb**: Open this notebook in Jupyter or any compatible environment. Follow the instructions provided within the notebook to run and evaluate the CNN-based facial expression recognition model
2. **FER_using_Hog_with_SVM.ipynb**: Similar to the above, follow the instructions in this notebook to run the HOG with SVM model for facial expression recognition
3. **FER_using_SIFT_with_SVM.ipynb**: This notebook contains the implementation of facial expression recognition using SIFT with SVM. Run the cells sequentially to execute the code
4. **Test_function.ipynb**: Open and run this notebook to evaluate the performance of the implemented models using the provided test function
5. **Video Recognition Results.mp4**: View the video file to observe the results of facial expression recognition in a demo setting

### Dependencies
Ensure that you have the necessary libraries and packages installed to run the provided code. Common dependencies include:
- Python (>=3.6)
- Jupyter Notebook
- NumPy
- OpenCV
- Scikit-learn
- TensorFlow (for CNN implementation)

----

## Computer Vision Report
The project revolves around facial emotion detection using a dataset comprising 12,271 training images and 3,068 test images. These images depict faces categorized into seven emotion labels: surprise, fear, disgust, happiness, sadness, anger, and neutral. To ensure uniformity, all images have been scaled to 100x100 pixels. Data cleaning involved removing the 'Aligned' word from file names. The images were paired with their corresponding labels using the Pandas library in Python. However, it's worth noting that the dataset suffers from class imbalance, with label 4 (happiness) being predominant.

### Implemented Methods
- #### Classifiers
- **Support Vector Machines (SVM):** Utilized for its ability to handle multi-classification problems, SVM was combined with two feature descriptors: Histogram of Oriented Gradients (HOGS) and Scale-Invariant Feature Transform (SIFT). SVM constructs a hyperplane to differentiate between classes in the feature space.
- **Convolutional Neural Network (CNN):** A deep neural network architecture specifically tailored for image-related tasks. CNN automatically extracts features from images through convolutional and pooling layers, followed by classification layers.

- #### Feature Extractors
- **SIFT (Scale-Invariant Feature Transform):** Chosen for its ability to provide distinctive and invariant features across different scales and orientations in images. Implemented using OpenCV, SIFT features were combined with SVM for classification.
- **HOGS (Histograms of Oriented Gradients):** A descriptor that captures the distribution of intensity gradients in images, commonly used for object detection. Implemented using Scikit Image, HOGS features were also combined with SVM for classification.

- #### In the Wild Video Recognition
To apply the best performing model to real-world scenarios, we developed a pipeline for facial emotion recognition in videos. This involved pre-processing the video data, performing face detection using Haar Cascade, and subsequently applying the chosen classification model.

### Results
- **Accuracy and Time Comparison**
  - SVM + HOGS achieved an accuracy of 0.62 in 31 seconds
  - SVM + SIFT achieved an accuracy of 0.39 in 718 seconds
  - CNN achieved the highest accuracy of 0.76 with a training time of 810 seconds
- **Discussion**
  - SVM with HOGS outperformed SVM with SIFT and achieved competitive results compared to CNN
  - CNN exhibited high accuracy but required more computational resources
  - Overfitting was observed in SVM models due to the dominance of the 'happiness' class
- **Video Recognition**
  - The CNN model applied to the video accurately recognized facial emotions, albeit with occasional misclassifications, especially in crowded scenes

### Future Work
- Addressing class imbalance through techniques like SMOTE or label combination
- Exploring alternative face detection algorithms such as MobileNet or ResNet for improved accuracy in video recognition
- Conducting a more extensive hyperparameter search to further enhance model accuracies
- Investigating the integration of additional feature extraction methods or classifiers to improve performance

### Conclusion
The project successfully implemented various computer vision techniques for facial emotion recognition, showcasing the effectiveness of SVM with HOGS and CNN. Despite challenges such as class imbalance and computational requirements, the models demonstrated promising results in both image and video recognition tasks. Further improvements and optimizations can be explored for future iterations of the project.
