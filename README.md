# Project: Child Seat Localization
## Goal
Build AI solution for child seat localization in the passenger vehicle
<p align="center">
   <img width="300" src="doc/inside-car-768x512.png">
</p>

## Problem statement
Localization of the child seat in the passenger vehicle enables the business to high-level services
and use cases.

## Data set
The data set shall be https://sviro.kl.dfki.de/data/ or any appropriate data set.
<a href="https://sviro.kl.dfki.de/data/" target="_parent"></a>

## Solution scope
The proposed software solution shall / may
* Instrument the data set
* State or use prior work from academic / industry
* Use appropriate state-of-the-art AI algorithm (preferably deep learning based)
* Enhance the data with synthetic
* Reasonable accuracy on the given scope of time to develop
* Rationales for the current Approach to enhance
* Use any appropriate open-source libraries and framework
* Focus more on the algorithm than the overall application
  
## Bonus Challenge (optional)
Extend the current dataset with synthetic data by inserting smoke inside the vehicle and
implement an AI smoke detection system.

## Delivery
The solution shall be presented in GitHub / GitLab code with optional supported documents in
email.

## Introduction
There are several frameworks that provide pre-trained deep learning models that can be customized for the most diverse tasks. As an example of frameworks we have: [TensorFlow 2 Object Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) and [Ultralytics YOLOv8](https://docs.ultralytics.com/modes/)

**TensorFlow** is an open-source software library for dataflow and differentiable programming across a range of tasks. It is widely used by data scientists and software engineers for building machine learning models, including object detection models. TensorFlow provides a detection model zoo, which is a collection of pre-trained object detection models that can be used for a variety of applications

**YoLo: You Only Look Once** is an extremely fast multi-object detection algorithm that uses a convolutional neural network (CNN) to detect and identify objects.
The neural network has this network architecture.

<p align="center">
   <img src="doc/yolo1_net.png">
</p>

* This [guide](https://neptune.ai/blog/how-to-train-your-own-object-detector-using-tensorflow-object-detection-api) provides step-by-step instructions for how to train a custom Object Detector using **TensorFlow Object Detection API**.

* This [guide](https://docs.ultralytics.com/modes/export/#arguments) provides step-by-step instructions for how to train a custom Object Detector using **YOLOv8**.

After studying the available documentation on training deep learning models using TensorFlow or YoLo, just for reasons of code simplicity, we decided to use YoLo for the development of our Object Detector.

From the family of models based on YoLo, we chose to use YoLov8 because, at the time of this work, it is considered a state-of-art (SOTA) for training real-time object detectors, offering cutting-edge performance in terms of accuracy and speed. We know that there is a family of models [YOLO-NAS](https://docs.ultralytics.com/models/yolo-nas/#overview) that in some comparative studies presented superior performance to YoLov8, but the YoLo-NAS models have not yet available for customization.

## Step 1. Data preparation
Explicar sobre o formato dos dados para treinamento. Falar sobre o software que usamos para rotular os dados

## Step 2. Training The Model
### Using Google Colab (recommended)
The easiest way to train, convert, and export a YoLo model is using Google Colab. Colab provides you with a free GPU-enabled virtual machine on Google's servers that comes pre-installed with the libraries and packages needed for training.

I wrote a [Google Colab notebook](./yolo_object_detection.ipynb) that can be used to train custom YoLo models. It goes through the process of preparing data, selecting a model for training, training the model, and running it on test images. It makes training a custom YoLo model as easy as uploading an image dataset and clicking Play on a few blocks of code!

<a href="https://colab.research.google.com/github/francineimorais/bh_assignment/blob/main/yolo_object_detection.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Open the Colab notebook in your browser by clicking the icon above. Work through the instructions in the notebook to start training your own model. Once it's trained is possible to [export to other formats](https://docs.ultralytics.com/modes/export/).

## Step 3. Other tasks
Além da deteção de objetos também possível treinar modelos baseados em YoLo para a realização de tarefas como: segmentation, classification, and pose estimation. Mais informações disponíveis no [link](https://docs.ultralytics.com/tasks/)

<details>
   <summary>Webcam</summary>
Make sure you have a USB webcam plugged into your computer. If you’re on a laptop with a built-in camera, you don’t need to plug in a USB webcam. 

From the `tflite1` directory, issue: 

```
python TFLite_detection_webcam.py --modeldir=TFLite_model 
```

After a few moments of initializing, a window will appear showing the webcam feed. Detected objects will have bounding boxes and labels displayed on them in real-time.
</details>

## FAQs
<details>
<summary>What's the difference between training, transfer learning, and fine-tuning?</summary>
<br>
Using correct terminology is important in a complicated field like machine learning. 
Here's my attempt at defining the terms:

* **Training**: The process of taking a full neural network with randomly initialized weights, passing in image data, calculating the resulting loss from its predictions on those images, and using backpropagation to adjust the weights in every node of the network and reduce its loss. In this process, the network learns how to extract features of interest from images and correlate those features to classes. Training a model from scratch typically takes millions of training steps and a large dataset of 100,000+ images (such as ImageNet or COCO). Let's leave actual training to companies like Google and Microsoft!
* **Transfer learning**: Taking a model that has already been trained, unfreezing the last layer of the model (i.e. making it so only the last layer's weights can be modified), and retraining the last layer with a new dataset so it can learn to identify new classes. Transfer learning takes advantage of the feature extraction capabilities that have already been learned in the deep layers of the trained model. It takes the extracted features and recategorizes them to predict new classes. 
* **Fine-tuning**: Fine-tuning is similar to transfer learning, except more layers are unfrozen and retrained. Instead of just unfreezing the last layer, a significant amount of layers (such as the last 20% to 50% of layers) are unfrozen. This allows the model to modify some of its feature extraction layers so it can extract features that are more relevant to the classes trying to identify.
</details>

<details>
<summary>Should I get a Google Colab Pro subscription?</summary>
<br>
If you plan to use Colab frequently for training models, I recommend getting a Colab Pro subscription. It provides several benefits:

* Idle Colab sessions remain connected for longer before timing out and disconnecting
* Allows for running multiple Colab sessions at once
* Priority access to TPU and GPU-enabled virtual machines
* Virtual machines have more RAM

Colab keeps track of how much GPU time you use and cuts you off from using GPU-enabled instances once you reach a certain use time. If you get the message telling you you're cut off from GPU instances, then that's a good indicator that you use Colab enough to justify paying for a Pro subscription.
</details>
