# Emotion detection using deep learning

## Introduction

This project aims to classify the emotion on a person's face into one of **seven categories**:

- angry
- disgusted
- fearful
- happy
- neutral
- sad
- surprised

The project uses deep convolutional neural networks. The model is trained on the **FER-2013** dataset, The dataset consists of 35887 grayscale, 48x48 sized face images.

The User interface is built using eel, a Python library that allows to create web applications in Python.

## Dependencies

- Python 3
- [OpenCV](https://opencv.org/)
- [Tensorflow](https://www.tensorflow.org/)
- eel
- tkinter
- matplotlib

##### Download the FER-2013 dataset from [here](https://drive.google.com/file/d/1X60B-uR3NtqPd4oosdotpbDgy8KOfUdr/view?usp=sharing)

- To run the project (windows)

```bash
python main.py
```

- To run the project (linux/unix)

```bash
python3 main.py
```

## Algorithm

- First, the **haar cascade** method is used to detect faces in each frame of the webcam feed.

- The region of image containing the face is resized to **48x48** and is passed as input to the CNN.

- The network outputs a list of **softmax scores** for the seven classes of emotions.

- The emotion with maximum score is displayed on the screen.

- Algorithm Code [here](https://github.com/atulapra/Emotion-detection)
