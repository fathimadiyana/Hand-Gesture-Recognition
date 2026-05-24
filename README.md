# Hand Gesture Recognition

## Internship
SkillCraft Technology - Machine Learning Task 4

## Objective
To develop a hand gesture recognition model that can identify and classify different hand gestures from image data, enabling gesture-based human-computer interaction.

## Dataset
LeapGestRecog Dataset from Kaggle  
Dataset Link: https://www.kaggle.com/datasets/gti-upm/leapgestrecog

## Algorithm Used
Random Forest Classifier

## Features Used
- Hand gesture images
- Grayscale image pixels
- Resized images of 64x64 pixels

## Steps Performed
1. Loaded the hand gesture image dataset
2. Extracted the dataset
3. Preprocessed images
4. Converted images to grayscale
5. Resized images to 64x64
6. Flattened images into numerical features
7. Encoded gesture labels
8. Split data into training and testing sets
9. Trained a Random Forest classification model
10. Evaluated model accuracy
11. Built a Streamlit dashboard for prediction
12. Displayed predicted gesture and confidence score

## Technologies Used
- Python
- NumPy
- OpenCV
- Scikit-learn
- Streamlit
- Pillow
- VS Code
- GitHub

## How to Run

Clone the repository:

```bash
git clone https://github.com/fathimadiyana/Hand-Gesture-Recognition.git
cd Hand-Gesture-Recognition