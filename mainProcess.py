import cv2 as cv2
import numpy as np
from functions import *
import keras
import matplotlib.pyplot as plt



# CV2.FINDING CONTOURS
# CV2.GAUSSIAN BLURR
# CV2.ADAPTIVE THERSHOLD
# CV2.ARCLENGTH
# CV2.APPROXPOLYDP
# CV2.GETPERSPECTIVETRANSFORM
# CV2.WARPPERSPECTIVE


# STEP 1. PREPARING THE IMAGE
# sudokuImagePath = 'Images/image_1.jpg'
# img = cv2.imread(sudokuImagePath)


def main(img):
    imgHeight = 360
    imgWidth = 360
    dimesnions = (imgHeight,imgWidth)
    img = cv2.resize(img,dimesnions)
    blankImg = np.zeros((imgHeight,imgWidth,3), dtype=np.uint8)


    # STEP 2. PRE PROCESS THROUGH GREY SCALING, GAUSSIAN BLURR AND THERSHOLDING THE IMAGE

    thersholdImg = preProcess(img)

    # STEP 3. FINDING ALL THE CONTOURS 

    contourImg  = img.copy()
    bigCountourImg = img.copy()
    contours, hierarchy = cv2.findContours(thersholdImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contourImg, contours, -1, (0, 255, 0), 3)


    # STEP 4. FINDING THE BIGGEST CONTOURS

    biggest = findBiggestContour(contours)
    cv2.drawContours(bigCountourImg, biggest, -1, (0, 200, 0), 15)


    # STEP 4. REORDERING THE POINTS
    reorderedPoints = reorder(biggest)
    cv2.drawContours(contourImg, reorderedPoints, -1, (0, 200, 0), 15)
    # print(reorderedPoints)


    # STEP 5. WRAPPING THE SUDOKU GRID IMAGE
    pts1 = np.float32(reorderedPoints)
    pts2 = np.float32([[0,0],[imgHeight,0],[0,imgWidth],[imgHeight,imgWidth]])

    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    final = cv2.warpPerspective(img, matrix, dimesnions)


    # # SHOWING ALL THE PROCESSED IMAGES 

    # titles = ["Original Image","Thersholded Image for Detecting Contours", "Detected Contours", "Found Sudoku Grid Points", "Extracted Sudoku Grid"]
    # stackedImages = [img,thersholdImg,contourImg,bigCountourImg,final]
    # fig = plt.figure(figsize=(15, 10))
    # plt.suptitle("PreProcessed Images",fontsize=16, y=0.95)
    # i=1
    # plt.title("PreProcessed Images")
    # for img in stackedImages:
    #     plt.subplot(2, 3, i)
    #     plt.title(titles[i-1], y=1.05)
    #     i+=1
    #     plt.axis('off')
    #     plt.imshow(img)
    # plt.show()



    # STEP 6. SPLITTING THE DIGITS IN SUDOKU GRID
    digits = getDigitBoxes(final)

    # STEP 7. CROPING THE BORDERS AND RESHAPING THOSE DIGIT BOXES
    digits = reshapeDigits(digits,6)

    # STEP 8. APPLYING FILTERS ON THOSES DIGIT BOXES FOR PREDICTION WITH TRAINED MODEL
    processedDigs = preProcessDigits(digits)

    # TO SHOW THE FINAL PROCEESED DIGIT CELL OF SUDOKU
    #showDigitBoxes(processedDigs)

    # STEP 9. LOADING THE TRAINED MODEL 

    loaded_model = keras.models.load_model('Trained_Model.h5')


    # STEP 10. MAKING A PREDICTION USING THE TRAINED MODEL

    predictions = loaded_model.predict(processedDigs)


    #  STEP 11. PROCESSING THE NUMBERS FROM NEURAL NETWORKS OUTPUT AND STORING THEM IN A LIST

    predicted_digits = []
    prediction_accuracy = []

    for pred in predictions:
        max = np.amax(pred)
        idx = np.argmax(pred)
        if max>0.97:
            predicted_digits.append(idx)
            prediction_accuracy.append(np.round(max,3))
        else:
            predicted_digits.append(0)
            prediction_accuracy.append(0.00)



    # plt.figure(figsize=(6, 6))
    # for i in range(81):
    #     plt.subplot(9, 9, i+1)
    #     plt.axis('off')
    #     plt.text(0.5, 0.5, f' {predicted_digits[i]}', color='black',
    #              fontsize=15, ha='center', va='center')
    # plt.show()


    #print(predicted_digits)
    #printSudoku(predicted_digits)
    # print("--------------------------------------------------------------")
    # printSudoku(prediction_accuracy)

    return predicted_digits










