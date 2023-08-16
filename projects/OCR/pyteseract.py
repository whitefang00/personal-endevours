import pytesseract
import cv2
import pandas as pd
import re
imgFiles = ['jpg_21.JPG', 'jpg_7.JPG']

currData = pd.DataFrame(columns=["Raw Data"])


# loading the image into memory
def greyScale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def ocr(img):
    img1 = cv2.imread(img)
    grayImage = greyScale(img1)
    thresh, im_bw = cv2.threshold(grayImage, 50, 200, cv2.THRESH_BINARY)

    return str(pytesseract.image_to_string(im_bw))


ind = 0
for i in imgFiles:
    strData = ocr(i)
    data = re.sub(r"","", strData)
    print(type(data))
    currData.at[i, 'Raw Data'] = data
    ind += 1
currData.to_excel("rawData.xlsx")

def cleanReformat(DataFrame):
    for i in range(len(DataFrame.iloc[:,0])):
        print(len(DataFrame.iloc[:,0]))

cleanReformat(currData)