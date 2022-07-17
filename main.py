# COMPUTER VISION AND IOT
#THE SPARKS FOUNDATION

#TASK 1 : Optical Character Recognition (ORC)

#NAME : Suhaila Ahmed Farouk


# importing computer Vision Library
import cv2
#Python package that allows you to extract text from images
import pytesseract

# path of tesseract
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
# Uploading The Image
img = cv2.imread("C:\\Users\\moham\\Desktop\\ocr\\ocr project\\photo.jpg")

#Store the text
extract_text = pytesseract.image_to_string(img)
print(extract_text)

# DISPLAYING THE OUTPUT IMAGE
cv2.imshow("ocr",img)
cv2.waitKey(0)
