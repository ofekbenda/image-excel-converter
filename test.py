# https://towardsdatascience.com/a-table-detection-cell-recognition-and-text-extraction-algorithm-to-convert-tables-to-excel-files-902edcf289ec
import datetime

import cv2
import numpy as np
import pandas as pd

import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


def convert(imagePath, outputFolderPath):

    # read your file
    #file = r'table_ara_1.png'
    file = imagePath
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    # thresholding the image to a binary image
    thresh, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    cv2.imwrite('images\\test_b&w.png', img_bin)

    # Length(width) of kernel as 100th of total width
    #todo: test with real tables and to check the optimal default
    #todo: fix the masking for arabic words - need longer mask because the letters connect to each other
    #fit mask to vertical line
    kernel_ver_len = 5
    if len(str(img.shape[0])) == 4:
        kernel_ver_len = img.shape[0] // 100
    elif len(str(img.shape[0])) == 3:
        kernel_ver_len = img.shape[0] // 10

    # fit mask to horizon line
    kernel_hor_len = 5
    if len(str(img.shape[1])) == 4:
        kernel_hor_len = img.shape[1] // 100
    elif len(str(img.shape[1])) == 3:
        kernel_hor_len = img.shape[1] // 10

    #Defining a vertical kernel to detect all vertical lines of image
    #MORPH_RECT = shape of rectangular
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_ver_len))

    # Defining a horizontal kernel to detect all horizontal lines of image
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_hor_len, 1))
    # A kernel of 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # Use vertical kernel to detect and save the vertical lines in a jpg
    #todo: work with 2 iterations. check runTime with 2 iterations
    image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
    cv2.imwrite("images\\vertical.jpg", vertical_lines)

    # Use horizontal kernel to detect and save the horizontal lines in a jpg
    image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
    cv2.imwrite("images\\horizontal.jpg", horizontal_lines)


    # Combine horizontal and vertical lines in a new third image, with both having same weight.
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)

    # Eroding and thesholding the image - enlarge grid lines for easy detection
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    thresh, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite("images\\img_vh.jpg", img_vh)

    #detect the text from original img and the lines from img_vh
    bitxor = cv2.bitwise_xor(img, img_vh)
    bitnot = cv2.bitwise_not(bitxor)
    cv2.imwrite("images\\bitnot.jpg", bitnot)


    # Detect contours for following box detection
    #contours is array of arrays. each array include 4 points that built a frame
    #first frame- present the border of the picture
    #second frame - present the border of the table
    #and the others are the border of each cell in the table
    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    def sort_contours(cnts, method="left-to-right"):
        # initialize the reverse flag and sort index
        reverse = False
        i = 0
        # handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        # handle if we are sorting against the y-coordinate rather than
        # the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        # construct the list of bounding boxes and sort them from top to
        # bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))
        # return the list of sorted contours and bounding boxes
        return (cnts, boundingBoxes)


    # Sort all the contours by top to bottom and right to left by default.
    #boundringBoxes return list that each node
    #(y's coordinate of the left-up point of the box, x's coordinate, width, height)
    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")


    # Create list box to store all boxes in
    box = []
    heights = []
    # Get position (x,y), width and height for every contour and show the contour on image
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w < img.shape[1]/2 or h < img.shape[0]/2):
            box.append([x, y, w, h])
            heights.append(h)

    # Get mean of heights
    mean = np.mean(heights)

    # Creating two lists to define row and column in which cell is located
    row = []
    column = []
    j = 0

    # Sorting the boxes to their respective row and column
    for i in range(len(box)):
        if (i == 0):
            column.append(box[i])
            previous = box[i]

        else:
            #todo: check table with unstracture row and column
            if (box[i][1] <= previous[1] + mean / 2):
                column.append(box[i])
                previous = box[i]

                if (i == len(box) - 1):
                    row.append(column)

            else:
                row.append(column)
                column = []
                previous = box[i]
                column.append(box[i])

                if (i == len(box) - 1):
                    row.append(column)


    # calculating maximum number of columns in row
    countcol = 0
    ind = 0
    for i in range(len(row)):
        if len(row[i]) > countcol:
            countcol = len(row[i])
            ind = i

    #Retrieving the center of each column

    center = [int(row[ind][j][0] + row[ind][j][2] / 2) for j in range(len(row[ind]))]
    center = np.array(center)
    center.sort()

    # Regarding the distance to the columns center, the boxes are arranged in respective order
    finalboxes = []
    for i in range(len(row)):
        lis = []
        for k in range(countcol):
            lis.append([])
        for j in range(len(row[i])):
            diff = abs(center - (row[i][j][0] + row[i][j][2] / 4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            lis[indexing].append(row[i][j])
        finalboxes.append(lis)


    # from every single image-based cell/box the strings are extracted via pytesseract and stored in a list
    outer = []
    for i in range(len(finalboxes)):
        for j in range(len(finalboxes[i])):
            inner = ''
            if (len(finalboxes[i][j]) == 0):
                outer.append(' ')
            else:
                for k in range(len(finalboxes[i][j])):
                    y, x, w, h = finalboxes[i][j][k][0], finalboxes[i][j][k][1], finalboxes[i][j][k][2], \
                                 finalboxes[i][j][k][3]
                    finalimg = bitnot[x:x + h, y:y + w]
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                    border = cv2.copyMakeBorder(finalimg, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
                    resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

                    erosion = cv2.erode(resizing, kernel, iterations=2)
                    cv2.imwrite("images\\erosion.jpg", erosion)

                    dilation = cv2.dilate(erosion, kernel, iterations=1)

                    #thresh, dilation = cv2.threshold(dilation, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    cv2.imwrite("images\\dilation.jpg", dilation)

                    out = pytesseract.image_to_string(dilation,lang='Arabic+ara')
                    #todo: test eith real table in the MAABADA
                    #todo: psm 6 for small font and psm 3 for large font
                    if (len(out) == 0):
                        out = pytesseract.image_to_string(dilation,lang='Arabic+ara', config='--psm 6')
                        print("herre:", out)
                    inner = inner + " " + out
                outer.append(inner)

    # Creating a dataframe of the generated OCR list
    arr = np.array(outer)
    dataframe = pd.DataFrame(arr.reshape(len(row), countcol))
    print(dataframe)

    data = dataframe.style.set_properties(align="left")
    # Converting it in a excel-file
    data.to_excel(outputFolderPath + "//" + datetime.datetime.now().strftime("%d_%m_%Y-%H_%M") + ".xlsx")
