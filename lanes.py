import cv2
import numpy as np

def make_cordinates(image, line_parameters):
    slope, intercept = line_parameters
    #represent image height
    y1 = image.shape[0]
    #start from 704 and go 3/5 way upwards
    y2 = int(y1*(3/5))
    x1= int((y1-intercept)/slope)
    x2= int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])

def avg_slope_intercept(image, lines):
    #takes the cordinate of left & right lines
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameter = np.polyfit((x1,x2) , (y1,y2) , 1)
        slope = parameter[0]
        intercept = parameter[1]
        #this will give all the cordinates of line and add it to the list
        if slope < 0:
            left_fit.append((slope , intercept))
        else:
            right_fit.append((slope , intercept))
        #this will average out cordinates into single slope and y intercept
    left_fit_avg = np.average(left_fit , axis=0)
    right_fit_avg = np.average(right_fit , axis=0)

    left_line = make_cordinates(image , left_fit_avg)
    right_line = make_cordinates(image , right_fit_avg)
    return np.array([left_line , right_line])

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5),0)
    canny = cv2.Canny(blur,50,150)
    return canny

def region_of_intrst(image):
        height = image.shape[0]
        polygons = np.array([[(200, height), (1000 , height) , (550, 250)]])
        mask = np.zeros_like(image)
        cv2.fillPoly(mask , polygons, 255)
        masked_image =cv2.bitwise_and(image , mask)
        return masked_image

def Display_line(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1,y1,x2,y2 in lines:
            cv2.line(line_image, (x1 , y1) , (x2 , y2) , (300,0,0) , 10)
    return line_image

cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _,frame = cap.read()
    canny_image = canny(frame)
    cropped_img=region_of_intrst(canny_image)
    lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    avg_line = avg_slope_intercept(frame, lines)
    line_image = Display_line(frame , avg_line)
    final_image = cv2.addWeighted(frame,0.8,  line_image, 1, 1)
    cv2.imshow('resut' ,final_image)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
