
import cv2
import numpy as np

vid = cv2.VideoCapture(0)
out = cv2.VideoWriter('video.mp4', -1, 20.0, (640,480))

# инициализировать распознаватель лиц (каскад Хаара по умолчанию)
face_cascade = cv2.CascadeClassifier('C:\\Users\\MVideo\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python310\\site-packages\\cv2\\data/haarcascade_frontalface_default.xml')
rect_cascade = cv2.CascadeClassifier()
# параметры цветового фильтра
#hsv_min = np.array((25, 45,45), np.uint8)
#hsv_max = np.array((255, 255, 255), np.uint8)
hsv_min = np.array((45, 60, 70), np.uint8)
hsv_max = np.array((255, 255, 255), np.uint8)
while(True):
    ret, frame = vid.read()
    filterd_image  = cv2.GaussianBlur(frame, (7,7),0)
    img_grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    img_grey = cv2.GaussianBlur(img_grey,(5,5),0)
    #сегменты
    thresh = 150
    #применение hsv фильтра
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #получение сегментированного изображения
    #ret, thresh_img = cv2.threshold(hsv, thresh, 255, cv2.THRESH_BINARY)
    thresh_img = cv2.inRange(hsv, hsv_min,hsv_max);

    #ищем контуры
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    #cv2.drawContours(frame, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    # перебираем все найденные контуры в цикле
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)  # пытаемся вписать прямоугольник
        box = cv2.boxPoints(rect)  # поиск четырех вершин прямоугольника
        box = np.int0(box) # округление координат
        area = int(rect[1][0]*rect[1][1]) # вычисление площади
        if area > 19000:
            cv2.drawContours(frame,[box],0,(255,0,0),2)
    
    # обнаружение всех лиц на изображении
    faces = face_cascade.detectMultiScale(img_grey)
    # для всех обнаруженных лиц рисуем квадрат
    for x, y, width, height in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), color=(0,0, 255), thickness=2)
    
    # выводим итоговое изображение в окно    
    cv2.imshow('origin', frame)
    cv2.imshow("thresh", thresh_img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    if cv2.waitKey(1) & 0xFF==ord('s'):
        cv2.imwrite('img.png',frame)
    #записывает при нажатии
    if cv2.waitKey(1) & 0xFF==ord('v'):
        out.write(frame)
