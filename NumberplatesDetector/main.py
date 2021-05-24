import cv2
import numpy as np
import vehicles
import time
import os

cnt_up = 0
cnt_down = 0

cap = cv2.VideoCapture("clip3.mp4")
# cap=cv2.VideoCapture("demo1.mp4")
# cap=cv2.VideoCapture("Freewa.mp4")
# cap = cv2.VideoCapture(0)

# Get width and height of video

width = cap.get(3)
height = cap.get(4)
print(width)
print(height)
frameArea = width * height
areaTH = frameArea / 400

# Testing


# Testing change position
line_up = int(3 * (height / 5))
line_down = int(2 * (height / 5))

print(line_up)
print(line_down)
up_limit = int(1 * (height / 5))
down_limit = int(4 * (height / 5))

print("Red line y:", str(line_down))
print("Blue line y:", str(line_up))
line_down_color = (255, 0, 0)
line_up_color = (225, 0, 255)
pt1 = [0, line_down]
pt2 = [width, line_down]
pts_L1 = np.array([pt1, pt2], np.int32)
pts_L1 = pts_L1.reshape((-1, 1, 2))
pt3 = [0, line_up * 1.12]
pt4 = [width, line_up * 1.12]

pts_L2 = np.array([pt3, pt4], np.int32)
pts_L2 = pts_L2.reshape((-1, 1, 2))

pt5 = [0, down_limit]
pt6 = [width, down_limit]
pts_L3 = np.array([pt5, pt6], np.int32)
pts_L3 = pts_L3.reshape((-1, 1, 2))
pt7 = [0, up_limit]
pt8 = [width, up_limit]
pts_L4 = np.array([pt7, pt8], np.int32)
pts_L4 = pts_L4.reshape((-1, 1, 2))

# Background Subtractor
fgbg = cv2.bgsegm.createBackgroundSubtractorGSOC(noiseRemovalThresholdFacBG=0.01, noiseRemovalThresholdFacFG=0.0001)

# Kernals
kernalOp = np.ones((3, 3), np.uint8)
kernalOp2 = np.ones((5, 5), np.uint8)
kernalCl = np.ones((11, 11), np.uint8)

font = cv2.FONT_HERSHEY_SIMPLEX
cars = []

max_p_age = 5
pid = 1

img_num = 0

# initialize array to store mouse coords for drawing a line
lineDrawn = []

# specifying output folder for exports
path = 'output'
os.makedirs(path, exist_ok=True)

def mouse_handler(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # there can only be two coords, a starting and
        # an end point to form a line
        # TODO: ability to draw multiple lines separately
        if len(lineDrawn) > 2:
            lineDrawn.clear()

        cv2.circle(img, (x, y), 3, (0, 0, 255), 3, cv2.FILLED)
        lineDrawn.append((x, y))
    if event == cv2.EVENT_RBUTTONDOWN:
        lineDrawn.clear()

while (cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    for i in cars:
        i.age_one()
    fgmask = fgbg.apply(frame)
    fgmask2 = fgbg.apply(frame)

    if ret == True:

        # Binarization
        ret, imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        ret, imBin2 = cv2.threshold(fgmask2, 200, 255, cv2.THRESH_BINARY)
        # OPening i.e First Erode the dilate
        mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernalOp)
        mask2 = cv2.morphologyEx(imBin2, cv2.MORPH_CLOSE, kernalOp)

        # Closing i.e First Dilate then Erode
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernalCl)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernalCl)

        # Find Contours
        countours0, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in countours0:
            area = cv2.contourArea(cnt)
            print("this is area:" + str(area))
            if area > areaTH:
                ####Tracking######
                m = cv2.moments(cnt)
                cx = int(m['m10'] / m['m00'])
                cy = int(m['m01'] / m['m00'])
                x, y, w, h = cv2.boundingRect(cnt)

                new = True
                if cy in range(up_limit, down_limit):
                    for i in cars:
                        if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h:
                            new = False
                            i.updateCoords(cx, cy)

                            if i.going_UP(line_down, line_up) == True:
                                cnt_up += 1
                                print("ID:", i.getId(), 'crossed going up at', time.strftime("%c"))
                            elif i.going_DOWN(line_down, line_up) == True:
                                cnt_down += 1
                                roi = frame[y:y + h, x:x + w]
                                img_num += 1
                                file_name = "test" + str(img_num) + ".png"
                                cv2.imwrite(os.path.join(path, file_name), roi)
                                if img_num > 30:
                                    exit()
                                print("ID:", i.getId(), 'crossed going up at', time.strftime("%c"))
                            break
                        if i.getState() == '1':
                            if i.getDir() == 'down' and i.getY() > down_limit:
                                i.setDone()
                            elif i.getDir() == 'up' and i.getY() < up_limit:
                                i.setDone()
                        if i.timedOut():
                            index = cars.index(i)
                            cars.pop(index)
                            del i

                    if new == True:  # If nothing is detected,create new
                        p = vehicles.Car(pid, cx, cy, max_p_age)
                        cars.append(p)
                        pid += 1

                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for i in cars:
            cv2.putText(frame, 'ID: ' + str(i.getId()), (i.getX(), i.getY()), font, 0.3, i.getRGB(), 1, cv2.LINE_AA)

        str_up = 'UP: ' + str(cnt_up)
        str_down = 'DOWN: ' + str(cnt_down)
        frame = cv2.polylines(frame, [pts_L1], False, line_down_color, thickness=2)
        frame = cv2.polylines(frame, [pts_L2], False, line_up_color, thickness=2)
        frame = cv2.polylines(frame, [pts_L3], False, (255, 255, 255), thickness=1)
        frame = cv2.polylines(frame, [pts_L4], False, (255, 255, 255), thickness=1)
        cv2.putText(frame, str_up, (10, 40), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str_up, (10, 40), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, str_down, (10, 90), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str_down, (10, 90), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, str(width), (10, 60), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # grabbing coords stored in array if present
        # then start drawing over latter frames
        if len(lineDrawn) == 2:
            cv2.circle(frame, lineDrawn[0], 3, (0, 0, 255), 3, cv2.FILLED)
            cv2.circle(frame, lineDrawn[1], 3, (0, 0, 255), 3, cv2.FILLED)
            cv2.line(frame, lineDrawn[0], lineDrawn[1], (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Frame', frame)
        cv2.setMouseCallback('Frame', mouse_handler)
        cv2.imshow('Frame2', mask2)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
