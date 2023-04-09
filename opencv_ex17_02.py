#특정 색 기반으로 추적하기
#Meanshift 활용 : 특정 색이 밀집한 곳을 찾음.
import cv2 as cv

is_clicking_mouse = False

s_x, s_y, e_x, e_y = -1, -1, -1, -1
step = 0
track_win = None

#마우스 콜백함수
def on_click_mouse(event, x, y, flags, param):
    global s_x, s_y, e_x, e_y, step, track_win
    if event == cv.EVENT_LBUTTONDOWN:
        step = 1
        is_clicking_mouse = True
        s_x = x
        s_y = y
    
    elif event == cv.EVENT_MOUSEMOVE:
        step = 2
        e_x = x
        e_y = y

    elif event == cv.EVENT_LBUTTONUP:
        step = 3
        is_clicking_mouse = False
        e_x = x
        e_y = y

cap = cv.VideoCapture(0)
if cap.isOpened() == False:
    print("카메라를 열 수 없습니다.")
    exit(1)

cv.namedWindow('Color')
cv.setMouseCallback('Color', on_click_mouse)

while(True):
    ret, img_color = cap.read()
    if ret == False:
        print("캡쳐 실패")
        break
 
    if step == 1: #처음 클릭 시 원을 보여줍니다.        
        cv.circle(img_color, (s_x, s_y), 10, (0, 255, 0), -1)
    elif step == 2: #마우스 이동 중 사각형을 그려줍니다.
        cv.rectangle(img_color, (s_x, s_y), (e_x, e_y), (0, 255, 0), 3)
    elif step == 3: #손을 뗀 경우 ROI 얻음.
        if s_x > e_x:
            s_x, e_x = e_x, s_x
            s_y, e_y = e_y, s_y
        #초기 사각형 위치
        track_win = (s_x, s_y, e_x - s_x, e_y - s_y)

        # HSV 색공간으로 변환
        img_hsv = cv.cvtColor(img_color, cv.COLOR_BGR2HSV)
        print(f'({s_x}, {s_y}) .....({e_x}, {e_y}) ')
        img_ROI = img_hsv[s_y:e_y, s_x:e_x]

        cv.imshow("ROI", img_ROI)
        # ROI 히스토그램을 계산합니다.
        objectHistogram = cv.calcHist([img_ROI], [0], None, [180], 
                (0, 180))
        # 히스토그램 0~255 사이 값을 갖도록 정규화
        cv.normalize(objectHistogram, objectHistogram, alpha=0, beta=255, 
                    norm_type=cv.NORM_MINMAX)
        step = step + 1

    elif step == 4:
        
        img_hsv = cv.cvtColor(img_color, cv.COLOR_BGR2HSV)
        # Histogram BackProject 수행하여 img_hsv 에서 objectHistogram 갖는 영역을 찾음.
        bp = cv.calcBackProject([img_hsv], [0], objectHistogram, [0,180], 1)
        # meanshift 적용하여 새로운 Object 위치를 얻음.
        ret, track_win = cv.meanShift(bp, track_win, 
            ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 ))
        x, y, w, h = track_win
        # object에 빨간색 사각형을 그려준다.
        cv.rectangle(img_color, (x,y), (x+w, y+h), (0, 0, 255), 2)


    cv.imshow("Color", img_color)
 
    if cv.waitKey(25) >= 0:
        break
