#특정 색 기반으로 추적하기
import cv2 as cv


cap = cv.VideoCapture(0)
if cap.isOpened() == False:
    print("카메라를 열 수 없습니다.")
    exit(1)


while(True):
    ret, img_frame = cap.read()
    if ret == False:
        print("캡쳐 실패")
        break

    # HSV 색공간으로 변환
    img_hsv = cv.cvtColor(img_frame, cv.COLOR_BGR2HSV)

    # Hue  값 기준으로 파란색 영역을 검출함.
    lower_blue = (120-20, 70, 0)
    upper_blue = (120+20, 255, 255)
    img_mask = cv.inRange(img_hsv, lower_blue, upper_blue)

    # 모폴로지 연산으로 바이너리 이미지 개선함.
    kernel = cv.getStructuringElement( cv.MORPH_RECT, ( 11, 11 ) )
    img_mask = cv.morphologyEx(img_mask, cv.MORPH_CLOSE, kernel)


    img_result = cv.bitwise_and(img_frame, img_frame, mask = img_mask)


    nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(img_mask)


    for i in range(nlabels):

        if i < 1:
            continue

        area = stats[i, cv.CC_STAT_AREA]
        center_x = int(centroids[i, 0])
        center_y = int(centroids[i, 1]) 
        left = stats[i, cv.CC_STAT_LEFT]
        top = stats[i, cv.CC_STAT_TOP]
        width = stats[i, cv.CC_STAT_WIDTH]
        height = stats[i, cv.CC_STAT_HEIGHT]

        print(area)
 
        if area > 5000:
            cv.rectangle(img_frame, (left, top), (left + width, top + height), 
                    (0, 0, 255), 3)
            # 영역의 중심 좌표에 원을 그립니다.
            cv.circle(img_frame, (center_x, center_y), 5, (255, 0, 0), 3)
            # 라벨 번호를 표시합니다.
            cv.putText(img_frame, str(i), (left + 20, top + 20), 
                    cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    cv.imshow('Color', img_frame)
    cv.imshow('Result', img_result)

    key = cv.waitKey(1) 
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()
