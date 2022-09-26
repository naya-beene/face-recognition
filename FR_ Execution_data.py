import cv2

labels = ["chan_youn", "hwe_eun", "nam_kyung", "seong_bin"]  # 라벨 지정

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face-trainner.yml")  # 저장된 값 가져오기

cap = cv2.VideoCapture(0)  # 카메라 실행

cof_sum = 0
conf_list = []
cnt = 0
cnt2 = 0

if cap.isOpened() == False:  # 카메라 생성 확인
    exit()

while True:

    ret, img = cap.read()  # 현재 이미지 가져오기
    img = cv2.flip(img, 1)
    # img = img[50:360, 160:480]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 흑백으로 변환
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)  # 얼굴 인식

    for (x, y, w, h) in faces:
        b_center = (x + w / 2, y + h / 2)
        # print(b_center)
        if b_center[0] < 260 or b_center[0] > 380 or b_center[1] < 180 or b_center[1] > 300:
            break

        roi_gray = gray[y:y + h, x:x + w]  # 얼굴 부분만 가져오기

        id_, conf = recognizer.predict(roi_gray)  # 얼마나 유사한지 확인
        conf_list.append(conf)

        if conf < 500:
            print(id_, conf)
            cof_sum += conf
            cnt += 1
            if id_ == 3:
                cnt2 += 1
            font = cv2.FONT_HERSHEY_SIMPLEX  # 폰트 지정
            name = labels[id_]  # ID를 이용하여 이름 가져오기
            cv2.putText(img, name, (x, y), font, 1, (0, 0, 255), 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.rectangle(img, (230, 150), (410, 330), (255, 0, 0), 2)
    cv2.rectangle(img, (260, 180), (380, 300), (255, 0, 0), 2) # 중앙점 범위
    cv2.imshow('Preview', img)  # 이미지 보여주기
    if cv2.waitKey(10) >= 0:  # 키 입력 대기, 10ms
        break

print('Frame width:', int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print('Frame height:', int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('Frame count:', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

fps = cap.get(cv2.CAP_PROP_FPS)
print('FPS:', fps)
print(cof_sum/cnt)
print(cnt2, cnt, str(int(cnt2 / cnt * 100)) + '%')

# with open('conf_data.txt', 'w', encoding='UTF-8') as f:
#     for conf in conf_list:
#         f.write(str(conf)+'\n')



# 전체 종료
cap.release()
cv2.destroyAllWindows()