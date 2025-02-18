import cv2 #OpenCV 라이브러리 불러오기
import numpy as np #Numpy 라이브러리 불러오기(배열을 다룸)


# YOLO 모델 불러오기(.weights는 가중치 파일 / .cfg는 모델의 구조를 정의하는 파일) *파일 경로 확인 필수*
yolo_net = cv2.dnn.readNet(r"C:\OpenCV_Project\YOLO\yolov3.weights", r"C:\OpenCV_Project\YOLO\yolov3.cfg")

# coco.names 파일 불러오기(YOLO 모델이 인식할 수 있는 객체들의 이름이 담긴 파일)
with open(r"C:\OpenCV_Project\YOLO\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 테스트할 비디오 파일 경로(업로드 받은 파일 가져오기로..)
cap = cv2.VideoCapture(r"C:\OpenCV_Project\OpenCV_test_videos\OpenCV_test_video2_middle.mp4")  

# YOLO 모델의 출력 레이어 정보 가져오기
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

# 부정행위로 인식하지 않을 사물들(이 사물들 빼고는 모두 다 부정행위)
normal_objects = ["person", "chair", "laptop", "desktop"] 

speed = 1 # 배속 (1은 정상 속도, 2는 2배속, 0.5는 절반 속도)


while True:
    ret, frame = cap.read() #비디오에서 프레임을 한 장씩 읽어오고, 성공 여부를 반환함
    if not ret: #아닌 경우 or 비디오가 끝날 경우
        break



    # 이미지 크기 조정
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False) 
    #->(YOLO 모델에 맞게 전처리함, 이미지 정규화 비율, (YOLO 모델이 요구하는 이미지 크기), (색상의 평균값을 빼는 옵션), 이미지 크롭 X)
    yolo_net.setInput(blob)
    outputs = yolo_net.forward(output_layers) #YOLO모델의 출력 결과를 계산

    # 객체 탐지 결과 처리
    class_ids = [] #탐지된 객체가 어떤 클래스인지 저장하는 변수
    confidences = [] #객체를 탐지했을 때, 정확도에 대한 확신 정도를 저장 
    boxes = [] #객체를 탐지했을 때, 그 객체가 화면에서 위치하는 곳과 크기를 저장
    for output in outputs:
        for detection in output:
            scores = detection[5:] #각 객체에 대한 신뢰도
            class_id = np.argmax(scores) #신뢰도가 가장 높은 클래스를 선택택
            confidence = scores[class_id] #해당 클래스의 신뢰도를 codfidence라고 저장
            if confidence > 0.5: #신뢰도가 0.5 이상일 때만 해당 객체를 처리함(테스트 해보고 조절 가능)
                center_x = int(detection[0] * frame.shape[1]) #객체의 중심 좌표 계산
                center_y = int(detection[1] * frame.shape[0]) #객체의 중심 좌표 계산
                w = int(detection[2] * frame.shape[1]) #객체의 너비 계산
                h = int(detection[3] * frame.shape[0]) #객체의 높이 계산

                # 직사각형 좌표 계산
                x = int(center_x - w / 2) #중심 좌표 - 너비의 반 = 왼쪽 좌표
                y = int(center_y - h / 2) #중심 좌표 - 높이의 반 = 상단 좌표

                boxes.append([x, y, w, h]) #객체로 처리된 중심, 너비, 높이 좌표를 저장함
                confidences.append(float(confidence)) #객체로 처리된 신뢰도를 저장함
                class_ids.append(class_id) #객체가 어떤 클래스(사람, 랩탑 등)에 속하는지 ID를 저장함

    # 중복 인식된 객체 중 신뢰도가 높은 객체를 제외하고 나머지는 제거함 
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) #0.5 = 신뢰도 임계값 / 0.4 = 겹치는 정도를 나타내는 임계값

    # 탐지된 객체가 '정상'인지 확인하고, 부정행위(정상 객체가 아닌 것들) 가능성이 있는 경우 경고문 출력
    for i in range(len(boxes)):
        if i in indexes: #객체의 좌표와 크기 정보를 가져옴
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]]) #해당 객체의 레이블을 가져옴
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) #객체의 위치에 직사각형을 그림
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) #객체의 이름을 화면에 표시함 ex)laptop, person등

            if label not in normal_objects:
                print(f"부정행위 가능성 감지: {label}") #미리 정해둔 '정상'객체에 없는 객체면 부정행위 가능성으로 인지함

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) #객체의 위치에 직사각형을 그림
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) #객체의 이름을 화면에 표시함 ex)laptop, person등

    # 결과 영상 출력
    cv2.imshow("Frame", frame)

    #1ms동안 키보드 입력을 기다리고, 배속 조절, 'q'키를 입력하면 프로그램 종료
    if cv2.waitKey(int(1 / speed * 1000)) & 0xFF == ord('q'):
        break


cap.release() #비디오나 웹캠을 닫고 자원 해제
cv2.destroyAllWindows() #OpenCV로 열었던 모든 창 닫기기
