# Đoạn code này sử dụng thuật toán nhận diện khuôn mặt và nhận dạng khuôn mặt sử dụng thư viện face_recognition và OpenCV
import os, sys
import face_recognition
import cv2
import numpy as np
import math

# Tính toán độ tin cậy của khuôn mặt dựa trên khoảng cách khuôn mặt và ngưỡng khớp khuôn mặt cho trước
def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'
    
class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()
        
    # Mã hóa các khuôn mặt đã biết từ các hình ảnh trong thư mục 'faces'
    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f'faces/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)

        print(self.known_face_names)

    
    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)
        global counter
        counter = 0

        while True:
            ret, frame = video_capture.read()

            if self.process_current_frame:
                if counter % 25 == 0:
                    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
                    rgb_small_frame = small_frame[:, :, ::-1]
                    
                    # Xác định vị trí và mã hóa khuôn mặt
                    self.face_locations = face_recognition.face_locations(rgb_small_frame)
                    self.face_encodings = face_recognition.face_encodings(small_frame, self.face_locations)

                    self.face_names = []

                    # So sánh mã hóa khuôn mặt với các mã đã biết để xác định xem có sự khớp hay không.
                    for face_encoding in self.face_encodings:
                        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                        name = 'Unknown'
                        confidence = 'Unknown'

                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)

                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                            confidence = face_confidence(face_distances[best_match_index])

                        self.face_names.append(f'{name}({confidence})')
                counter += 1

            self.process_current_frame = not self.process_current_frame

            # Vẽ bounding box cho khuôn mặt và hiển thị thông tin
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)
                cv2.rectangle(frame, (left, bottom - 20), (right, bottom), (0,0,255), -1)
                cv2.putText(frame, name, (left+6, bottom-6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1)

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1)==ord('q'):
                break
                
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()














        
