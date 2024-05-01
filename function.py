import cv2
import numpy as np
import dlib
import pandas as pd
from math import hypot
from datetime import datetime, timedelta

def detect_cheating(socketio):
    # Initialize the video capture object
    cap = cv2.VideoCapture(0)

    # Initialize the face detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r"C:\Users\Dina\hu\shape_predictor_68_face_landmarks.dat")
    font = cv2.FONT_HERSHEY_PLAIN

    def midpoint(p1, p2):
        return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

    def get_blinking_ratio(eye_points, facial_landmarks):
        left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
        right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
        center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
        center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

        hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
        ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

        ratio = hor_line_length / ver_line_length
        return ratio

    def get_gaze_ratio(eye_points, facial_landmarks, frame):
        left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                    (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                    (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                    (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                    (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                    (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)

        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [left_eye_region], True, 255, 2)
        cv2.fillPoly(mask, [left_eye_region], 255)
        eye = cv2.bitwise_and(gray, gray, mask=mask)

        min_x = np.min(left_eye_region[:, 0])
        max_x = np.max(left_eye_region[:, 0])
        min_y = np.min(left_eye_region[:, 1])
        max_y = np.max(left_eye_region[:, 1])

        gray_eye = eye[min_y: max_y, min_x: max_x]
        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
        
        if threshold_eye is None: 
            return None, None
        
        height, width = threshold_eye.shape
        left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
        left_side_white = cv2.countNonZero(left_side_threshold)

        right_side_threshold = threshold_eye[0: height, int(width / 2): width]
        right_side_white = cv2.countNonZero(right_side_threshold)

        if left_side_white == 0:
            gaze_ratio = 0.5
        elif right_side_white == 0:
            gaze_ratio = 2.5
        else:
            gaze_ratio = left_side_white / right_side_white

        return gaze_ratio, (min_x, min_y, max_x, max_y)

    left_counter = 0  
    right_counter = 0   
    counter = 0
    multi = 0
    results = []

    recording_duration = 10  
    recording_fps = 20  

    out = None

    while True:
        new_frame = np.zeros((500, 500, 3), np.uint8)
        ret, frame = cap.read()
        if not ret:
            break
            
        cv2.putText(frame, "you need to look center most of the time or it will considered as cheating",  (20, 50), font, 1, (255, 0, 0), 2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)

        if len(faces) > 1:
            multi += 1
            if multi >= 30:
                print("Multiple faces detected!")
                socketio.emit('update_detection_results', "multiple!")

                multi = 0
                results.append(("multiple faces detected!", 0, 0, datetime.now()))
                recording_start_time = datetime.now()
                recording_end_time = recording_start_time + timedelta(seconds=recording_duration)
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(f'cheating_clip_multiple_{recording_start_time.strftime("%Y%m%d_%H%M%S")}.avi', fourcc, recording_fps, (frame.shape[1], frame.shape[0]))


           
        elif len(faces) > 0:
            for face in faces:
                landmarks = predictor(gray, face)

                left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
                right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
                blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

                if blinking_ratio > 5.7:
                    cv2.putText(frame, "BLINKING", (50, 150), font, 7, (255, 0, 0), thickness=5)

                gaze_ratio_left_eye, left_eye_coords = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks, frame)
                gaze_ratio_right_eye, right_eye_coords = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks, frame)
                gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2

                if gaze_ratio < 0.5:
                    cv2.putText(frame, "RIGHT", (50, 100), font, 2, (0, 0, 255), 3)
                    new_frame[:] = (0, 0, 255)
                    right_counter += 1
                    left_counter = 0
                    if right_counter >= 35:
                        print("Student is cheating by looking RIGHT!")
                        socketio.emit('update_detection_results', "Student is cheating by looking RIGHT!")
                        right_counter = 0
                        results.append(("RIGHT", right_eye_coords, left_eye_coords, datetime.now()))
                        recording_start_time = datetime.now()
                        recording_end_time = recording_start_time + timedelta(seconds=recording_duration)
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        out = cv2.VideoWriter(f'cheating_clip_right_{recording_start_time.strftime("%Y%m%d_%H%M%S")}.avi', fourcc, recording_fps, (frame.shape[1], frame.shape[0]))

                elif 0.5 < gaze_ratio < 2:
                    cv2.putText(frame, "CENTER", (50, 100), font, 2, (0, 0, 255), 3)
                    left_counter = 0
                    right_counter = 0
                    
                elif gaze_ratio > 2:
                    cv2.putText(frame, "LEFT", (50, 100), font, 2, (0, 0, 255), 3)
                    new_frame[:] = (255, 0, 0)
                    left_counter += 1
                    right_counter = 0
                    if left_counter >= 35:
                        print("Student is cheating by looking LEFT!")
                        socketio.emit('update_detection_results', "Student is cheating by looking LEFT!")

                        left_counter = 0
                        results.append(("LEFT", right_eye_coords, left_eye_coords, datetime.now()))
                        recording_start_time = datetime.now()
                        recording_end_time = recording_start_time + timedelta(seconds=recording_duration)
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        out = cv2.VideoWriter(f'cheating_clip_left_{recording_start_time.strftime("%Y%m%d_%H%M%S")}.avi', fourcc, recording_fps, (frame.shape[1], frame.shape[0]))
        else:
            cv2.putText(frame, "No face detected!", (50, 100), font, 2, (0, 0, 255), 3)
            counter += 1
            if counter >= 30:
                print("No face detected!")
                counter = 0
                results.append(("No face detected!", 0, 0, datetime.now()))
                recording_start_time = datetime.now()
                recording_end_time = recording_start_time + timedelta(seconds=recording_duration)
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(f'cheating_clip_no_face_{recording_start_time.strftime("%Y%m%d_%H%M%S")}.avi', fourcc, recording_fps, (frame.shape[1], frame.shape[0]))

        cv2.imshow("Frame", frame)
        cv2.imshow("New frame", new_frame)
       
        if out is not None:
            if datetime.now() <= recording_end_time:
                out.write(frame)
            else:
                out.release()

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    results_df = pd.DataFrame(results, columns=["Direction", "Right Eye Coordinates", "Left Eye Coordinates", "Time"])

    results_df.to_excel('cheating_results.xlsx', index=False)
    return results_df

# Call the function to execute the code
# detect_cheating()
