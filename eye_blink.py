import cv2
import numpy as np
import mediapipe as mp
import time

class EyeBlinkDetector:
    def __init__(self, face_mesh=None):

        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

        if face_mesh is not None:
            self.face_mesh = face_mesh
            self._owns_mesh = False
        else:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )
            self._owns_mesh = True

        # Eye landmark points
        self.l_eyelids = [362, 385, 387, 263, 373, 380]
        self.r_eyelids = [33, 160, 158, 133, 153, 144]

        self.ear_threshold = 0.18
        self.counter = 0
        self.total_blinks = 0

        # Blink timing
        self.blink_times = []
        self.time_window = 5  # seconds
        self.blink_threshold = 3

        self.blink_cooldown = 0
        self.cooldown_frames = 3

        self.eye_flag = 0

    def reset(self):
        """Reset detector"""
        self.counter = 0
        self.total_blinks = 0
        self.blink_cooldown = 0
        self.eye_flag = 0
        self.blink_times = []

    def eye_aspect_ratio(self, eye):

        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])

        if C == 0:
            return 0.0

        return (A + B) / (2.0 * C)

    def process_frame(self, image):

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        ear = 0.0

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                h, w, _ = image.shape

                landmarks = np.array([
                    [int(lm.x * w), int(lm.y * h)]
                    for lm in face_landmarks.landmark
                ])

                left_eye = landmarks[self.l_eyelids]
                right_eye = landmarks[self.r_eyelids]

                left_ear = self.eye_aspect_ratio(left_eye)
                right_ear = self.eye_aspect_ratio(right_eye)

                ear = (left_ear + right_ear) / 2.0

                # Blink detection
                if self.blink_cooldown > 0:

                    self.blink_cooldown -= 1
                    self.counter = 0

                elif ear < self.ear_threshold:

                    self.counter += 1

                else:

                    if self.counter >= 1:

                        self.total_blinks += 1
                        self.blink_times.append(time.time())

                        self.blink_cooldown = self.cooldown_frames

                    self.counter = 0

                # Remove blinks older than 5 seconds
                current_time = time.time()

                self.blink_times = [
                    t for t in self.blink_times
                    if current_time - t <= self.time_window
                ]

                # Activate flag once and keep it active
                if len(self.blink_times) >= self.blink_threshold and self.eye_flag == 0:
                    self.eye_flag = 1

                # Draw eye landmarks
                self.mp_drawing.draw_landmarks(
                    image,
                    face_landmarks,
                    self.mp_face_mesh.FACEMESH_LEFT_EYE,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(0,255,0), thickness=2
                    )
                )

                self.mp_drawing.draw_landmarks(
                    image,
                    face_landmarks,
                    self.mp_face_mesh.FACEMESH_RIGHT_EYE,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(0,255,0), thickness=2
                    )
                )

        # Display metrics
        cv2.putText(image, f"EAR: {ear:.2f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        cv2.putText(image, f"Blinks(5s): {len(self.blink_times)} | Eye Flag: {self.eye_flag}",
                    (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,255),
                    2)

        return {
            'ear': ear,
            'blinks': self.total_blinks,
            'eye_flag': self.eye_flag,
            'image': image
        }