import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose =  mp.solutions.pose

def calculate_angle(a, b, c):
  a = np.array(a)
  b = np.array(b)
  c = np.array(c)

  radias = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
  angle = np.abs(radias * 180.0/np.pi)

  if angle > 180.0:
    angle = 360 - angle

  return angle

# Setup mediapipe instance
cap = cv2.VideoCapture(0)
counter = 0
stage_left = None
stage_right = None
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    ret, frame = cap.read()

    # Recolor image
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Make detection
    results = pose.process(image)

    # Recolor back to RGB
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Extract landmarks
    try:
      landmarks = results.pose_landmarks.landmark

      # Get coordinates
      shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
      elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
      wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

      shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
      elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
      wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

      # Calculate angle
      angle_left = calculate_angle(shoulder_left, elbow_left, wrist_left)
      angle_right = calculate_angle(shoulder_right, elbow_right, wrist_right)



      # Visualize angle
      cv2.putText(image, str(angle_left),
                  tuple(np.multiply(elbow_left, [640, 480]).astype(int)),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                  )
      cv2.putText(image, str(angle_right),
                  tuple(np.multiply(elbow_right, [640, 480]).astype(int)),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                  )

      # Curl counter logic
      if angle_left > 160:
        stage_left = "down"
      elif angle_left < 70 and stage_left == 'down':
        stage_left = "up"
        counter += 1
        print(counter)

      if angle_right > 160:
        stage_right = "down"
      elif angle_right < 70 and stage_right == 'down':
        stage_right = "up"
        counter += 1
        print(counter)



    except:
      pass



    # Render detection
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


    cv2.imshow('Mediapipe Feed', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()





