from flask import Flask, render_template, request, redirect, url_for
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Create the screenshots directory if it doesn't exist.
if not os.path.exists('screenshots'):
    os.makedirs('screenshots')

# Create the uploads directory if it doesn't exist.
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a = np.array(a)  # First point.
    b = np.array(b)  # Mid point.
    c = np.array(c)  # End point.

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle


def get_coordinates(landmarks, frame_shape):
    """Get coordinates of key landmarks."""
    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame_shape[1],
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame_shape[0]]
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * frame_shape[1],
           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * frame_shape[0]]
    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * frame_shape[1],
             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * frame_shape[0]]
    head = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * frame_shape[1],
            landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * frame_shape[0]]
    return shoulder, hip, ankle, head


def visualize_angle(image, angle, position, color=(255, 255, 255)):
    """Visualize the angle on the image."""
    cv2.putText(image, str(angle),
                tuple(np.multiply(position, [1, 1]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)


def save_screenshot(image, prefix='good_plank_position'):
    """Save a screenshot with a timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cv2.imwrite(f'screenshots/{prefix}_{timestamp}.png', image)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        process_video(file_path)
        return redirect(url_for('index'))


@app.route('/capture')
def capture():
    process_video(0)  # 0 for webcam
    return redirect(url_for('index'))


def process_video(file_name):
    try:
        cap = cv2.VideoCapture(file_name)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file or device: {file_name}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the BGR image to RGB.
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Process the image and find pose landmarks.
            results = pose.process(image)

            # Convert the image back to BGR.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                shoulder, hip, ankle, head = get_coordinates(landmarks, frame.shape)

                # Calculate angles.
                angle = calculate_angle(shoulder, hip, ankle)
                headshoulderhipangle = calculate_angle(head, shoulder, hip)

                # Visualize angles.
                visualize_angle(image, angle, hip)
                visualize_angle(image, headshoulderhipangle, shoulder)

                # Determine plank quality.
                if 160 < angle < 180 and 175 < headshoulderhipangle < 180:
                    cv2.putText(image, 'Good Plank', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.line(image, tuple(np.multiply(shoulder, [1, 1]).astype(int)),
                             tuple(np.multiply(hip, [1, 1]).astype(int)), (0, 255, 0), 3)
                    cv2.line(image, tuple(np.multiply(hip, [1, 1]).astype(int)),
                             tuple(np.multiply(ankle, [1, 1]).astype(int)), (0, 255, 0), 3)

                    #MZ add - now draw line for Head--> shoulder --> hip
                    cv2.putText(image, 'Head, Shoulder and Hip is aligned', (850, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.line(image, tuple(np.multiply(head, [1, 1]).astype(int)),
                             tuple(np.multiply(shoulder, [1, 1]).astype(int)), (0, 255, 0), 3)
                    cv2.line(image, tuple(np.multiply(shoulder, [1, 1]).astype(int)),
                             tuple(np.multiply(hip, [1, 1]).astype(int)), (0, 255, 0), 3)

                    cv2.imshow('Plank Position', image)
                    save_screenshot(image)
                elif angle > 180 or headshoulderhipangle < 175:
                    cv2.putText(image, 'Bad Plank', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.line(image, tuple(np.multiply(shoulder, [1, 1]).astype(int)),
                             tuple(np.multiply(hip, [1, 1]).astype(int)), (0, 0, 255), 3)
                    cv2.line(image, tuple(np.multiply(hip, [1, 1]).astype(int)),
                             tuple(np.multiply(ankle, [1, 1]).astype(int)), (0, 0, 255), 3)
                else:
                    cv2.putText(image, 'Adjust Plank', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                cv2.LINE_AA)

            # Display the frame.
            cv2.imshow('Plank Position', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    except cv2.error as e:
        print(f"OpenCV error: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    app.run(debug=True)