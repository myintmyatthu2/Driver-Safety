import cv2
import numpy as np
import pygame
from utils import crop_eye_from_face, load_model, vlog
import config

FACE_CASCADE_PATH = r"D:\projects\driver_safety\haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = r"D:\projects\driver_safety\haarcascade_eye.xml"

def eyes_center_offset(face_w, left_eye_box, right_eye_box):
    centers = []
    for box in (left_eye_box, right_eye_box):
        if box is None:
            centers.append(None)
            continue
        ex, ey, ew, eh = box
        cx = ex + ew / 2.0
        centers.append(cx / float(face_w))
    return centers

def run_detection():
    model = load_model(config.MODEL_PATH)

    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        vlog(f"Camera index {config.CAMERA_INDEX} failed, trying 1...")
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise SystemExit("Cannot open webcam. Check CAMERA_INDEX or connect a camera.")

    look_away_counter = 0
    no_eye_counter = 0

    # Initialize pygame mixer for alarm sound
    pygame.mixer.init()
    alarm_sound = pygame.mixer.Sound(config.ALARM_WAV)
    alarm_active = False  # Flag to track if alarm is currently playing

    while True:
        ret, frame = cap.read()
        if not ret:
            vlog("Frame not read from camera; breaking.")
            break

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        status_text = "No face"
        open_count = 0
        closed_count = 0

        if len(faces) > 0:
            faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
            x, y, w, h = faces[0]
            face_gray = gray[y:y+h, x:x+w]
            face_color = display[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

            filtered_eyes = []
            for (ex, ey, ew, eh) in eyes:
                if ew < 20 or eh < 20 or ew > w * 0.5 or eh > h * 0.5:
                    continue
                if ey > h * 0.5:
                    continue
                filtered_eyes.append((ex, ey, ew, eh))
            eyes = filtered_eyes

            left_eye = None
            right_eye = None
            if len(eyes) >= 2:
                eyes = sorted(eyes, key=lambda e: e[0])
                left_eye, right_eye = eyes[0], eyes[1]
            elif len(eyes) == 1:
                left_eye = eyes[0]

            for eye_box in (left_eye, right_eye):
                if eye_box is None:
                    # Missing eye, skip prediction
                    continue

                ex, ey, ew, eh = eye_box
                eye_crop = crop_eye_from_face(face_color, eye_box, target_size=config.IMAGE_SIZE)

                if eye_crop is None:
                    # Failed crop treated as closed
                    closed_count += 1
                    continue

                inp = np.expand_dims(eye_crop, axis=0)
                pred = model.predict(inp, verbose=0)[0][0]
                is_open = pred > config.PREDICTION_THRESHOLD

                if is_open:
                    open_count += 1
                else:
                    closed_count += 1

                color = (0, 255, 0) if is_open else (0, 0, 255)
                cv2.rectangle(face_color, (ex, ey), (ex+ew, ey+eh), color, 2)
                cv2.putText(face_color, f"{'Open' if is_open else 'Closed'} {pred:.2f}",
                            (ex, max(ey-6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            # Fix counting logic for missing eyes:
            if len(eyes) == 2:
                # Both eyes detected: counts are accurate
                pass
            elif len(eyes) == 1:
                if closed_count == 1:
                    # One eye detected and closed, missing eye unknown, do NOT assume missing eye is closed
                    pass
                else:
                    # One eye detected and open, so no closed eyes
                    closed_count = 0
            else:
                # No eyes detected - treat as both closed
                closed_count = 2

            # Check alarms
            sleep_alert = (closed_count == 2)
            look_away_alert = False
            eye_centers = eyes_center_offset(w, left_eye, right_eye)
            if eye_centers[0] is not None and eye_centers[1] is not None:
                avg = (eye_centers[0] + eye_centers[1]) / 2.0
                if avg < 0.35 or avg > 0.65:
                    look_away_alert = True
            elif eye_centers[0] is not None and eye_centers[1] is None:
                c = eye_centers[0]
                if c < 0.3 or c > 0.7:
                    look_away_alert = True

            if look_away_alert:
                look_away_counter += 1
            else:
                look_away_counter = max(0, look_away_counter - 1)

            no_eyes_alert = (len(eyes) == 0)
            if no_eyes_alert:
                no_eye_counter += 1
            else:
                no_eye_counter = 0

            look_away_alert = look_away_counter >= config.LOOK_AWAY_FRAMES_THRESHOLD
            no_eyes_alert = no_eye_counter >= config.NO_EYE_FRAMES_THRESHOLD

            status_text = f"Open:{open_count} Closed:{closed_count} | LookCtr:{look_away_counter}"
            cv2.rectangle(display, (x, y), (x+w, y+h), (255, 0, 0), 2)
        else:
            look_away_counter = 0
            no_eye_counter += 1
            status_text = "No face detected"
            sleep_alert = False
            look_away_alert = False
            no_eyes_alert = False

        # Alarm sound control: play alarm if any alert active, else stop alarm
        if sleep_alert or look_away_alert or no_eyes_alert:
            if not alarm_active:
                alarm_sound.play(loops=-1)  # Loop alarm sound forever
                alarm_active = True
        else:
            if alarm_active:
                alarm_sound.stop()
                alarm_active = False

        # Display alerts on screen
        if sleep_alert:
            cv2.putText(display, "SLEEP ALERT!!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            vlog("ALERT: Both eyes closed - alarm triggered immediately.")
        if look_away_alert:
            cv2.putText(display, "LOOK AWAY ALERT!!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
            vlog("ALERT: Look-away threshold reached.")
        if no_eyes_alert:
            cv2.putText(display, "NO EYES FOUND!!", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            vlog("ALERT: No eyes found for many frames.")

        cv2.putText(display, status_text, (10, display.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Driver Safety System", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    # Make sure alarm sound is stopped when exiting
    if alarm_active:
        alarm_sound.stop()

if __name__ == "__main__":
    run_detection()
