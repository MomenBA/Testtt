import cv2
import mediapipe as mp
import numpy as np
import time
import warnings
import pyttsx3
from tensorflow.keras.models import load_model
import traceback

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load the model
model = load_model('model_Tensorflow_new.keras')

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Setup MediaPipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=1)

# Define the sign language labels
labels_dict = {i: chr(65 + i) for i in range(26)}
labels_dict.update({26: "Space", 27: "DEL", 28: "please", 29: "Hello"})

sign_string = ""
letter_confirmation_time = 1
current_letter = None
letter_start_time = None

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                try:
                    x_, y_ = [], []
                    data_aux = []
                    for i in range(len(hand_landmarks.landmark)):
                        x_.append(hand_landmarks.landmark[i].x)
                        y_.append(hand_landmarks.landmark[i].y)

                    for i in range(len(hand_landmarks.landmark)):
                        data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                        data_aux.append(hand_landmarks.landmark[i].y - min(y_))
                
                    # Ensure the length of data_aux is 42
                    if len(data_aux) < 42:
                        data_aux += [0] * (42 - len(data_aux))
                    elif len(data_aux) > 42:
                        data_aux = data_aux[:42]

                    input_data = np.expand_dims(np.array(data_aux), axis=-1)
                    input_data = np.expand_dims(input_data, axis=0)

                    prediction = model.predict(input_data)
                    predicted_index = np.argmax(prediction)
                    predicted_character = labels_dict.get(predicted_index, "?")

                    # Confirmation logic for letter detection
                    if predicted_character == current_letter:
                        if time.time() - letter_start_time > letter_confirmation_time:
                            if current_letter == "DEL":
                                sign_string = sign_string[:-1]
                            elif current_letter == "Space":
                                sign_string += " "
                            elif current_letter in ["please", "Hello"]:
                                sign_string += f" {current_letter} "
                            else:
                                sign_string += current_letter
                            current_letter = None
                    else:
                        current_letter = predicted_character
                        letter_start_time = time.time()

                    # Display predicted character on the frame
                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

                    # Speak the predicted character
                    if current_letter and time.time() - letter_start_time < letter_confirmation_time:
                        engine.say(predicted_character)
                        engine.runAndWait()

                except Exception as e:
                    print(f"Inner exception: {e}")
                    traceback.print_exc()

        # Display the current text on the frame
        cv2.putText(frame, f"Current Text: {sign_string}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Sign Language Recognition', frame)
        
        # Key press handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('d') and sign_string:
            sign_string = sign_string[:-1]
        elif key == ord('q'):
            print("Final sign string:", sign_string)
            break

    except Exception as e:
        print("Error:", e)
        traceback.print_exc()

cap.release()
cv2.destroyAllWindows()
