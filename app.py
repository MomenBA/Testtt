import streamlit as st
import av
import cv2
import mediapipe as mp
import numpy as np
import time
from gtts import gTTS
from io import BytesIO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from tensorflow.keras.models import load_model


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Sign Language Recognition", layout="wide")
st.title("🤟 Sign Language Recognition with gTTS Audio")


# Load Model
model = load_model("model_Tensorflow_new.keras")

# MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Labels
labels_dict = {i: chr(65 + i) for i in range(26)}
labels_dict.update({26: "Space", 27: "DEL", 28: "please", 29: "Hello"})

# Streaming audio output
audio_placeholder = st.empty()


# -------------------------------
# Function: Convert text to speech
# -------------------------------
def speak_text(text):
    tts = gTTS(text=text, lang="en")
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_placeholder.audio(audio_bytes.getvalue(), format="audio/mp3")


# -------------------------------
# Video transformer class
# -------------------------------
class SignLanguageTransformer(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=1)
        self.sign_string = ""
        self.current_letter = None
        self.letter_start_time = None
        self.letter_confirmation_time = 1  # seconds

    def predict_sign(self, frame):
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            x_, y_, data_aux = [], [], []

            for i in range(21):
                x_.append(hand_landmarks.landmark[i].x)
                y_.append(hand_landmarks.landmark[i].y)

            for i in range(21):
                data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                data_aux.append(hand_landmarks.landmark[i].y - min(y_))

            # Ensure length is 42
            if len(data_aux) < 42:
                data_aux += [0] * (42 - len(data_aux))

            data_aux = data_aux[:42]

            input_data = np.expand_dims(np.array(data_aux), axis=-1)
            input_data = np.expand_dims(input_data, axis=0)

            prediction = model.predict(input_data)
            predicted_index = np.argmax(prediction)
            predicted_char = labels_dict.get(predicted_index, "?")

            # Letter confirmation
            if predicted_char == self.current_letter:
                if time.time() - self.letter_start_time > self.letter_confirmation_time:
                    if self.current_letter == "DEL":
                        self.sign_string = self.sign_string[:-1]
                    elif self.current_letter == "Space":
                        self.sign_string += " "
                    else:
                        self.sign_string += self.current_letter

                    # 🔥 Call gTTS to speak the letter
                    speak_text(self.current_letter)

                    self.current_letter = None
            else:
                self.current_letter = predicted_char
                self.letter_start_time = time.time()

            # Draw prediction
            x1 = int(min(x_) * W)
            y1 = int(min(y_) * H)
            cv2.putText(frame, predicted_char, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

        return frame

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = self.predict_sign(img)

        cv2.putText(img, f"Current Text: {self.sign_string}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2)

        return img


# -------------------------------
# Camera Stream
# -------------------------------
st.subheader("📸 Start Camera")

webrtc_streamer(
    key="sign_lang",
    video_transformer_factory=SignLanguageTransformer,
    media_stream_constraints={"video": True, "audio": False"},
)

