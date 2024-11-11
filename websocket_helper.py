import os

import cv2
from keras.models import load_model  # type: ignore
import logging
from mediapipe.python.solutions.holistic import Holistic  # type: ignore
import numpy as np
import tensorflow as tf  # type: ignore

from constants import (
    MAX_LENGTH_FRAMES,
    MODELS_PATH,
    MODEL_NAME,
)


# Cargar el modelo LSTM
model_path = os.path.join(MODELS_PATH, MODEL_NAME)
lstm_model = load_model(model_path)


# Variables para mantener el estado del websocket
class State:
    def __init__(self, threshold=0.7):
        self.count_frame = 0
        self.repe_sent = 1
        self.kp_sequence = []
        self.sentence = []
        self.holistic = None
        self.threshold = threshold

    def initialize_holistic(self):
        if self.holistic is None:
            self.holistic = Holistic()

    def cleanup_holistic(self):
        if self.holistic is not None:
            self.holistic.close()
            self.holistic = None


def process_frame_data(frame_data):
    """
    Convierte los datos del frame recibidos vía WebSocket a una imagen OpenCV
    """
    # Decodificar la imagen base64 a numpy array
    import base64
    import numpy as np

    # Eliminar el encabezado de data URL si existe
    if "," in frame_data:
        frame_data = frame_data.split(",")[1]

    # Decodificar base64 a bytes
    frame_bytes = base64.b64decode(frame_data)

    # Convertir bytes a numpy array
    np_arr = np.frombuffer(frame_bytes, np.uint8)

    # Decodificar array a imagen
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    return frame


def predict_sequence(sequence):
    """
    Realiza la predicción fuera de cualquier función tf.function
    """
    try:
        sequence_for_prediction = np.expand_dims(
            sequence[-MAX_LENGTH_FRAMES:],
            axis=0,
        )
        with tf.device("/CPU:0"):  # Cambiar a "/GPU:0" si se desea usar GPU
            prediction = lstm_model.predict(sequence_for_prediction)
            logging.info(f"Prediction: {prediction}")
            return prediction[0]
    except Exception as e:
        logging.error(f"Error en predict_sequence: {str(e)}", exc_info=True)
        return None
