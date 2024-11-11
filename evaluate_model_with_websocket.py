import os

import atexit
from flask import Flask
from flask_cors import CORS  # type: ignore
from flask_socketio import SocketIO, emit  # type: ignore
from keras.models import load_model  # type: ignore
import logging
import numpy as np

from helpers import (
    extract_keypoints,
    format_sentences,
    get_actions,
    mediapipe_detection,
    there_hand,
)
from constants import (
    DATA_PATH,
    MAX_LENGTH_FRAMES,
    MIN_LENGTH_FRAMES,
    MODELS_PATH,
    MODEL_NAME,
)
from websocket_helper import State, process_frame_data, predict_sequence


app = Flask(__name__)
# En producción, cambia esto a tu dominio específico
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="gevent")

# Configurar CORS
CORS(
    app,
    resources={
        r"/*": {
            # En producción, cambia esto a tu dominio específico
            "origins": "*",
            "allow_headers": ["Content-Type"],
            "expose_headers": ["Content-Type"],
            "methods": ["GET", "POST", "OPTIONS"],
            "max_age": 1728000,
        }
    },
)

# Cargar el modelo LSTM
model_path = os.path.join(MODELS_PATH, MODEL_NAME)
lstm_model = load_model(model_path)
actions = get_actions(DATA_PATH)


state = State(threshold=0.7)


@app.route("/")
def index():
    return "Conexión establecida con el servidor WebSocket"


@socketio.on("connect")
def handle_connect():
    state.initialize_holistic()
    emit("connected")


@socketio.on("disconnect")
def handle_disconnect():
    state.cleanup_holistic()


@socketio.on("frame")
def handle_frame(frame_data):
    """
    Maneja cada frame recibido por WebSocket
    """
    try:
        # Procesar el frame recibido
        frame = process_frame_data(frame_data)

        if state.holistic is None:
            state.initialize_holistic()

        # Usar mediapipe para detectar los puntos clave
        _, results = mediapipe_detection(frame, state.holistic)

        # Extraer keypoints y añadirlos a la secuencia
        keypoints = extract_keypoints(results)
        state.kp_sequence.append(keypoints)

        # Verificar si hay una mano presente y procesar la secuencia
        if len(state.kp_sequence) > MAX_LENGTH_FRAMES and there_hand(results):
            state.count_frame += 1
        else:
            if state.count_frame >= MIN_LENGTH_FRAMES:
                # Predecir el gesto
                res = predict_sequence(state.kp_sequence)

                if res is not None and res[np.argmax(res)] > state.threshold:
                    sent = actions[np.argmax(res)]
                    state.sentence.insert(0, sent)
                    state.sentence, state.repe_sent = format_sentences(
                        sent, state.sentence, state.repe_sent
                    )

                    # Emitir la sentencia actual al cliente
                    emit(
                        "prediction",
                        {
                            "sentence": state.sentence,
                            "confidence": float(res[np.argmax(res)]),
                        },
                    )

                # Reiniciar el estado de la secuencia
                state.count_frame = 0
                state.kp_sequence = []

    except Exception as e:
        logging.error(f"Error al procesar el frame: {str(e)}", exc_info=True)
        emit("error", {"message": str(e)})


@socketio.on("reset")
def handle_reset():
    """Reinicia el estado de la aplicación"""
    state.count_frame = 0
    state.repe_sent = 1
    state.kp_sequence = []
    state.sentence = []
    emit("reset_confirmed")


# Asegurarse de limpiar los recursos al cerrar la aplicación
def cleanup():
    state.cleanup_holistic()


atexit.register(cleanup)


if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)
