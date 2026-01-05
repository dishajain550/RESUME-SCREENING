import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH = "model_files/model.h5"
TOKENIZER_PATH = "model_files/tokenizer.pkl"

MAX_LEN = 20   # MUST match Colab

class ResumeModel:
    def __init__(self):
        self.model = load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, "rb") as f:
            self.tokenizer = pickle.load(f)

    def predict_score(self, resume_text, career_text, jd_text):
        r_seq = pad_sequences(
            self.tokenizer.texts_to_sequences([resume_text]),
            maxlen=MAX_LEN
        )
        c_seq = pad_sequences(
            self.tokenizer.texts_to_sequences([career_text]),
            maxlen=MAX_LEN
        )
        j_seq = pad_sequences(
            self.tokenizer.texts_to_sequences([jd_text]),
            maxlen=MAX_LEN
        )

        score = self.model.predict([r_seq, c_seq, j_seq], verbose=0)[0][0]
        return float(score)
