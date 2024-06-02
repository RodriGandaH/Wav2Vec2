from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import sounddevice as sd
import numpy as np
import torch

# Cargar el modelo y el procesador de Wav2Vec 2.0
processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-spanish")
model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-spanish")


buffer = []

# Definir una función de callback para procesar el audio del micrófono
def callback(indata, frames, time, status):
    buffer.extend(indata[:, 0])


with sd.InputStream(callback=callback, channels=1, samplerate=16000):
    while True:
        print("Escuchando...")
        # Dormir durante 4 segundo para grabar el audio
        sd.sleep(4000)
        # Convertir el buffer a un array de numpy
        audio = np.array(buffer)

        # Procesar el audio grabado con Wav2Vec 2.0
        input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])
        print(transcription)
        buffer.clear()


