# import multiprocessing as mp

# list1 = []

# def foo(list1, q):
#     print("qqq")
#     q.put('hello')
#     list1.append('hello')

# if __name__ == '__main__':
#     # mp.set_start_method('spawn')
#     q = mp.Queue()
#     p = mp.Process(target=foo, args=(list1,q))
#     p.start()
#     # print(q.get(False))
#     print("abc")
#     p.join()
#     print("1233")
#     print(q.get_nowait())

#     # time.sleep(5)
#     # print(list1)


import pyaudio
import numpy as np
from queue import Queue
import atexit
import sounddevice as sd
import noisereduce as nr
from collections import Counter
import requests


import asyncio
import concurrent.futures
import threading
import time
import multiprocessing as mp
from multiprocessing import Pool

import time

# Constants for audio stream configuration
FORMAT = pyaudio.paInt16  # Sample format (16-bit)
CHANNELS = 1             # Number of audio channels (1 for mono, 2 for stereo)
# RATE = 44100             # Sample rate (samples per second)
# CHUNK_SIZE = 44100        # Number of audio frames per buffer
CHUNK_SIZE = 16000        # Number of audio frames per buffer
SAMPLING_RATE = 16000     # Sample rate (samples per second)

DETECT_THRESHOLD = 1000   # Threshold to detect a clap
DETECT_DURATION = 0.1    # Time in seconds to detect a clap
RECORD_DURATION = 1.5      # Time in seconds to record audio after clap is detected
MAX_RECORD_DURATION = 20 # Maximum time in seconds to record audio


import whisper

model = whisper.load_model("small")
# model = whisper.load_model("medium")


def inference(audio, text):
# async def inference(audio):
    if type(audio) == str:
        audio = whisper.load_audio(audio)
    print(type(audio), audio.dtype, audio.shape)
    audio = whisper.pad_or_trim(audio)
    
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    
    # _, probs = model.detect_language(mel)
    
    options = whisper.DecodingOptions(fp16 = False)
    result = whisper.decode(model, mel, options)
    
    text = result.text
    print(result.text)
    return result.text


def text_filter(text):
    
    char_count = Counter(text)
    # print(char_count)
    n_chars = len(char_count.keys())
    print("n_chars = ", n_chars)
    flag_n_chars_too_little = n_chars < 4
    # flag_any_char_too_many = np.all(np.array(list(char_count.values())) > 20)

    # return flag_n_chars_too_little and flag_any_char_too_many
    return flag_n_chars_too_little 

from pynput import keyboard

def on_press(key):
    if key == keyboard.Key.esc:
        # Stop listener
        return False

def on_release(key):
    pass

# ...or, in a non-blocking fashion:
listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()

async def stt_func(audio_data):
    text = await inference(audio_data)
    print(text)

async def playback(audio_data, samplerate):
# def playback(audio_data, samplerate):
    sd.play(audio_data.astype(float)/4096, samplerate=samplerate)
    # raise Exception("not consumed")

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=SAMPLING_RATE, input=True, frames_per_buffer=CHUNK_SIZE)

print("start recording...")

while True:
    data = stream.read(int(DETECT_DURATION * SAMPLING_RATE), exception_on_overflow=False)
    audio_data_in = np.frombuffer(data, dtype=np.int16)

    # print(audio_data_in[:20])
    # print("detecting:", np.max(audio_data_in))

    if np.max(audio_data_in) < DETECT_THRESHOLD:
        continue
    
    print(np.max(audio_data_in))

    audio_data = audio_data_in
    while np.max(audio_data_in) > DETECT_THRESHOLD:
        data = stream.read(int(RECORD_DURATION * SAMPLING_RATE), exception_on_overflow=False)
        audio_data_in = np.frombuffer(data, dtype=np.int16)
        audio_data = np.concatenate((audio_data, audio_data_in))
        if(len(audio_data) > SAMPLING_RATE*30):
            break

    # data = stream.read(int(RECORD_DURATION * SAMPLING_RATE))
    # audio_data_record = np.frombuffer(data, dtype=np.int16)

    # audio_data = np.concatenate((audio_data_in, audio_data_record))
    # await playback(audio_data, SAMPLING_RATE)
    asyncio.run(playback(audio_data.astype(np.float32), SAMPLING_RATE))
    # asyncio.create_task(playback(audio_data, SAMPLING_RATE))
    # await playback(audio_data, SAMPLING_RATE)
    # audio_data = audio_data_in

    
    print("audio_data min,max = ", np.min(audio_data), np.max(audio_data), audio_data.shape)



    # # Reduce noise from the audio using noisereduce
    # audio_noise_reduced = nr.reduce_noise(y=audio_data, sr=SAMPLING_RATE)
    
    # sd.play(audio_data, samplerate=SAMPLING_RATE)

    # print("audio_noise_reduced min,max = ", np.min(audio_noise_reduced), np.max(audio_noise_reduced), audio_noise_reduced.shape)

    # await stt_func(audio_data.astype(np.float32))
    
    # text = inference(audio_noise_reduced)
    # text = inference(audio_noise_reduced.astype(np.float32))
    # text = inference(audio_data.astype(np.float32))
    
    print("inference...")
    text = ""
    print(inference(audio_data.astype(np.float32), text))

    # p = mp.Process(target=inference, args=(audio_data.astype(np.float32), text))
    # p.start()
    # p.join()


    # flag_skip = text_filter(text)

    # if flag_skip:
    #     print("skip +: ", text)
    #     continue

    # print(text)
    

    stream.stop_stream()
    stream.close()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=SAMPLING_RATE, input=True, frames_per_buffer=CHUNK_SIZE)


    if not listener.running:
        break