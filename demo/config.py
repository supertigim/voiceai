# -*- coding: utf-8 -*-
import pyaudio

class Config:

    FORMAT = pyaudio.paInt16
    # Network/VAD rate-space
    RATE_PROCESS = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50

    VAD_AGGRESSIVE = 3

    BEAM_WIDTH = 500
    DEFAULT_SAMPLE_RATE = 16000
    LM_ALPHA = 0.75
    LM_BETA = 1.85

    S_WIDTH = 2
    SHORT_NORMALIZE = (1.0/32768.0)
    VOICE_DETECTION_THRESHOLD = 10

    BROKER = '10.10.10.235:9092'
    TOPIC_SPEECH_TTS = 'speech.tts'

# end of file