# -*- coding: utf-8 -*-
import time
import pyaudio
from collections import deque
import numpy as np

from config import Config as CFG  
from audio_processing import rms
import wavTranscriber

class VoiceRecognition:
    def __init__(self, dirName = "../deepspeech-0.6.1-models"): 
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=CFG.CHANNELS,
                                  rate=CFG.RATE,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=CFG.CHUNK_SIZE)

        output_graph, lm, trie = wavTranscriber.resolve_models(dirName)
        self.model = wavTranscriber.load_model(output_graph, lm, trie)  # Load output_graph, alpahbet, lm and trie

    def __call__(self):
        current = time.time()
        end = time.time() + CFG.VOICE_END_CHECK_TIME

        dummy = self.stream.read(CFG.CHUNK_SIZE)
        active_noise_list = deque([rms(dummy)])
        _, _ = wavTranscriber.stt(self.model[0], np.frombuffer(dummy, dtype=np.int16), CFG.RATE) # model loading & initialization 
        
        rec = []
        start_i = float("inf")

        print("Started...")
        
        try:
            while True:
                if current > end and start_i != float("inf"):
                    if start_i + CFG.MIN_VOICE_LENGTH < len(rec): self.stt(start_i, rec)
                    rec = []
                    start_i = float("inf")
                    end = time.time() + CFG.VOICE_END_CHECK_TIME 
                    
                frame = self.stream.read(CFG.CHUNK_SIZE)
                cur_sound_level = rms(frame)

                if start_i == float("inf") and len(rec) > CFG.MAX_RECORDING_BUF_SIZE: 
                    rec = rec[-CFG.SILENCE_PADING:]   # recording buffer size management 

                average_noise = sum(active_noise_list) / len(active_noise_list)
                
                if  cur_sound_level >= average_noise + CFG.VOICE_DETECTION_THRESHOLD:
                    end = time.time() + CFG.VOICE_END_CHECK_TIME
                    start_i = min(start_i, max(0,len(rec)-1))
                    #print("Sound is detected", cur_sound_level, " > ", average_noise)
                else:
                    if len(active_noise_list) >= CFG.MAX_ACTIVE_NOISE_LIST_SIZE: active_noise_list.popleft()    
                    active_noise_list.append(cur_sound_level)

                rec.append(frame)
                current = time.time()
                
        except KeyboardInterrupt:
            print("Stopped by Keyboard")

        finally:
            pass 

    def stt(self, start_i, rec):
        print("Speech to Text", start_i, len(rec))
        audio = np.frombuffer(np.array(rec[start_i-CFG.SILENCE_PADING:]).flatten(), dtype=np.int16)
        output, inference_time = wavTranscriber.stt(self.model[0], audio, CFG.RATE) 
        if len(output): print("Inference:", output, "(time elapsed:", inference_time, ")")

def main():
    vr = VoiceRecognition()
    vr()

if __name__ == "__main__":
    main()

# end of file 