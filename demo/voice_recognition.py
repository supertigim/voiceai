# -*- coding:utf-8 -*-

import time
from datetime import datetime
import threading, collections, queue, os, os.path
import deepspeech
import numpy as np
import pyaudio
import wave
import webrtcvad
from halo import Halo
from scipy import signal
#from confluent_kafka import Producer

from config import Config as cfg
from audio_processing import rms

class Audio(object):
    
    def __init__(self, callback=None, device=None, input_rate=cfg.RATE_PROCESS, file=None):
        def proxy_callback(in_data, frame_count, time_info, status):
            if self.chunk is not None:
                in_data = self.wf.readframes(self.chunk)
            callback(in_data)
            return (None, pyaudio.paContinue)
        if callback is None: callback = lambda in_data: self.buffer_queue.put(in_data)
        self.buffer_queue = queue.Queue()
        self.device = device
        self.input_rate = input_rate
        self.sample_rate = cfg.RATE_PROCESS
        self.block_size = int(cfg.RATE_PROCESS / float(cfg.BLOCKS_PER_SECOND))
        self.block_size_input = int(self.input_rate / float(cfg.BLOCKS_PER_SECOND))
        self.pa = pyaudio.PyAudio()
        #frame_duration_ms = property(lambda self: 1000 * self.block_size // self.sample_rate)
        self.frame_duration_ms = 1000 * self.block_size // self.sample_rate

        kwargs = {
            'format': cfg.FORMAT,
            'channels': cfg.CHANNELS,
            'rate': self.input_rate,
            'input': True,
            'frames_per_buffer': self.block_size_input,
            'stream_callback': proxy_callback,
        }

        self.chunk = None
        if file is not None:
            self.chunk = 320
            self.wf = wave.open(file, 'rb')

        self.stream = self.pa.open(**kwargs)
        self.stream.start_stream()
    
    def resample(self, data, input_rate):

        data16 = np.fromstring(string=data, dtype=np.int16)
        resample_size = int(len(data16) / self.input_rate * self.RATE_PROCESS)
        resample = signal.resample(data16, resample_size)
        resample16 = np.array(resample, dtype=np.int16)
        return resample16.tostring()

    def read_resampled(self):
        return self.resample(data=self.buffer_queue.get(),
                             input_rate=self.input_rate)

    def read(self):
        return self.buffer_queue.get()

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    def write_wav(self, filename, data):
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        assert cfg.FORMAT == pyaudio.paInt16
        wf.setsampwidth(2)
        wf.setframerate(self.sample_rate)
        wf.writeframes(data)
        wf.close()

class VADAudio(Audio):
    
    def __init__(self, aggressiveness=3, device=None, input_rate=None, file=None):
        super().__init__(device=device, input_rate=input_rate, file=file)
        self.vad = webrtcvad.Vad(aggressiveness)

    def frame_generator(self):
        
        if self.input_rate == cfg.RATE_PROCESS:
            while True:
                yield self.read()
        else:
            while True:
                yield self.read_resampled()

    def vad_collector(self, padding_ms=300, ratio=0.75, frames=None):
        
        if frames is None: frames = self.frame_generator()
        num_padding_frames = padding_ms // self.frame_duration_ms
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        noises = collections.deque(maxlen=30)
        
        triggered = False

        for frame in frames:
            if len(frame) < 640:
                return

            sound_level = rms(frame)
            noises.append(sound_level)
            average_noise = sum(noises) / len(noises)

            is_speech = True if  sound_level >= average_noise + cfg.VOICE_DETECTION_THRESHOLD else False 
            if is_speech: is_speech = self.vad.is_speech(frame, self.sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > ratio * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        yield f
                    ring_buffer.clear()

            else:
                yield frame
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > ratio * ring_buffer.maxlen:
                    triggered = False
                    yield None
                    ring_buffer.clear()

def main(ARGS):
    # Load DeepSpeech model
    if os.path.isdir(ARGS.model):
        model_dir = ARGS.model
        ARGS.model = os.path.join(model_dir, 'output_graph.pb')
        ARGS.lm = os.path.join(model_dir, ARGS.lm)
        ARGS.trie = os.path.join(model_dir, ARGS.trie)

    print('Initializing model...')
    model = deepspeech.Model(ARGS.model, cfg.BEAM_WIDTH)
    if ARGS.lm and ARGS.trie:
        model.enableDecoderWithLM(ARGS.lm, ARGS.trie, cfg.LM_ALPHA, cfg.LM_BETA)

    vad_audio = VADAudio(aggressiveness=cfg.VAD_AGGRESSIVE,
                         device=None,
                         input_rate=cfg.DEFAULT_SAMPLE_RATE,
                         file=ARGS.file)
    
    print("Listening (ctrl-C to exit)...")
    frames = vad_audio.vad_collector()

    spinner = Halo(spinner='line')
    stream_context = model.createStream()
    wav_data = bytearray()
    current = None
    for frame in frames:
        if frame is not None:
            if spinner: spinner.start()
            if current is None: current = time.time()
            model.feedAudioContent(stream_context, np.frombuffer(frame, np.int16))
            if ARGS.savewav: wav_data.extend(frame)
        else:
            if spinner: spinner.stop()
            if ARGS.savewav:
                vad_audio.write_wav(os.path.join(ARGS.savewav, datetime.now().strftime("savewav_%Y-%m-%d_%H-%M-%S_%f.wav")), wav_data)
                wav_data = bytearray()
            text = model.finishStream(stream_context)
            if len(text): 
                print("Recognized: %s (%2.2f seconds)" % (text, time.time()-current))
                
            current = None
            stream_context = model.createStream()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="SST with Mic")

    parser.add_argument('-w', '--savewav',
                        help="Save .wav files of utterences to given directory")
    parser.add_argument('-f', '--file',
                        help="Read from .wav file instead of microphone")
    parser.add_argument('-m', '--model', required=True,
                        help="Path to the model (protocol buffer binary file, or entire directory containing all standard-named files for model)")
    parser.add_argument('-l', '--lm', default='lm.binary',
                        help="Path to the language model binary file. Default: lm.binary")
    parser.add_argument('-t', '--trie', default='trie',
                        help="Path to the language model trie file created with native_client/generate_trie. Default: trie")
    
    ARGS = parser.parse_args()
    if ARGS.savewav: os.makedirs(ARGS.savewav, exist_ok=True)
    main(ARGS)

# end of file