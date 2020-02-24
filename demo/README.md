
### How to setup Python Environment  

```  
    conda create -n voiceai python=3  
    conda activate voiceai  
    (voiceai) pip install deepspeech  
    (voiceai) conda install nwani::portaudio nwani::pyaudio  
 
```  
### How to run   

```  
    # Download pre-trained English model and extract
    (voiceai) curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.6.1/deepspeech-0.6.1-models.tar.gz
    (voiceai) tar xvf deepspeech-0.6.1-models.tar.gz

    # Run voice recognition 
    (voiceai) python voice_recognition.py -m ../deepspeech-0.6.0-models  
```  


### Reference  

- [Deep Speech Release](https://github.com/mozilla/DeepSpeech/releases/)  
- [Audio Device Detection Problem in PyAudio](https://stackoverflow.com/questions/47640188/pyaudio-cant-detect-sound-devices)  
