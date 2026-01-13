# TTS test project
## Install dependency

Need to install cu128+ for RTX 50xx GPU
~~~
pip uninstall torch torchvision torchaudio

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
~~~

~~~
Install FFmpeg from here:
https://github.com/BtbN/FFmpeg-Builds/releases
Then make sure add it to envionment variable (for example: PATH=C:\apps\ffmpeg\bin)
~~~

~~~
pip install stanza
pip install melo
pip install pydub
~~~
* stanza: full text to paragraph
* melo : text to speech
* pydub: concat wav files to a single one

## Run
### split novel.txt files by chapter (第???章), save each chapter into ./novel folder as individual txt file
~~~
python .\split_novel.py
~~~
###

### generate 
~~~
python .\tts4.py
~~~

## Files
### test.py
test if pytorch etc are installed correctly

### tts3.py
* Use stanza to parse text to smaller 'sentenses', then feed to meloTTS. Give better results.
* Use melo to convert text to speech （fregment_number.wav)
* Use pydub to concate wav files to a single wav file, then convert to mp3


### tts4.py
a copy of tts3.py

### tts5.py
a copy of tts4.py

