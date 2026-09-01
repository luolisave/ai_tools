# TTS test project
The goal of this project is use TTS to convert txt to novels (mp3).

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
py -3.11 .\split_novel.py
~~~
###

### generate 
~~~
py -3.11 .\tts4.p
~~~

