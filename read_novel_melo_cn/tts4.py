import stanza
from melo.api import TTS
from pydub import AudioSegment
import tempfile
import os
import shutil
import re
import nltk


# 下载必要的 NLTK 数据, 避免中文中有英文时出现错误
for res in ["averaged_perceptron_tagger_eng", "cmudict"]:
    try:
        nltk.data.find(f"taggers/{res}")
    except LookupError:
        nltk.download(res)


# 初始化 TTS
tts = TTS(language="ZH", device="cuda")  # GPU 可用改成 "cuda"

# 初始化 Stanza
stanza.download("zh")
nlp = stanza.Pipeline("zh")

# 输入和输出目录
input_dir = "./novel"
output_dir = "./novel_mp3"
os.makedirs(output_dir, exist_ok=True)

def clean_input_text(text):
    """
    清理输入文本，去除特殊字符和无效内容。
    """
    # 1️⃣ Remove control characters and other invisible junk
    text = re.sub(r'[\r\n\x00-\x1f\x85\xa0]+', ' ', text)

    # 2️⃣ Keep everything else (including the full‑width brackets you use for IDs)
    #    Only strip characters that are *guaranteed* to be problematic.
    #    Here we keep letters, numbers, whitespace, CJK characters and the
    #    punctuation you listed, plus the two bracket styles.
    allowed = r'\w\s\u4e00-\u9fff.,!?;:，。！？；：【】'
    # Build a pattern that matches anything NOT in the allowed set
    text = re.sub(f'[^{allowed}]', '', text)

    # 3️⃣ Collapse multiple spaces into a single space (optional)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 遍历 novel 目录下的所有 txt 文件
for file_name in sorted(os.listdir(input_dir)):
    if file_name.endswith(".txt"):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.mp3")

        # 读小说文件
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        # 清理文本
        text = clean_input_text(text)

        # 分句
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sentences if sent.text.strip()]

        # 临时目录存每句音频
        tmp_dir = tempfile.mkdtemp()
        audio_segments = []

        for i, sentence in enumerate(sentences):
            tmp_path = os.path.join(tmp_dir, f"seg_{i}.wav")
            try:
                tts.tts_to_file(
                    text=sentence,
                    speaker_id=1,
                    output_path=tmp_path
                )
            except Exception as e:
                print(f"[跳过] 无法合成的句子: {sentence!r}\n原因: {e}")
                continue
            audio_segments.append(AudioSegment.from_wav(tmp_path))

        # 合并音频，中间可加 500ms 停顿
        final_audio = AudioSegment.silent(duration=0)
        for seg in audio_segments:
            final_audio += seg + AudioSegment.silent(duration=500)

        # 输出完整小说朗读
        final_audio.export(output_path, format="mp3")

        # 清理临时文件
        shutil.rmtree(tmp_dir)

        print(f"朗读生成完成 → {output_path}")
