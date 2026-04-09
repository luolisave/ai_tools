import os
from gtts import gTTS

# Convert text to speech and save as MP3
def text_to_speech(text, output_path, lang="zh-cn"):
    """
    Convert text to speech using Google Text-to-Speech (gTTS).
    
    Args:
        text: The text to convert
        output_path: Path where the MP3 file will be saved
        lang: Language code (default: zh-cn for Chinese)
    """
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(output_path)
        print(f"✓ Converted: {output_path}")
    except Exception as e:
        print(f"✗ Error converting {output_path}: {e}")

# Process all text files in the ./novel directory
def process_novel_files():
    novel_dir = "./novel"
    output_dir = "./mp3"
    os.makedirs(output_dir, exist_ok=True)

    txt_files = sorted([f for f in os.listdir(novel_dir) if f.endswith(".txt")])
    
    if not txt_files:
        print(f"No .txt files found in {novel_dir}")
        return
    
    print(f"Found {len(txt_files)} files to convert...\n")
    
    for filename in txt_files:
        input_path = os.path.join(novel_dir, filename)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.mp3")

        with open(input_path, "r", encoding="utf-8") as file:
            text = file.read()

        text_to_speech(text, output_path, lang="zh-cn")
    
    print(f"\nAll files converted to {output_dir}/")

if __name__ == "__main__":
    process_novel_files()