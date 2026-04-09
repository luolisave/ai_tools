import os
import glob
import asyncio
from edge_tts import Communicate

# Define directories
NOVEL_DIR = "./novel"
MP3_DIR = "./mp3"


async def convert_one_file(txt_path):
    try:
        base_filename = os.path.splitext(os.path.basename(txt_path))[0]
        print(f"\nProcessing file: {os.path.basename(txt_path)}...")

        # 1. Read text
        with open(txt_path, 'r', encoding='utf-8') as f:
            text_content = f.read()

        if not text_content.strip():
            print(f"Skipping {os.path.basename(txt_path)}: File is empty.")
            return

        # 2. Output path
        mp3_filename = f"{base_filename}.mp3"
        mp3_path = os.path.join(MP3_DIR, mp3_filename)

        print(f"Generating MP3 at: {mp3_path}")

        # ❗关键：await
        communicate = Communicate(text_content, 
                                  voice="zh-CN-XiaoxiaoNeural", # zh-CN-YunxiNeural  # zh-CN-XiaoxiaoNeural 尖 # zh-CN-XiaoyiNeural 更柔和  # zh-CN-XiaohanNeural
                                  pitch="-15Hz",
                                  rate="-5%"
                                  ) # 语速和音调微调（可选）
        await communicate.save(mp3_path)

        # 验证文件是否真的存在
        if os.path.exists(mp3_path):
            print(f"✔ Successfully converted {os.path.basename(txt_path)}")
        else:
            print(f"❌ Failed to save {mp3_filename}")

    except Exception as e:
        print(f"Error processing {os.path.basename(txt_path)}: {e}")


async def convert_text_to_mp3():
    print("--- Starting Text to MP3 Conversion ---")

    # Ensure output directory exists
    os.makedirs(MP3_DIR, exist_ok=True)

    txt_files = glob.glob(os.path.join(NOVEL_DIR, "*.txt"))

    if not txt_files:
        print(f"No .txt files found in: {NOVEL_DIR}")
        return

    print(f"Found {len(txt_files)} text files to process.")

    # 顺序处理（稳定）
    for txt_path in txt_files:
        await convert_one_file(txt_path)

    print("\n--- Conversion process finished ---")


if __name__ == "__main__":
    asyncio.run(convert_text_to_mp3())