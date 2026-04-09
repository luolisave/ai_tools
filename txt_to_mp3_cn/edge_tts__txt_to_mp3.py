import os
import glob
import asyncio
from edge_tts import Communicate

# ===== 配置 =====
NOVEL_DIR = "./novel"
MP3_DIR = "./mp3"
VOICE = "zh-CN-XiaoxiaoNeural"
MAX_CHARS = 3000          # 每段最大字符数（很关键）
CONCURRENCY = 2           # 并发数量（别太高）


# ===== 工具：切分长文本 =====
def split_text(text, max_len=3000):
    chunks = []
    while len(text) > max_len:
        # 尽量在句号处切
        split_pos = text.rfind("。", 0, max_len)
        if split_pos == -1:
            split_pos = max_len
        chunks.append(text[:split_pos + 1])
        text = text[split_pos + 1:]
    if text:
        chunks.append(text)
    return chunks


# ===== 转换单个文件 =====
semaphore = asyncio.Semaphore(CONCURRENCY)

async def convert_one_file(txt_path):
    async with semaphore:
        try:
            base_filename = os.path.splitext(os.path.basename(txt_path))[0]
            mp3_path = os.path.join(MP3_DIR, f"{base_filename}.mp3")

            # ✅ 断点续跑
            if os.path.exists(mp3_path):
                print(f"⏩ Skipping (exists): {base_filename}")
                return

            print(f"\n📄 Processing: {base_filename}")

            # 读文本
            with open(txt_path, "r", encoding="utf-8") as f:
                text_content = f.read().strip()

            if not text_content:
                print(f"⚠ Empty file: {base_filename}")
                return

            # ✅ 切分文本
            chunks = split_text(text_content, MAX_CHARS)
            print(f"✂ Split into {len(chunks)} chunks")

            # 临时文件（拼接用）
            temp_files = []

            for i, chunk in enumerate(chunks):
                temp_mp3 = os.path.join(MP3_DIR, f"{base_filename}_part{i}.mp3")

                communicate = Communicate(chunk, voice=VOICE)
                await communicate.save(temp_mp3)

                temp_files.append(temp_mp3)

            # ✅ 合并 MP3
            with open(mp3_path, "wb") as outfile:
                for temp in temp_files:
                    with open(temp, "rb") as infile:
                        outfile.write(infile.read())

            # 删除临时文件
            for temp in temp_files:
                os.remove(temp)

            print(f"✅ Done: {base_filename}")

        except Exception as e:
            print(f"❌ Error in {txt_path}: {e}")


# ===== 主函数 =====
async def main():
    print("=== Text → MP3 (Novel Mode) ===")

    os.makedirs(MP3_DIR, exist_ok=True)

    # ✅ 排序（关键！）
    txt_files = sorted(glob.glob(os.path.join(NOVEL_DIR, "*.txt")))

    if not txt_files:
        print("No txt files found.")
        return

    print(f"Found {len(txt_files)} files")

    tasks = [convert_one_file(p) for p in txt_files]
    await asyncio.gather(*tasks)

    print("\n🎉 All done!")


if __name__ == "__main__":
    asyncio.run(main())