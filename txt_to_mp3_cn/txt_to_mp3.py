import os
import torch
import torchaudio
from pydub import AudioSegment
import warnings
warnings.filterwarnings('ignore')

# Check CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")


def initialize_silero_tts():
    """
    Initialize Silero TTS model locally with CUDA support.
    Silero TTS is fast and lightweight, optimized for local inference.
    """
    try:
        # Load model from torch hub (supports CUDA acceleration)
        model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-tts:main',
            model='silero_tts',
            language='zh',  # Chinese
            speaker='kseniya'  # or other available speakers
        )
        model = model.to(device)
        return model
    except Exception as e:
        print(f"ERROR initializing Silero TTS: {e}")
        print("Make sure you have internet connection for first-time model download")
        return None


def text_to_speech(text, output_path, tts_model):
    """
    Convert text to speech using Silero TTS with CUDA acceleration.
    
    Args:
        text: The text to convert (Chinese text for best results)
        output_path: Path where the MP3 file will be saved
        tts_model: Initialized Silero TTS model
    """
    try:
        # Split long text into chunks to avoid memory issues
        max_chunk_length = 1000
        chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
        
        audio_list = []
        
        for chunk in chunks:
            # Generate speech (returns audio tensor)
            audio = tts_model.apply_tts(
                text=chunk,
                speaker='kseniya',
                sample_rate=24000
            )
            audio_list.append(audio)
        
        # Concatenate audio chunks
        if len(audio_list) > 1:
            full_audio = torch.cat(audio_list)
        else:
            full_audio = audio_list[0]
        
        # Save as WAV first
        wav_path = output_path.replace(".mp3", ".wav")
        torchaudio.save(wav_path, full_audio.unsqueeze(0).cpu(), 24000)
        
        # Convert WAV to MP3
        wav_audio = AudioSegment.from_wav(wav_path)
        wav_audio.export(output_path, format="mp3", bitrate="192k")
        
        # Clean up temporary WAV file
        if os.path.exists(wav_path):
            os.remove(wav_path)
        
        print(f"✓ Converted: {output_path}")
    except Exception as e:
        print(f"✗ Error converting {output_path}: {str(e)}")


def process_novel_files():
    """
    Process all text files in ./novel directory and convert to MP3 using local T5-TTS.
    """
    novel_dir = "./novel"
    output_dir = "./mp3"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize model
    print("Initializing Silero TTS model...")
    tts_model = initialize_silero_tts()
    if tts_model is None:
        return
    
    txt_files = sorted([f for f in os.listdir(novel_dir) if f.endswith(".txt")])
    
    if not txt_files:
        print(f"No .txt files found in {novel_dir}")
        return
    
    print(f"Found {len(txt_files)} files to convert...\n")
    
    for idx, filename in enumerate(txt_files, 1):
        input_path = os.path.join(novel_dir, filename)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.mp3")

        print(f"[{idx}/{len(txt_files)}] Processing: {filename}")
        with open(input_path, "r", encoding="utf-8") as file:
            text = file.read()

        text_to_speech(text, output_path, tts_model)
    
    print(f"\nAll files converted to {output_dir}/")


if __name__ == "__main__":
    process_novel_files()