import os

import librosa
from openai import OpenAI
import soundfile as sf
import youtube_dl
from youtube_dl.utils import DownloadError
import time

def find_audio_files(path, extension=".mp3"):
    """Recursively find all files with extension in path."""
    audio_files = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(extension):
                audio_files.append(os.path.join(root, f))

    return audio_files

def youtube_to_mp3(youtube_url: str, output_dir: str) -> str:
    """Download the audio from a youtube video, save it to output_dir as an .mp3 file.

    Returns the filename of the savied video.
    """

    # config
    ydl_config = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "verbose": True,
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Downloading video from {youtube_url}")

    try:
        with youtube_dl.YoutubeDL(ydl_config) as ydl:
            ydl.download([youtube_url])
    except DownloadError:
        # weird bug where youtube-dl fails on the first download, but then works on second try... hacky ugly way around it.
        with youtube_dl.YoutubeDL(ydl_config) as ydl:
            ydl.download([youtube_url])

    audio_filename = find_audio_files(output_dir)[0]
    return audio_filename

def chunk_audio(filename, segment_length: int, output_dir):
    """segment lenght is in seconds"""

    print(f"Chunking audio to {segment_length} second segments...")

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # load audio file
    audio, sr = librosa.load(filename, sr=44100)

    # calculate duration in seconds
    duration = librosa.get_duration(y=audio, sr=sr)

    # calculate number of segments
    num_segments = int(duration / segment_length) + 1

    print(f"Chunking {num_segments} chunks...")

    # iterate through segments and save them
    for i in range(num_segments):
        start = i * segment_length * sr
        end = (i + 1) * segment_length * sr
        segment = audio[start:end]
        sf.write(os.path.join(output_dir, f"segment_{i}.mp3"), segment, sr)

    chunked_audio_files = find_audio_files(output_dir)
    return sorted(chunked_audio_files)

def transcribe_audio(audio_files: list, output_file=None, model="whisper-1") -> list:

    print("converting audio to text...")

    transcripts = []

    try:
        client = OpenAI(api_key='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')
        for audio_file in audio_files:
            audio = open(audio_file, "rb")
            response = client.audio.transcriptions.create(model=model, file=audio)
            transcripts.append(response.text)

        if output_file is not None:
            # save all transcripts to a .txt file
            with open(output_file, "a") as file:
                for transcript in transcripts:
                    file.write(transcript + " ")
                file.write('\n')
    except Exception as e:
        print(e)
    return transcripts

def transcribe_youtube_videos(youtube_urls):
    raw_audio_dir = f"raw_audio/"
    chunks_dir = f"chunks/"
    transcripts_file = f"transcripts.txt"

    segment_length = 10 * 60  # chunk to 10 minute segments

    for youtube_url in youtube_urls:
        try:
            # download the video using youtube-dl
            audio_filename = youtube_to_mp3(youtube_url, output_dir=raw_audio_dir)

            # chunk each audio file to shorter audio files (not necessary for shorter videos...)
            chunked_audio_files = chunk_audio(
                audio_filename, segment_length=segment_length, output_dir=chunks_dir
            )

            # transcribe each chunked audio file using whisper speech2text
            transcribe_audio(chunked_audio_files, transcripts_file)

            time.sleep(5)

            audio_list = os.listdir(raw_audio_dir)
            for file_name in audio_list:
                file_path = os.path.join(raw_audio_dir, file_name)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        except Exception as e:
            print('Error transcribing:', e)

urls = [
      'https://www.youtube.com/watch?v=_QL2Vr-Rbhk',    'https://www.youtube.com/watch?v=vW2pn2LsrZU',    'https://www.youtube.com/watch?v=795ttHjcuNA',
      'https://www.youtube.com/watch?v=g8IVEuGy3dk',    'https://www.youtube.com/watch?v=4Kq4hoCWG4c',    'https://www.youtube.com/watch?v=43FbmrkHoiY',
      'https://www.youtube.com/watch?v=O6lVXP1XJrc',    'https://www.youtube.com/watch?v=51Lal4CqJfM',    'https://www.youtube.com/watch?v=-9JXEzCUmKE',
      'https://www.youtube.com/watch?v=r7bsamy9n5c',    'https://www.youtube.com/watch?v=tFf5HiuK6v0',    'https://www.youtube.com/watch?v=Na44QV_Q7ic',
      'https://www.youtube.com/watch?v=iKkkRqBL3pM',        'https://www.youtube.com/watch?v=A68hJll7Us4',
          'https://www.youtube.com/watch?v=67roKfGj_Fo',    'https://www.youtube.com/watch?v=Euei-fpFlrQ',
          'https://www.youtube.com/watch?v=pDsabB1HVAM',    'https://www.youtube.com/watch?v=6_h6FpHnuIs',
      'https://www.youtube.com/watch?v=PkSv__cAYfw',    'https://www.youtube.com/watch?v=gRUWMIL9l8g',    'https://www.youtube.com/watch?v=s-iqqWnmgkc',
      'https://www.youtube.com/watch?v=mOFAP0KF2x4',    'https://www.youtube.com/watch?v=E1I9eof_szE',    
      'https://www.youtube.com/watch?v=qrKbmzYGLEM',        'https://www.youtube.com/watch?v=xGh4GciXPQk',
      'https://www.youtube.com/watch?v=1ISKxiGw4K8',    'https://www.youtube.com/watch?v=ShkyV3DyTD4',    'https://www.youtube.com/watch?v=XYlDNfS48sg',
          'https://www.youtube.com/watch?v=TNMG98EhKLU',    'https://www.youtube.com/watch?v=TRX-BmKdltY',
          'https://www.youtube.com/watch?v=RWaOoa0UMcI',
      'https://www.youtube.com/watch?v=Ye4cVwfSdAc',    'https://www.youtube.com/watch?v=cF3junSjIAA',    'https://www.youtube.com/watch?v=d18-39m9NoE',
      'https://www.youtube.com/watch?v=rzOddgKPPtI',    'https://www.youtube.com/watch?v=px4EAg0Vbg4',    'https://www.youtube.com/watch?v=-DQuJTTcbdw',
      'https://www.youtube.com/watch?v=3_B3smljvkQ',    'https://www.youtube.com/watch?v=3us3vUoLkac',    'https://www.youtube.com/watch?v=ol4WW_IVGOQ',
      'https://www.youtube.com/watch?v=2cLT4mTqWz4',    'https://www.youtube.com/watch?v=TPZPB2uCh8k',    
      'https://www.youtube.com/watch?v=9rRp2gDE7Yc',    'https://www.youtube.com/watch?v=hqrlgNP7E-0',    'https://www.youtube.com/watch?v=fXhf9Yh7tf8',
      'https://www.youtube.com/watch?v=jV3FUzCqQAQ',    'https://www.youtube.com/watch?v=uhBw9DO-z7A',    'https://www.youtube.com/watch?v=rczhz6xeIfQ'
  ]

# transcribe_youtube_videos(urls[8:])


# https://www.youtube.com/watch?v=neqC6alOv5s stadium background noise to make composite clip for whole video. subclip for 0 to video_clip.duration and make composite with audio
youtube_to_mp3('https://www.youtube.com/watch?v=neqC6alOv5s', output_dir="raw_audio/")