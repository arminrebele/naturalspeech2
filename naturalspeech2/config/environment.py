import os
from dotenv import load_dotenv

def setup_environment():
    """
    Load environment variables from a .env file and set up FFmpeg DLL path on Windows.
    """
    
    load_dotenv() 

    ffmpeg_bin_path = os.environ.get("FFMPEG_BIN_PATH") 

    if os.name == 'nt' and ffmpeg_bin_path:
        try:
            os.add_dll_directory(ffmpeg_bin_path)
        except FileNotFoundError:
            print(f"Error: FFMPEG_BIN_PATH '{ffmpeg_bin_path}' was found in .env but the directory doesn't exist.")