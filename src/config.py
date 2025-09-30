# config.py
import os

# Project root = directory one level up from the current file's directory
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.join(PROJ_DIR, "app")
DATA_DIR = os.path.join(PROJ_DIR, "data")

MP3_DIR = os.path.join(DATA_DIR, "mp3")
ANNOTATIONS_PATH = os.path.join(DATA_DIR, "annotations_final.csv")
SPECTROGRAMS_DIR = os.path.join(PROJ_DIR, "spectrograms")