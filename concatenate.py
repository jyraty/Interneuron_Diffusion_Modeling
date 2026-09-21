import os
import subprocess
from pathlib import Path

# Directory containing the MP4 files
INPUT_DIR = Path(".")

# Output file
OUTPUT_FILE = "merged_output.mp4"

# Find all MP4 files (alphabetically)
mp4_files = sorted(INPUT_DIR.glob("*.mp4"))

if not mp4_files:
    raise FileNotFoundError("No MP4 files found.")

# Exclude output file if it already exists
mp4_files = [f for f in mp4_files if f.name != OUTPUT_FILE]

if len(mp4_files) < 2:
    raise ValueError("Need at least two MP4 files to concatenate.")

# Create FFmpeg concat list
list_file = INPUT_DIR / "concat_list.txt"

with open(list_file, "w", encoding="utf-8") as f:
    for video in mp4_files:
        # Escape single quotes for ffmpeg
        path = video.resolve().as_posix().replace("'", "'\\''")
        f.write(f"file '{path}'\n")

# Run FFmpeg
cmd = [
    "ffmpeg",
    "-f", "concat",
    "-safe", "0",
    "-i", str(list_file),
    "-c", "copy",
    OUTPUT_FILE,
]

try:
    subprocess.run(cmd, check=True)
    print(f"Created: {OUTPUT_FILE}")
finally:
    if list_file.exists():
        list_file.unlink()




