# YouTube Downloader Usage Guide

## Installation

First, install the required dependency:

```bash
# Option 1: Using pipx (recommended for isolated environments)
brew install pipx
pipx install yt-dlp

# Option 2: Using pip with virtual environment
python3 -m venv youtube_env
source youtube_env/bin/activate
pip install yt-dlp

# Option 3: System-wide installation (use with caution)
pip3 install --break-system-packages yt-dlp
```

## Basic Usage

### Download a video (best quality)

```bash
python3 youtube_downloader.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Download to specific directory

```bash
python3 youtube_downloader.py "https://www.youtube.com/watch?v=VIDEO_ID" -o "/path/to/downloads"
```

### Download audio only (MP3)

```bash
python3 youtube_downloader.py "https://www.youtube.com/watch?v=VIDEO_ID" --audio-only
```

### Extract audio from video

```bash
python3 youtube_downloader.py "https://www.youtube.com/watch?v=VIDEO_ID" --extract-audio
```

### Get video information without downloading

```bash
python3 youtube_downloader.py "https://www.youtube.com/watch?v=VIDEO_ID" --info
```

### Specify video quality

```bash
python3 youtube_downloader.py "https://www.youtube.com/watch?v=VIDEO_ID" -q "720p"
```

### Use custom cookies file

```bash
python3 youtube_downloader.py "https://www.youtube.com/watch?v=VIDEO_ID" --cookies "path/to/cookies.txt"
```

## Quality Options

- `best` - Best available quality (default)
- `worst` - Worst available quality
- `720p` - Specific resolution
- `480p` - Specific resolution
- `bestvideo+bestaudio` - Best video and audio separately

## Features

- ✅ Downloads YouTube videos in various formats
- ✅ Audio-only downloads (MP3 conversion)
- ✅ Video information extraction
- ✅ Quality selection
- ✅ Custom output directories
- ✅ Automatic cookies detection and support
- ✅ Error handling and validation
- ✅ Support for various YouTube URL formats

## Supported URL Formats

- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `https://m.youtube.com/watch?v=VIDEO_ID`

## Cookies Setup

The script automatically looks for cookies files in this order:

1. `www.youtube.com_cookies.txt` (included in this project)
2. `cookies.txt`
3. `youtube_cookies.txt`

If you need to use a different cookies file, use the `--cookies` option:

```bash
python3 youtube_downloader.py "URL" --cookies "custom_cookies.txt"
```

### How to get YouTube cookies:

1. Install a browser extension like "Get cookies.txt" for Chrome/Firefox
2. Visit youtube.com and log in to your account
3. Use the extension to export cookies to a text file
4. Save as `www.youtube.com_cookies.txt` in the script directory

## Notes

- The script creates a `downloads` directory by default if no output path is specified
- Audio extraction requires FFmpeg to be installed on your system
- Cookies help bypass YouTube's bot detection and access restricted videos
- The script validates YouTube URLs before attempting downloads
- Large videos may take time to download depending on your internet connection

## Error Handling

The script includes comprehensive error handling for:

- Invalid URLs
- Network issues
- Missing dependencies
- Download failures
- File system errors

## Examples

```bash
# Download a video to current directory
python3 youtube_downloader.py "https://youtu.be/dQw4w9WgXcQ"

# Download audio only to music folder
python3 youtube_downloader.py "https://youtu.be/dQw4w9WgXcQ" --audio-only -o "./music"

# Check video info before downloading
python3 youtube_downloader.py "https://youtu.be/dQw4w9WgXcQ" --info

# Download specific quality
python3 youtube_downloader.py "https://youtu.be/dQw4w9WgXcQ" -q "480p"

# Use specific cookies file
python3 youtube_downloader.py "https://youtu.be/dQw4w9WgXcQ" --cookies "my_cookies.txt"

# Download with cookies and custom settings
python3 youtube_downloader.py "https://youtu.be/dQw4w9WgXcQ" --cookies "www.youtube.com_cookies.txt" --audio-only -o "./downloads"
```
