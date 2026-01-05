# Audio Tools Quick Reference

## Scripts Overview

| Script                       | Purpose                          | Key Features                                                   |
| ---------------------------- | -------------------------------- | -------------------------------------------------------------- |
| `audio_transcriber.py`       | Transcribe MP3/audio files       | OpenAI Whisper API, automatic file splitting, multiple formats |
| `download_and_transcribe.py` | Complete YouTube → Text pipeline | Downloads audio + transcribes in one step                      |

## Quick Start

### 1. Transcribe an existing MP3 file

```bash
python3 audio_transcriber.py "audio_file.mp3"
```

### 2. Download YouTube video and transcribe

```bash
python3 download_and_transcribe.py "https://youtu.be/VIDEO_ID"
```

### 3. With language specification

```bash
python3 download_and_transcribe.py "https://youtu.be/VIDEO_ID" --language "en"
```

## Prerequisites

1. **OpenAI API Key**: Set environment variable

   ```bash
   export OPENAI_API_KEY_MIRACLE='your-api-key-here'
   ```

2. **Dependencies**: Already in `requirements.txt`

   - `openai`
   - `yt-dlp`
   - `pydub` (for large file splitting)

3. **Cookies**: YouTube cookies file `www.youtube.com_cookies.txt` (auto-detected)

## Common Use Cases

### Academic Research

```bash
# Transcribe lecture recordings
python3 audio_transcriber.py "lecture.mp3" --language "en" --prompt "University lecture on machine learning"

# YouTube educational content
python3 download_and_transcribe.py "https://youtu.be/LECTURE_ID" --language "en" --keep-audio
```

### Podcast/Interview Transcription

```bash
# Local podcast file
python3 audio_transcriber.py "podcast.mp3" --output-format json

# YouTube podcast
python3 download_and_transcribe.py "https://youtu.be/PODCAST_ID" --prompt "Technology podcast interview"
```

### Meeting Notes

```bash
# Meeting recording
python3 audio_transcriber.py "meeting.m4a" --prompt "Business meeting about project planning"
```

## Output Examples

### Text Output

```
=== YOUTUBE VIDEO TRANSCRIPTION ===

🔗 URL: https://youtu.be/dQw4w9WgXcQ
📹 Title: Rick Astley - Never Gonna Give You Up
👤 Uploader: Rick Astley
⏱️  Duration: 213 seconds
...

🎙️  TRANSCRIBED TEXT:
============================================================

Never gonna give you up, never gonna let you down...
```

### JSON Output

```json
{
  "text": "Never gonna give you up...",
  "processing_time": 15.3,
  "file_size_mb": 8.2,
  "video_info": {
    "title": "Rick Astley - Never Gonna Give You Up",
    "uploader": "Rick Astley"
  }
}
```

## Error Handling

| Error                                 | Cause                        | Solution                     |
| ------------------------------------- | ---------------------------- | ---------------------------- |
| "OpenAI package not installed"        | Missing dependency           | `pip install openai`         |
| "API key not found"                   | Missing environment variable | Set `OPENAI_API_KEY_MIRACLE` |
| "pydub is required for large files"   | Missing dependency           | `pip install pydub`          |
| "Sign in to confirm you're not a bot" | YouTube bot detection        | Use cookies file             |

## File Limits

- **Audio files**: No practical limit (automatic splitting for >25MB files)
- **Supported formats**: MP3, MP4, M4A, WAV, WEBM, MPEG
- **Languages**: Auto-detect or specify (en, zh, es, fr, de, ja, etc.)
- **Large file handling**: Automatic chunking with intelligent splitting

## Tips for Best Results

1. **Use prompts**: Provide context about the audio content
2. **Specify language**: Improves accuracy for non-English content
3. **Good audio quality**: Clear recordings = better transcription
4. **Cookies for YouTube**: Avoid authentication issues
5. **Keep audio files**: Use `--keep-audio` for reference

## Integration Examples

### Batch Processing

```bash
# Process multiple YouTube videos
for url in "https://youtu.be/ID1" "https://youtu.be/ID2"; do
    python3 download_and_transcribe.py "$url" --language "en"
done
```

### Custom Pipeline

```python
from download_and_transcribe import download_and_transcribe_youtube

# Programmatic usage
results = download_and_transcribe_youtube(
    url="https://youtu.be/VIDEO_ID",
    language="en",
    output_dir="./my_transcriptions",
    keep_audio=True
)
```

## Cost Considerations

- **OpenAI Whisper**: Charged per minute of audio
- **Typical costs**: ~$0.006 per minute
- **File size optimization**: Compress audio to reduce costs
- **Rate limits**: OpenAI has API usage limits

For detailed usage, see:

- `AUDIO_TRANSCRIBER_USAGE.md` - Complete documentation
- `YOUTUBE_DOWNLOADER_USAGE.md` - YouTube downloader guide
