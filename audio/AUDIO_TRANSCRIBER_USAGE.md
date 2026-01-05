# Audio Transcriber Usage Guide

## Overview

The `audio_transcriber.py` script uses OpenAI's Whisper API to transcribe audio files and extract all text content. It supports multiple audio formats and provides flexible output options.

## Installation

The script requires the `openai` and `pydub` packages (already included in `requirements.txt`):

```bash
pip install openai pydub
```

**Note**: `pydub` is required for automatic splitting of large audio files (>25MB).

## Setup

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY_MIRACLE='your-api-key-here'
```

Or add it to your `.env` file.

## Basic Usage

### Simple transcription

```bash
python3 audio_transcriber.py "path/to/audio.mp3"
```

### Specify language

```bash
python3 audio_transcriber.py "audio.mp3" --language "en"
```

### Use a prompt to guide transcription

```bash
python3 audio_transcriber.py "audio.mp3" --prompt "This is a technical presentation about AI"
```

### Save as JSON format

```bash
python3 audio_transcriber.py "audio.mp3" --output-format json
```

### Don't save to file (display only)

```bash
python3 audio_transcriber.py "audio.mp3" --no-save
```

### Use custom API key

```bash
python3 audio_transcriber.py "audio.mp3" --api-key "your-api-key"
```

## Supported Audio Formats

- **MP3** (.mp3)
- **MP4** (.mp4)
- **MPEG** (.mpeg)
- **MPGA** (.mpga)
- **M4A** (.m4a)
- **WAV** (.wav)
- **WEBM** (.webm)

## Features

- ✅ **Multiple audio formats** supported
- ✅ **Automatic file splitting** for large files (>25MB)
- ✅ **Language detection** or manual specification
- ✅ **Custom prompts** to guide transcription accuracy
- ✅ **Chunk-aware processing** with intelligent prompts
- ✅ **Text and JSON output** formats
- ✅ **Processing time tracking**
- ✅ **Text preview** before saving
- ✅ **Comprehensive error handling**

## Language Codes

Common language codes for the `--language` option:

- `en` - English
- `zh` - Chinese
- `es` - Spanish
- `fr` - French
- `de` - German
- `ja` - Japanese
- `ko` - Korean
- `pt` - Portuguese
- `ru` - Russian
- `ar` - Arabic

## Output Files

The script automatically generates output files based on the input filename:

### Text Format (default)

```
audio_filename_transcription.txt
```

Contains:

- File metadata (size, processing time, model used)
- Full transcribed text

### JSON Format

```
audio_filename_transcription.json
```

Contains structured data:

- `text`: Transcribed content
- `processing_time`: Time taken for transcription
- `file_size_mb`: Original file size
- `model_used`: Whisper model used
- `language`: Language detected/specified

## Advanced Examples

### Transcribe with language and prompt

```bash
python3 audio_transcriber.py "meeting.mp3" \
  --language "en" \
  --prompt "This is a business meeting about project planning" \
  --output-format "json"
```

### Batch processing multiple files

```bash
for file in *.mp3; do
  python3 audio_transcriber.py "$file" --language "en"
done
```

### Transcribe with custom settings

```bash
python3 audio_transcriber.py "lecture.m4a" \
  --language "en" \
  --prompt "Educational content about machine learning" \
  --output-format "txt"
```

## Integration with YouTube Downloader

You can combine the YouTube downloader with the audio transcriber for a complete workflow:

```bash
# 1. Download audio from YouTube
python3 youtube_downloader.py "https://youtu.be/VIDEO_ID" --audio-only -o "./audio"

# 2. Transcribe the downloaded audio
python3 audio_transcriber.py "./audio/Video_Title.mp3" --language "en"
```

Or create a pipeline script:

```bash
#!/bin/bash
# download_and_transcribe.sh

URL="$1"
LANGUAGE="${2:-en}"

echo "Downloading audio from: $URL"
python3 youtube_downloader.py "$URL" --audio-only -o "./temp_audio"

echo "Finding downloaded audio file..."
AUDIO_FILE=$(find ./temp_audio -name "*.mp3" -type f | head -1)

if [ -n "$AUDIO_FILE" ]; then
    echo "Transcribing: $AUDIO_FILE"
    python3 audio_transcriber.py "$AUDIO_FILE" --language "$LANGUAGE"

    echo "Cleaning up temporary files..."
    rm -rf ./temp_audio

    echo "Complete! Check for _transcription.txt file."
else
    echo "Error: No audio file found after download"
fi
```

Usage:

```bash
chmod +x download_and_transcribe.sh
./download_and_transcribe.sh "https://youtu.be/VIDEO_ID" "en"
```

## Error Handling

The script handles various error conditions:

- **File not found**: Validates file existence before processing
- **File size limits**: Checks 25MB OpenAI limit
- **Unsupported formats**: Warns about potentially unsupported formats
- **API errors**: Handles network and authentication issues
- **Empty files**: Validates file content before upload

## Large File Handling

The script automatically handles files larger than 25MB:

### Automatic Splitting

- **Files >25MB** are automatically split into smaller chunks
- **Intelligent chunking** based on duration to keep chunks under 24MB
- **Sequential processing** of each chunk with context-aware prompts
- **Automatic cleanup** of temporary files after processing

### Example with Large File

```bash
# Large file (50MB) - automatically splits and processes
python3 audio_transcriber.py "large_lecture.mp3" --language "en"

# Output:
# 🔄 Large file detected (50.2MB > 25MB limit)
# Splitting into chunks of max 24.0MB each...
# Total duration: 3600.0 seconds
# Splitting into 3 chunks of ~1200.0 seconds each
# 📂 Processing chunk 1/3: chunk_001_large_lecture.mp3
# ✅ Chunk 1 completed: 2,456 characters
# ...
```

### Chunk Information in Output

```
=== AUDIO TRANSCRIPTION RESULTS ===

Original File: large_lecture.mp3
Processing Time: 45.3 seconds
File Size: 50.2MB
Chunks Processed: 3

Chunk Details:
  Chunk 1: 2,456 chars, 15.1s
  Chunk 2: 2,123 chars, 14.8s
  Chunk 3: 1,987 chars, 15.4s
```

## Performance Notes

- **File size**: No practical limit (automatic splitting for files >25MB)
- **Processing time**: Varies by file length and complexity
- **Large files**: Processing time scales linearly with number of chunks
- **API costs**: Charged per minute of audio processed (same for split files)
- **Rate limits**: OpenAI has rate limits for API usage
- **Memory usage**: Efficient processing with temporary file cleanup

## Command Line Options

```
positional arguments:
  audio_file            Path to the audio file to transcribe

options:
  -h, --help            Show help message and exit
  --language LANGUAGE   Language of the audio (e.g., 'en', 'zh', 'es')
  --prompt PROMPT       Optional prompt to guide the transcription
  --output-format {txt,json}
                        Output format for saved results (default: txt)
  --no-save             Don't save results to file, only display
  --api-key API_KEY     OpenAI API key (default: uses OPENAI_API_KEY_MIRACLE env var)
```

## Tips for Better Transcription

1. **Use prompts**: Provide context about the audio content
2. **Specify language**: Helps with accuracy for non-English content
3. **Good audio quality**: Clear recordings produce better results
4. **Appropriate file size**: Larger files may take longer to process
5. **Technical content**: Use prompts to provide domain context

## Troubleshooting

### Common Issues

**"OpenAI package not installed"**

```bash
pip install openai
```

**"API key not found"**

```bash
export OPENAI_API_KEY_MIRACLE='your-key-here'
```

**"pydub is required for processing large files"**

```bash
pip install pydub
```

**"Error splitting audio file"**

- Check if file is corrupted
- Ensure enough disk space for temporary files
- Try converting to MP3 format first

**"Unsupported file format"**

- Convert to MP3, WAV, or M4A
- Check file extension

## Examples Output

### Text Output Preview:

```
=== AUDIO TRANSCRIPTION RESULTS ===

Original File: meeting.mp3
Processing Time: 15.3 seconds
File Size: 8.2MB
Model Used: whisper-1
Language: en
Text Length: 2,456 characters

==================================================
TRANSCRIBED TEXT:
==================================================

Welcome to today's project meeting. We'll be discussing
the quarterly goals and reviewing our progress on the
new product launch...
```

### JSON Output Preview:

```json
{
  "text": "Welcome to today's project meeting...",
  "processing_time": 15.3,
  "file_size_mb": 8.2,
  "model_used": "whisper-1",
  "language": "en"
}
```
