#!/usr/bin/env python3
"""
YouTube Download and Transcribe Pipeline

A script that downloads audio from YouTube videos and transcribes them using OpenAI's Whisper API.
Combines the functionality of youtube_downloader.py and audio_transcriber.py.
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
import shutil

# Import modules from our other scripts
try:
    from youtube_downloader import download_video, get_video_info
    from audio_transcriber import process_audio_file
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure youtube_downloader.py and audio_transcriber.py are in the same directory")
    sys.exit(1)


def download_and_transcribe_youtube(
    url: str,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    output_dir: str = "./transcriptions",
    quality: str = "best",
    cookies_file: Optional[str] = None,
    api_key: Optional[str] = None,
    keep_audio: bool = False,
    output_format: str = "txt"
) -> Optional[Dict[str, Any]]:
    """
    Download audio from YouTube and transcribe it.
    
    Args:
        url: YouTube video URL
        language: Language of the audio for transcription
        prompt: Optional prompt to guide transcription
        output_dir: Directory to save transcription results
        quality: Video quality for download
        cookies_file: Path to cookies file for YouTube
        api_key: OpenAI API key
        keep_audio: Whether to keep the downloaded audio file
        output_format: Format for transcription output ('txt' or 'json')
    
    Returns:
        Dictionary with results or None if error
    """
    print("🎥 YouTube Download and Transcribe Pipeline")
    print("=" * 50)
    
    # Get video info first
    print("📋 Getting video information...")
    video_info = get_video_info(url, cookies_file)
    if not video_info:
        print("❌ Failed to get video information")
        return None
    
    print(f"📹 Title: {video_info['title']}")
    print(f"👤 Uploader: {video_info['uploader']}")
    print(f"⏱️  Duration: {video_info['duration']} seconds")
    print(f"👀 Views: {video_info['view_count']:,}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create temporary directory for audio download
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\n🎵 Downloading audio to temporary directory...")
        
        # Download audio only
        success = download_video(
            url=url,
            output_path=temp_dir,
            quality=quality,
            audio_only=True,
            extract_audio=False,
            cookies_file=cookies_file
        )
        
        if not success:
            print("❌ Failed to download audio")
            return None
        
        # Find the downloaded audio file
        audio_files = list(Path(temp_dir).glob("*.mp3"))
        if not audio_files:
            audio_files = list(Path(temp_dir).glob("*.m4a"))
        if not audio_files:
            audio_files = list(Path(temp_dir).glob("*.webm"))
        
        if not audio_files:
            print("❌ No audio file found after download")
            return None
        
        audio_file = audio_files[0]
        print(f"✅ Audio downloaded: {audio_file.name}")
        
        # Move to output directory if keeping audio
        final_audio_path = None
        if keep_audio:
            final_audio_path = output_path / audio_file.name
            shutil.copy2(audio_file, final_audio_path)
            print(f"💾 Audio saved to: {final_audio_path}")
        
        print(f"\n🎤 Starting transcription...")
        
        # Create enhanced prompt if not provided
        if not prompt:
            prompt = f"This is audio from a YouTube video titled '{video_info['title']}' by {video_info['uploader']}"
        
        # Transcribe the audio
        transcription_results = process_audio_file(
            audio_file_path=str(audio_file),
            api_key=api_key,
            language=language,
            prompt=prompt,
            output_format=output_format,
            save_results=False  # We'll handle saving ourselves
        )
        
        if not transcription_results:
            print("❌ Transcription failed")
            return None
        
        # Save transcription results with video info
        enhanced_results = {
            **transcription_results,
            'video_info': video_info,
            'youtube_url': url,
            'final_audio_path': str(final_audio_path) if final_audio_path else None
        }
        
        # Generate output filename based on video title
        safe_title = "".join(c for c in video_info['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title[:50]  # Limit length
        
        if output_format.lower() == "json":
            import json
            output_file = output_path / f"{safe_title}_transcription.json"
            
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(enhanced_results, f, indent=2, ensure_ascii=False)
                print(f"📄 JSON results saved to: {output_file}")
            except Exception as e:
                print(f"❌ Error saving JSON file: {e}")
                return None
        
        else:  # Text format
            output_file = output_path / f"{safe_title}_transcription.txt"
            
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write("=== YOUTUBE VIDEO TRANSCRIPTION ===\n\n")
                    f.write(f"🔗 URL: {url}\n")
                    f.write(f"📹 Title: {video_info['title']}\n")
                    f.write(f"👤 Uploader: {video_info['uploader']}\n")
                    f.write(f"⏱️  Duration: {video_info['duration']} seconds\n")
                    f.write(f"👀 Views: {video_info['view_count']:,}\n")
                    f.write(f"📅 Upload Date: {video_info['upload_date']}\n")
                    f.write(f"🎤 Processing Time: {enhanced_results['processing_time']:.2f} seconds\n")
                    f.write(f"🗣️  Language: {enhanced_results['language']}\n")
                    f.write(f"📝 Text Length: {len(enhanced_results['text'])} characters\n")
                    if final_audio_path:
                        f.write(f"🎵 Audio File: {final_audio_path}\n")
                    f.write("\n" + "="*60 + "\n")
                    f.write("📖 VIDEO DESCRIPTION:\n")
                    f.write("="*60 + "\n")
                    f.write(f"{video_info['description']}\n")
                    f.write("\n" + "="*60 + "\n")
                    f.write("🎙️  TRANSCRIBED TEXT:\n")
                    f.write("="*60 + "\n\n")
                    f.write(enhanced_results['text'])
                
                print(f"📄 Transcription saved to: {output_file}")
            except Exception as e:
                print(f"❌ Error saving text file: {e}")
                return None
        
        enhanced_results['output_file'] = str(output_file)
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📊 Summary:")
        print(f"   • Video: {video_info['title'][:50]}...")
        print(f"   • Duration: {video_info['duration']} seconds")
        print(f"   • Transcription: {len(enhanced_results['text'])} characters")
        print(f"   • Processing time: {enhanced_results['processing_time']:.2f} seconds")
        print(f"   • Output: {output_file}")
        if final_audio_path:
            print(f"   • Audio: {final_audio_path}")
        
        return enhanced_results


def main():
    """Main function to handle command line arguments and execute pipeline."""
    parser = argparse.ArgumentParser(
        description="Download YouTube video audio and transcribe it using OpenAI Whisper"
    )
    parser.add_argument(
        "url",
        help="YouTube video URL"
    )
    parser.add_argument(
        "--language",
        help="Language of the audio (e.g., 'en', 'zh', 'es'). Auto-detect if not specified"
    )
    parser.add_argument(
        "--prompt",
        help="Optional prompt to guide the transcription"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="./transcriptions",
        help="Output directory for transcription results (default: ./transcriptions)"
    )
    parser.add_argument(
        "-q", "--quality",
        default="best",
        help="Audio quality for download (default: best)"
    )
    parser.add_argument(
        "--cookies",
        help="Path to cookies file for YouTube authentication"
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (default: uses OPENAI_API_KEY_MIRACLE env var)"
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep the downloaded audio file"
    )
    parser.add_argument(
        "--output-format",
        choices=["txt", "json"],
        default="txt",
        help="Output format for transcription results (default: txt)"
    )
    
    args = parser.parse_args()
    
    # Check for API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY_MIRACLE')
    if not api_key:
        print("❌ Error: OpenAI API key not found.")
        print("Please set OPENAI_API_KEY_MIRACLE environment variable or use --api-key option.")
        sys.exit(1)
    
    print("🚀 Starting YouTube Download and Transcribe Pipeline")
    print(f"🔗 URL: {args.url}")
    if args.language:
        print(f"🗣️  Language: {args.language}")
    if args.prompt:
        print(f"💬 Prompt: {args.prompt}")
    print(f"📁 Output Directory: {args.output_dir}")
    print(f"🎵 Keep Audio: {args.keep_audio}")
    print(f"📄 Output Format: {args.output_format}")
    
    # Execute the pipeline
    results = download_and_transcribe_youtube(
        url=args.url,
        language=args.language,
        prompt=args.prompt,
        output_dir=args.output_dir,
        quality=args.quality,
        cookies_file=args.cookies,
        api_key=api_key,
        keep_audio=args.keep_audio,
        output_format=args.output_format
    )
    
    if results:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main() 