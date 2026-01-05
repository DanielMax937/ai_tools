#!/usr/bin/env python3
"""
YouTube Video Downloader

A script to download YouTube videos by URL using yt-dlp.
Supports various formats and quality options.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import yt_dlp
except ImportError:
    print("Error: yt-dlp is not installed. Install it with: pip install yt-dlp")
    sys.exit(1)


def download_video(
    url: str,
    output_path: str = "./downloads",
    quality: str = "best",
    audio_only: bool = False,
    extract_audio: bool = False,
    cookies_file: Optional[str] = None
) -> bool:
    """
    Download a YouTube video by URL.
    
    Args:
        url: YouTube video URL
        output_path: Directory to save the downloaded file
        quality: Video quality (best, worst, or specific format)
        audio_only: If True, download only audio
        extract_audio: If True, extract audio from video
        cookies_file: Path to cookies file for authentication
    
    Returns:
        True if download was successful, False otherwise
    """
    # Early validation
    if not url or not url.strip():
        print("Error: URL cannot be empty")
        return False
    
    if not _is_valid_youtube_url(url):
        print("Error: Invalid YouTube URL")
        return False
    
    # Auto-detect cookies file if not provided
    if cookies_file is None:
        default_cookies_files = [
            "www.youtube.com_cookies.txt",
            "cookies.txt",
            "youtube_cookies.txt"
        ]
        for cookie_file in default_cookies_files:
            if Path(cookie_file).exists():
                cookies_file = cookie_file
                print(f"Using cookies file: {cookies_file}")
                break
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure yt-dlp options
    ydl_opts = _get_download_options(
        output_path=str(output_dir),
        quality=quality,
        audio_only=audio_only,
        extract_audio=extract_audio,
        cookies_file=cookies_file
    )
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Downloading: {url}")
            ydl.download([url])
            print("Download completed successfully!")
            return True
            
    except yt_dlp.DownloadError as e:
        print(f"Download error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def _is_valid_youtube_url(url: str) -> bool:
    """Check if the URL is a valid YouTube URL."""
    youtube_domains = [
        "youtube.com",
        "youtu.be",
        "www.youtube.com",
        "m.youtube.com"
    ]
    return any(domain in url.lower() for domain in youtube_domains)


def _get_download_options(
    output_path: str,
    quality: str,
    audio_only: bool,
    extract_audio: bool,
    cookies_file: Optional[str] = None
) -> Dict[str, Any]:
    """Get yt-dlp download options based on parameters."""
    base_opts = {
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',
        'noplaylist': True,
    }
    
    # Add cookies if available
    if cookies_file and Path(cookies_file).exists():
        base_opts['cookiefile'] = cookies_file
    
    if audio_only:
        base_opts.update({
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }]
        })
    elif extract_audio:
        base_opts.update({
            'format': quality,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }]
        })
    else:
        base_opts['format'] = quality
    
    return base_opts


def get_video_info(url: str, cookies_file: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Get information about a YouTube video without downloading it.
    
    Args:
        url: YouTube video URL
        cookies_file: Path to cookies file for authentication
    
    Returns:
        Video information dictionary or None if error
    """
    if not _is_valid_youtube_url(url):
        print("Error: Invalid YouTube URL")
        return None
    
    # Auto-detect cookies file if not provided
    if cookies_file is None:
        default_cookies_files = [
            "www.youtube.com_cookies.txt",
            "cookies.txt",
            "youtube_cookies.txt"
        ]
        for cookie_file in default_cookies_files:
            if Path(cookie_file).exists():
                cookies_file = cookie_file
                break
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
    }
    
    # Add cookies if available
    if cookies_file and Path(cookies_file).exists():
        ydl_opts['cookiefile'] = cookies_file
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                'title': info.get('title', 'Unknown'),
                'uploader': info.get('uploader', 'Unknown'),
                'duration': info.get('duration', 0),
                'view_count': info.get('view_count', 0),
                'upload_date': info.get('upload_date', 'Unknown'),
                'description': info.get('description', '')[:200] + '...' if info.get('description') else 'No description'
            }
    except Exception as e:
        print(f"Error getting video info: {e}")
        return None


def main():
    """Main function to handle command line arguments and execute download."""
    parser = argparse.ArgumentParser(description="Download YouTube videos by URL")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "-o", "--output", 
        default="./downloads",
        help="Output directory (default: ./downloads)"
    )
    parser.add_argument(
        "-q", "--quality",
        default="best",
        help="Video quality: best, worst, or specific format (default: best)"
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Download audio only (MP3 format)"
    )
    parser.add_argument(
        "--extract-audio",
        action="store_true",
        help="Extract audio from video after download"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show video information without downloading"
    )
    parser.add_argument(
        "--cookies",
        help="Path to cookies file (default: auto-detect www.youtube.com_cookies.txt)"
    )
    
    args = parser.parse_args()
    
    if args.info:
        info = get_video_info(args.url, cookies_file=args.cookies)
        if info:
            print("\n=== Video Information ===")
            print(f"Title: {info['title']}")
            print(f"Uploader: {info['uploader']}")
            print(f"Duration: {info['duration']} seconds")
            print(f"Views: {info['view_count']:,}")
            print(f"Upload Date: {info['upload_date']}")
            print(f"Description: {info['description']}")
        return
    
    success = download_video(
        url=args.url,
        output_path=args.output,
        quality=args.quality,
        audio_only=args.audio_only,
        extract_audio=args.extract_audio,
        cookies_file=args.cookies
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 