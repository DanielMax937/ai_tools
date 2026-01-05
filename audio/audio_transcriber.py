#!/usr/bin/env python3
"""
Audio Transcriber Script

A script to transcribe MP3 audio files using OpenAI's Whisper API.
Extracts all text content from audio files.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import argparse
import time
import tempfile
import math
import dotenv

dotenv.load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    print("Error: OpenAI package not installed. Install it with: pip install openai")
    sys.exit(1)

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    print("Warning: pydub not installed. Large file splitting will not be available.")
    print("Install with: pip install pydub")
    PYDUB_AVAILABLE = False


def split_large_audio_file(
    audio_file_path: str,
    max_size_mb: float = 24.0
) -> List[str]:
    """
    Split a large audio file into smaller chunks under the size limit.
    
    Args:
        audio_file_path: Path to the audio file
        max_size_mb: Maximum size per chunk in MB (default: 24MB to be safe)
    
    Returns:
        List of paths to the split audio files
    """
    if not PYDUB_AVAILABLE:
        raise ImportError("pydub is required for audio file splitting. Install with: pip install pydub")
    
    audio_path = Path(audio_file_path)
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    
    print(f"Original file size: {file_size_mb:.1f}MB")
    print(f"Splitting into chunks of max {max_size_mb}MB each...")
    
    # Load the audio file
    try:
        audio = AudioSegment.from_file(audio_file_path)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        raise
    
    # Calculate duration and chunk size
    total_duration_ms = len(audio)
    total_duration_seconds = total_duration_ms / 1000
    
    # Estimate how many chunks we need based on file size
    num_chunks = math.ceil(file_size_mb / max_size_mb)
    chunk_duration_ms = total_duration_ms // num_chunks
    
    print(f"Total duration: {total_duration_seconds:.1f} seconds")
    print(f"Splitting into {num_chunks} chunks of ~{chunk_duration_ms/1000:.1f} seconds each")
    
    # Create temporary directory for chunks
    temp_dir = tempfile.mkdtemp(prefix="audio_chunks_")
    chunk_files = []
    
    for i in range(num_chunks):
        start_ms = i * chunk_duration_ms
        end_ms = min((i + 1) * chunk_duration_ms, total_duration_ms)
        
        # Extract chunk
        chunk = audio[start_ms:end_ms]
        
        # Generate chunk filename
        chunk_filename = f"chunk_{i+1:03d}_{audio_path.stem}.mp3"
        chunk_path = Path(temp_dir) / chunk_filename
        
        # Export chunk
        chunk.export(str(chunk_path), format="mp3")
        
        # Verify chunk size
        chunk_size_mb = chunk_path.stat().st_size / (1024 * 1024)
        print(f"  Chunk {i+1}/{num_chunks}: {chunk_size_mb:.1f}MB ({(end_ms-start_ms)/1000:.1f}s)")
        
        chunk_files.append(str(chunk_path))
    
    print(f"Created {len(chunk_files)} chunks in: {temp_dir}")
    return chunk_files


def transcribe_audio_file(
    audio_file_path: str,
    client: OpenAI,
    model: str = "whisper-1",
    language: Optional[str] = None,
    prompt: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Transcribe an audio file using OpenAI's Whisper API.
    
    Args:
        audio_file_path: Path to the audio file
        client: OpenAI client instance
        model: Whisper model to use (default: whisper-1)
        language: Language of the audio (optional, auto-detect if None)
        prompt: Optional prompt to guide transcription
    
    Returns:
        Dictionary with transcription results or None if error
    """
    # Early validation
    if not audio_file_path or not audio_file_path.strip():
        print("Error: Audio file path cannot be empty")
        return None
    
    audio_path = Path(audio_file_path)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_file_path}")
        return None
    
    # Check file size for logging (splitting is handled in process_audio_file)
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    
    # Check file extension
    supported_formats = {'.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm'}
    if audio_path.suffix.lower() not in supported_formats:
        print(f"Warning: File format {audio_path.suffix} may not be supported")
        print(f"Supported formats: {', '.join(supported_formats)}")
    
    print(f"Transcribing audio file: {audio_file_path}")
    print(f"File size: {file_size_mb:.1f}MB")
    
    try:
        with open(audio_file_path, 'rb') as audio_file:
            # Prepare transcription parameters
            transcribe_params = {
                'file': audio_file,
                'model': model,
            }
            
            # Add optional parameters if provided
            if language:
                transcribe_params['language'] = language
            if prompt:
                transcribe_params['prompt'] = prompt
            
            # Make API call
            print("Sending request to OpenAI Whisper API...")
            start_time = time.time()
            
            transcript = client.audio.transcriptions.create(**transcribe_params)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Extract text from response
            transcribed_text = transcript.text
            
            print(f"Transcription completed in {processing_time:.2f} seconds")
            print(f"Transcribed text length: {len(transcribed_text)} characters")
            
            return {
                'text': transcribed_text,
                'processing_time': processing_time,
                'file_size_mb': file_size_mb,
                'model_used': model,
                'language': language or 'auto-detected'
            }
            
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None


def save_transcription_results(
    results: Dict[str, Any],
    audio_file_path: str,
    output_format: str = "txt"
) -> Optional[str]:
    """
    Save transcription results to file.
    
    Args:
        results: Transcription results dictionary
        audio_file_path: Original audio file path
        output_format: Output format ('txt', 'json')
    
    Returns:
        Path to saved file or None if error
    """
    if not results:
        print("Error: No results to save")
        return None
    
    # Generate output filename
    audio_path = Path(audio_file_path)
    base_name = audio_path.stem
    
    if output_format.lower() == "json":
        import json
        output_file = audio_path.parent / f"{base_name}_transcription.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"JSON results saved to: {output_file}")
            return str(output_file)
        except Exception as e:
            print(f"Error saving JSON file: {e}")
            return None
    
    else:  # Default to txt format
        output_file = audio_path.parent / f"{base_name}_transcription.txt"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=== AUDIO TRANSCRIPTION RESULTS ===\n\n")
                f.write(f"Original File: {audio_file_path}\n")
                f.write(f"Processing Time: {results['processing_time']:.2f} seconds\n")
                f.write(f"File Size: {results['file_size_mb']:.1f}MB\n")
                f.write(f"Model Used: {results['model_used']}\n")
                f.write(f"Language: {results['language']}\n")
                f.write(f"Text Length: {len(results['text'])} characters\n")
                
                # Add chunk information if file was split
                if 'chunks_processed' in results:
                    f.write(f"Chunks Processed: {results['chunks_processed']}\n")
                    f.write("\nChunk Details:\n")
                    for chunk_detail in results['chunk_details']:
                        f.write(f"  Chunk {chunk_detail['chunk']}: {chunk_detail['text_length']} chars, {chunk_detail['processing_time']:.2f}s\n")
                
                f.write("\n" + "="*50 + "\n")
                f.write("TRANSCRIBED TEXT:\n")
                f.write("="*50 + "\n\n")
                f.write(results['text'])
            
            print(f"Transcription saved to: {output_file}")
            return str(output_file)
        except Exception as e:
            print(f"Error saving text file: {e}")
            return None


def process_audio_file(
    audio_file_path: str,
    api_key: Optional[str] = None,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    output_format: str = "txt",
    save_results: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Process an audio file and optionally save results.
    Automatically splits large files into chunks if over 25MB.
    
    Args:
        audio_file_path: Path to the audio file
        api_key: OpenAI API key (if not provided, will look for env variable)
        language: Language of the audio (optional)
        prompt: Optional prompt to guide transcription
        output_format: Format for saved results ('txt', 'json')
        save_results: Whether to save results to file
    
    Returns:
        Transcription results dictionary or None if error
    """
    # Early validation
    audio_path = Path(audio_file_path)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_file_path}")
        return None
    
    # Initialize OpenAI client
    if not api_key:
        api_key = os.getenv('OPENAI_API_KEY_MIRACLE')
    
    if not api_key:
        print("Error: OpenAI API key not found.")
        print("Please set OPENAI_API_KEY_MIRACLE environment variable or pass it as argument.")
        return None
    
    try:
        client = OpenAI(
            api_key=api_key,
            base_url='http://openai-proxy.miracleplus.com/v1'
        )
        
        # Check file size and decide whether to split
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        
        if file_size_mb > 25:
            print(f"🔄 Large file detected ({file_size_mb:.1f}MB > 25MB limit)")
            
            if not PYDUB_AVAILABLE:
                print("❌ Error: pydub is required for processing large files.")
                print("Install with: pip install pydub")
                return None
            
            # Split the file into chunks
            try:
                chunk_files = split_large_audio_file(audio_file_path)
            except Exception as e:
                print(f"❌ Error splitting audio file: {e}")
                return None
            
            # Process each chunk
            all_chunks_results = []
            combined_text = ""
            total_processing_time = 0
            
            print(f"\n🎤 Processing {len(chunk_files)} chunks...")
            
            for i, chunk_file in enumerate(chunk_files, 1):
                print(f"\n📂 Processing chunk {i}/{len(chunk_files)}: {Path(chunk_file).name}")
                
                # Add chunk context to prompt
                chunk_prompt = prompt
                if chunk_prompt:
                    chunk_prompt += f" (This is part {i} of {len(chunk_files)} chunks.)"
                else:
                    chunk_prompt = f"This is part {i} of {len(chunk_files)} chunks from a larger audio file."
                
                chunk_results = transcribe_audio_file(
                    audio_file_path=chunk_file,
                    client=client,
                    language=language,
                    prompt=chunk_prompt
                )
                
                if not chunk_results:
                    print(f"❌ Failed to process chunk {i}")
                    # Clean up temporary files
                    _cleanup_chunk_files(chunk_files)
                    return None
                
                all_chunks_results.append(chunk_results)
                combined_text += chunk_results['text'] + "\n\n"
                total_processing_time += chunk_results['processing_time']
                
                print(f"✅ Chunk {i} completed: {len(chunk_results['text'])} characters")
            
            # Clean up temporary files
            _cleanup_chunk_files(chunk_files)
            
            # Combine results
            combined_results = {
                'text': combined_text.strip(),
                'processing_time': total_processing_time,
                'file_size_mb': file_size_mb,
                'model_used': all_chunks_results[0]['model_used'],
                'language': all_chunks_results[0]['language'],
                'chunks_processed': len(chunk_files),
                'chunk_details': [
                    {
                        'chunk': i+1,
                        'text_length': len(result['text']),
                        'processing_time': result['processing_time']
                    }
                    for i, result in enumerate(all_chunks_results)
                ]
            }
            
            print(f"\n🎉 All chunks processed successfully!")
            print(f"📊 Combined results: {len(combined_results['text'])} total characters")
            
        else:
            # Single file processing (original logic)
            combined_results = transcribe_audio_file(
                audio_file_path=audio_file_path,
                client=client,
                language=language,
                prompt=prompt
            )
            
            if not combined_results:
                return None
        
        # Print results summary
        print("\n" + "="*50)
        print("TRANSCRIPTION SUMMARY")
        print("="*50)
        print(f"File: {audio_file_path}")
        print(f"Text Length: {len(combined_results['text'])} characters")
        print(f"Processing Time: {combined_results['processing_time']:.2f} seconds")
        print(f"Language: {combined_results['language']}")
        
        if 'chunks_processed' in combined_results:
            print(f"Chunks Processed: {combined_results['chunks_processed']}")
        
        # Show preview of transcribed text
        preview_length = 200
        preview_text = combined_results['text'][:preview_length]
        if len(combined_results['text']) > preview_length:
            preview_text += "..."
        
        print(f"\nTranscribed Text Preview:")
        print("-" * 30)
        print(preview_text)
        print("-" * 30)
        
        # Save results if requested
        if save_results:
            saved_file = save_transcription_results(
                results=combined_results,
                audio_file_path=audio_file_path,
                output_format=output_format
            )
            if saved_file:
                combined_results['saved_file'] = saved_file
        
        return combined_results
        
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return None


def _cleanup_chunk_files(chunk_files: List[str]) -> None:
    """Clean up temporary chunk files and directory."""
    if not chunk_files:
        return
    
    try:
        # Get the temporary directory from the first chunk file
        temp_dir = Path(chunk_files[0]).parent
        
        print(f"🧹 Cleaning up temporary files...")
        
        # Remove all chunk files
        for chunk_file in chunk_files:
            chunk_path = Path(chunk_file)
            if chunk_path.exists():
                chunk_path.unlink()
        
        # Remove the temporary directory if it's empty
        try:
            temp_dir.rmdir()
            print(f"✅ Temporary files cleaned up")
        except OSError:
            # Directory not empty, that's okay
            pass
            
    except Exception as e:
        print(f"⚠️  Warning: Could not clean up temporary files: {e}")


def main():
    """Main function to handle command line arguments and execute transcription."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using OpenAI's Whisper API"
    )
    parser.add_argument(
        "audio_file",
        help="Path to the audio file to transcribe"
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
        "--output-format",
        choices=["txt", "json"],
        default="txt",
        help="Output format for saved results (default: txt)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file, only display"
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (default: uses OPENAI_API_KEY_MIRACLE env var)",
        default=os.getenv('OPENAI_API_KEY_MIRACLE')
    )
    
    args = parser.parse_args()
    
    # Check for API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY_MIRACLE')
    if not api_key:
        print("Error: OpenAI API key not found.")
        print("Please set OPENAI_API_KEY_MIRACLE environment variable or use --api-key option.")
        sys.exit(1)
    
    print("Audio Transcription Tool")
    print("=" * 30)
    print(f"Audio File: {args.audio_file}")
    if args.language:
        print(f"Language: {args.language}")
    if args.prompt:
        print(f"Prompt: {args.prompt}")
    print(f"Output Format: {args.output_format}")
    print(f"Save Results: {not args.no_save}")
    print("-" * 30)
    
    # Process the audio file
    results = process_audio_file(
        audio_file_path=args.audio_file,
        api_key=api_key,
        language=args.language,
        prompt=args.prompt,
        output_format=args.output_format,
        save_results=not args.no_save
    )
    
    if results:
        print(f"\n✅ Transcription completed successfully!")
        if 'saved_file' in results:
            print(f"📄 Results saved to: {results['saved_file']}")
        sys.exit(0)
    else:
        print(f"\n❌ Transcription failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 