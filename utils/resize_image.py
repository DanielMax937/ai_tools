#!/usr/bin/env python3
from PIL import Image
import os
import argparse
from typing import Tuple, Literal, Optional


def resize_image(
    input_path: str,
    output_path: str,
    target_size: Tuple[int, int] = (720, 1280),
    method: Literal["crop", "pad"] = "crop",
    bg_color: Tuple[int, int, int] = (0, 0, 0)
) -> None:
    """
    Resize an image to the target size (width, height).
    
    Args:
        input_path: Path to the input image
        output_path: Path where the resized image will be saved
        target_size: Target size as (width, height)
        method: Resize method - 'crop' or 'pad'
        bg_color: Background color for padding as RGB tuple
    """
    img = Image.open(input_path)
    
    # Get original dimensions
    orig_width, orig_height = img.size
    target_width, target_height = target_size
    
    # Calculate aspect ratios
    orig_aspect = orig_width / orig_height
    target_aspect = target_width / target_height
    
    if method == "crop":
        # Resize and crop to fill target dimensions
        if orig_aspect > target_aspect:
            # Image is wider than target, resize to match height and crop width
            new_width = int(target_height * orig_aspect)
            new_height = target_height
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Calculate crop coordinates
            left = (new_width - target_width) // 2
            right = left + target_width
            top = 0
            bottom = target_height
            
        else:
            # Image is taller than target, resize to match width and crop height
            new_width = target_width
            new_height = int(target_width / orig_aspect)
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Calculate crop coordinates
            left = 0
            right = target_width
            top = (new_height - target_height) // 2
            bottom = top + target_height
            
        # Crop the image
        img = img.crop((left, top, right, bottom))
        
    elif method == "pad":
        # Resize and pad to fill target dimensions
        if orig_aspect > target_aspect:
            # Image is wider than target, resize to match width and pad height
            new_width = target_width
            new_height = int(target_width / orig_aspect)
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Create new image with target size and background color
            new_img = Image.new("RGB", target_size, bg_color)
            
            # Paste original image in the center
            paste_y = (target_height - new_height) // 2
            new_img.paste(img, (0, paste_y))
            img = new_img
            
        else:
            # Image is taller than target, resize to match height and pad width
            new_width = int(target_height * orig_aspect)
            new_height = target_height
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Create new image with target size and background color
            new_img = Image.new("RGB", target_size, bg_color)
            
            # Paste original image in the center
            paste_x = (target_width - new_width) // 2
            new_img.paste(img, (paste_x, 0))
            img = new_img
    
    # Save the resized image
    img.save(output_path)
    print(f"Resized image saved to {output_path}")


def process_directory(
    input_dir: str,
    output_dir: str,
    target_size: Tuple[int, int] = (720, 1280),
    method: Literal["crop", "pad"] = "crop",
    bg_color: Tuple[int, int, int] = (0, 0, 0)
) -> None:
    """
    Process all images in a directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory where resized images will be saved
        target_size: Target size as (width, height)
        method: Resize method - 'crop' or 'pad'
        bg_color: Background color for padding as RGB tuple
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each file in the input directory
    count = 0
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                resize_image(input_path, output_path, target_size, method, bg_color)
                count += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    print(f"Processed {count} images.")


def parse_color(color_str: str) -> Tuple[int, int, int]:
    """Parse a color string in format 'r,g,b'."""
    try:
        r, g, b = map(int, color_str.split(','))
        return (r, g, b)
    except:
        raise argparse.ArgumentTypeError("Color must be in format 'r,g,b'")


def main():
    parser = argparse.ArgumentParser(description="Resize images to 720x1280 pixels")
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument("-o", "--output", help="Output image file or directory")
    parser.add_argument(
        "-m", "--method", 
        choices=["crop", "pad"], 
        default="crop",
        help="Resize method: crop (maintains aspect ratio and crops) or pad (adds padding)"
    )
    parser.add_argument(
        "-c", "--color", 
        type=parse_color, 
        default="0,0,0",
        help="Background color for padding in format 'r,g,b'"
    )
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if not args.output:
        if os.path.isdir(args.input):
            args.output = args.input + "_resized"
        else:
            base, ext = os.path.splitext(args.input)
            args.output = f"{base}_resized{ext}"
    
    # Process input (file or directory)
    if os.path.isdir(args.input):
        process_directory(args.input, args.output, (720, 1280), args.method, args.color)
    else:
        resize_image(args.input, args.output, (720, 1280), args.method, args.color)


if __name__ == "__main__":
    main() 