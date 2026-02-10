"""
Utility functions for LLM judge evaluation.

These utilities handle image processing, bounding box drawing, and patch sampling
for evaluating interpretability of visual tokens.
"""

import json
import os
import random
import numpy as np
from copy import deepcopy
from PIL import Image, ImageDraw
import torch
import torchvision.transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import convert_image_dtype

# Standard CLIP normalization values
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

def clip_bbox_to_image(bbox, image_width, image_height):
    """
    Clip bounding box coordinates to image boundaries.
    
    Args:
        bbox (tuple): (left, top, right, bottom) coordinates
        image_width (int): Width of the image
        image_height (int): Height of the image
        
    Returns:
        tuple: Clipped (left, top, right, bottom) coordinates
    """
    left, top, right, bottom = bbox
    
    # Clip to image boundaries
    left = max(0, left)
    top = max(0, top)
    right = min(image_width, right)
    bottom = min(image_height, bottom)
    
    return (left, top, right, bottom)

def draw_bbox_on_image(image, bbox, outline_color="red", width=3, fill_alpha=30):
    """
    Draw bounding box on image with better visibility for edge cases.
    
    Args:
        image: PIL Image object
        bbox (tuple): (left, top, right, bottom) coordinates
        outline_color (str): Color of the outline (default: "red")
        width (int): Width of the outline (default: 3)
        fill_alpha (int): Transparency for fill overlay (0-255, default: 30)
        
    Returns:
        PIL Image: Image with bounding box drawn
    """
    new_image = deepcopy(image)
    img_width, img_height = new_image.size
    
    # Clip bounding box to image boundaries
    clipped_bbox = clip_bbox_to_image(bbox, img_width, img_height)
    left, top, right, bottom = clipped_bbox
    
    # Check if bounding box is valid after clipping
    if left >= right or top >= bottom:
        return new_image  # Return original image if bbox is invalid
    
    draw = ImageDraw.Draw(new_image)
    
    # Draw filled rectangle with transparency for better visibility
    if fill_alpha > 0:
        # Create a semi-transparent overlay
        overlay = Image.new('RGBA', new_image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # Convert color name to RGB if needed
        if outline_color == "red":
            fill_color = (255, 0, 0, fill_alpha)
        elif outline_color == "blue":
            fill_color = (0, 0, 255, fill_alpha)
        elif outline_color == "green":
            fill_color = (0, 255, 0, fill_alpha)
        else:
            fill_color = (255, 0, 0, fill_alpha)  # Default to red
            
        overlay_draw.rectangle(clipped_bbox, fill=fill_color)
        
        # Composite the overlay with the original image
        new_image = new_image.convert('RGBA')
        new_image = Image.alpha_composite(new_image, overlay)
        new_image = new_image.convert('RGB')
        
        # Now draw the outline on top
        draw = ImageDraw.Draw(new_image)
    
    # Draw the outline
    draw.rectangle(clipped_bbox, outline=outline_color, width=width)
    
    # Add corner markers for better visibility at edges
    corner_size = min(10, width * 2)
    
    # Top-left corner
    if left == 0 or top == 0:
        draw.line([(left, top), (left + corner_size, top)], fill=outline_color, width=width + 1)
        draw.line([(left, top), (left, top + corner_size)], fill=outline_color, width=width + 1)
    
    # Top-right corner  
    if right == img_width or top == 0:
        draw.line([(right - corner_size, top), (right, top)], fill=outline_color, width=width + 1)
        draw.line([(right, top), (right, top + corner_size)], fill=outline_color, width=width + 1)
    
    # Bottom-left corner
    if left == 0 or bottom == img_height:
        draw.line([(left, bottom - corner_size), (left, bottom)], fill=outline_color, width=width + 1)
        draw.line([(left, bottom), (left + corner_size, bottom)], fill=outline_color, width=width + 1)
    
    # Bottom-right corner
    if right == img_width or bottom == img_height:
        draw.line([(right - corner_size, bottom), (right, bottom)], fill=outline_color, width=width + 1)
        draw.line([(right, bottom - corner_size), (right, bottom)], fill=outline_color, width=width + 1)
    
    return new_image

def calculate_expanded_bbox_from_patch(row, col, patch_size=24, width=1, height=1):
    """
    Calculate bounding box coordinates for a rectangular area of patches.
    
    Args:
        row (int): Starting row index of the patch
        col (int): Starting column index of the patch  
        patch_size (int): Size of each patch (default: 24)
        width (int): Number of patches to include horizontally (default: 1)
        height (int): Number of patches to include vertically (default: 1)
        
    Returns:
        tuple: (left, top, right, bottom) coordinates of the expanded bounding box
    """
    left = col * patch_size
    top = row * patch_size
    right = (col + width) * patch_size
    bottom = (row + height) * patch_size
    
    return (left, top, right, bottom)

def calculate_square_bbox_from_patch(row, col, patch_size=24, size=3):
    """
    Calculate bounding box coordinates for a square area of patches starting from a given patch.
    
    Args:
        row (int): Starting row index of the patch (top-left corner)
        col (int): Starting column index of the patch (top-left corner)
        patch_size (int): Size of each patch (default: 24)
        size (int): Size of the square area (e.g., 3 for 3x3, 5 for 5x5) (default: 3)
        
    Returns:
        tuple: (left, top, right, bottom) coordinates of the square bounding box
    """
    left = col * patch_size
    top = row * patch_size
    right = (col + size) * patch_size
    bottom = (row + size) * patch_size
    
    return (left, top, right, bottom)

def get_high_confidence_words(annotations, patch_start_row, patch_start_col, size=3, num_words=5):
    """
    Extract nearest neighbor tokens from annotations within a specified patch area.
    
    Args:
        annotations (list): List of annotation dictionaries, each containing patch info and nearest_neighbors
        patch_start_row (int): Starting row of the patch area
        patch_start_col (int): Starting column of the patch area 
        size (int): Size of the square area (e.g., 3 for 3x3) (default: 3)
        num_words (int): Number of words to return (default: 5)
        
    Returns:
        list: List of dictionaries, each containing token and similarity
    """
    target_row = patch_start_row + size // 2
    target_col = patch_start_col + size // 2

    for annotation in annotations:
        if annotation['patch_row'] == target_row and annotation['patch_col'] == target_col:
            return annotation['nearest_neighbors'][:num_words]

    return []


def pad_to_bounding_box(
    image, offset_height, offset_width, target_height,
    target_width, value=0
):
    height, width = image.shape[:2]
    after_padding_width = target_width - offset_width - width
    after_padding_height = target_height - offset_height - height
    return np.pad(image, [
        [offset_height, after_padding_height],
        [offset_width, after_padding_width],
        [0, 0]
    ], constant_values=value)


def normalize_image(image, offset, scale):
    image -= np.array(offset, dtype=np.float32)[None, None, :]
    image /= np.array(scale, dtype=np.float32)[None, None, :]
    return image


def resize_and_pad(
    image,
    desired_output_size,
    resize_method="torch-bilinear",
    pad_value=0,
    normalize=True,
    image_mean=OPENAI_CLIP_MEAN,
    image_std=OPENAI_CLIP_STD,
):
    desired_height, desired_width = desired_output_size
    height, width = image.shape[:2]

    # Cast into float32 since the training code did this in float32 and it (very rarely) effects
    # the results after rounding.
    image_scale_y = np.array(desired_height, np.float32) / np.array(height, np.float32)
    image_scale_x = np.array(desired_width, np.float32) / np.array(width, np.float32)
    image_scale = min(image_scale_x, image_scale_y)
    scaled_height = int(np.array(height, np.float32) * image_scale)
    scaled_width = int(np.array(width, np.float32) * image_scale)

    if resize_method == "tensorflow":
        # This how the original training code did resizing, it can produce slightly different
        # results then using torch resize so we keep it just in case
        import tensorflow as tf
        image = tf.image.convert_image_dtype(tf.constant(image), dtype=tf.float32)
        image = tf.image.resize(
            image,
            [scaled_height, scaled_width],
            method=tf.image.ResizeMethod.BILINEAR,
            antialias=True,
        )
        image = tf.clip_by_value(image, 0.0, 1.0)
        image = image.numpy()
    elif resize_method == "torch-bilinear":
        image = torch.permute(torch.from_numpy(image), [2, 0, 1])
        image = convert_image_dtype(image)  # resize in float32 to match the training code
        image = torchvision.transforms.Resize(
            [scaled_height, scaled_width], InterpolationMode.BILINEAR, antialias=True
        )(image)
        image = torch.clip(image, 0.0, 1.0)
        image = torch.permute(image, [1, 2, 0]).numpy()
    else:
        raise NotImplementedError(resize_method)

    top_pad = (desired_height - scaled_height) // 2
    left_pad = (desired_width - scaled_width) // 2
    padding = [
        [top_pad, desired_height - scaled_height - top_pad],
        [left_pad, desired_width - scaled_width - left_pad],
        [0, 0]
    ]
    image_mask = np.pad(np.ones_like(image[:, :, 0], dtype=bool), padding[:2])
    image = np.pad(image, padding, constant_values=pad_value)
    if normalize:
        image = normalize_image(image, offset=image_mean, scale=image_std)
    return image, image_mask


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return np.array(image)


def process_image_with_mask(image_path, model_name=None):
    """
    Process image and return both the processed image and the mask indicating real vs padded areas.

    Preprocessing must match what the model saw during inference:
      - CLIP (vit-l-14): aspect-preserving resize + black padding (mask has False for padded areas)
      - SigLIP:          squash-resize to square, no padding (mask all True)
      - DINOv2:          squash-resize to square, no padding (mask all True)
      - Qwen2-VL:        center-crop to square, no padding (mask all True)
      - Default (no model_name or unknown encoder): resize + pad (safe fallback for CLIP)

    Args:
        image_path (str): Path to the image file
        model_name (str, optional): Model name to determine preprocessing method.

    Returns:
        tuple: (processed_image, image_mask) where image_mask is True for real image areas
    """
    image = load_image(image_path)
    name_lower = model_name.lower() if model_name else ""

    if "qwen2vl" in name_lower or "qwen2-vl" in name_lower:
        # Qwen2-VL off-the-shelf → center-crop
        pil_image = Image.fromarray(image)
        width, height = pil_image.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        cropped = pil_image.crop((left, top, left + min_dim, top + min_dim))
        processed_image = cropped.resize((512, 512), Image.BILINEAR)
        image_mask = np.ones((512, 512), dtype=bool)
    elif "siglip" in name_lower or "dinov2" in name_lower:
        # SigLIP / DINOv2 → squash-resize to 512x512, no padding
        # Matches siglip_resize_and_pad / dino_resize_and_pad in model_preprocessor.py
        pil_image = Image.fromarray(image)
        processed_image = pil_image.resize((512, 512), Image.BILINEAR)
        image_mask = np.ones((512, 512), dtype=bool)
    else:
        # CLIP (vit-l-14) and default → aspect-preserving resize + black padding
        processed_image, image_mask = resize_and_pad(image, (512, 512), normalize=False)
        processed_image = (processed_image * 255).astype(np.uint8)
        processed_image = Image.fromarray(processed_image)

    return processed_image, image_mask


def sample_valid_patch_positions(image_mask, bbox_size=3, num_samples=36, grid_size=24):
    """
    Sample random patch positions that fall entirely within the real image area (not padded).

    Args:
        image_mask (np.ndarray): Boolean mask where True indicates real image areas
        bbox_size (int): Size of the bounding box in patches (e.g., 3 for 3x3)
        num_samples (int): Number of unique positions to sample
        grid_size (int): Grid size for the model (Molmo=24, Qwen2-VL=16)

    Returns:
        list: List of (row, col) tuples representing valid patch positions
    """
    # Convert 512x512 image mask to grid_size x grid_size patch grid
    patch_size = 512 // grid_size
    patch_mask = np.zeros((grid_size, grid_size), dtype=bool)

    # Check each patch position to see if it's entirely within the real image
    for row in range(grid_size):
        for col in range(grid_size):
            # Calculate pixel boundaries for this patch
            start_row = row * patch_size
            end_row = min((row + 1) * patch_size, 512)
            start_col = col * patch_size
            end_col = min((col + 1) * patch_size, 512)

            # Check if this patch is entirely within the real image
            patch_area = image_mask[start_row:end_row, start_col:end_col]
            if patch_area.all():  # All pixels in this patch are real (not padded)
                patch_mask[row, col] = True

    # Find valid positions where a bbox_size x bbox_size area can fit entirely in real image
    valid_positions = []
    for row in range(grid_size - bbox_size + 1):
        for col in range(grid_size - bbox_size + 1):
            # Check if bbox_size x bbox_size area starting at (row, col) is entirely valid
            bbox_area = patch_mask[row:row+bbox_size, col:col+bbox_size]
            if bbox_area.all():  # All patches in this bbox are in real image
                valid_positions.append((row, col))

    # Randomly sample from valid positions
    if len(valid_positions) < num_samples:
        print(f"Warning: Only {len(valid_positions)} valid positions found, but {num_samples} requested")
        return valid_positions

    return random.sample(valid_positions, num_samples)

