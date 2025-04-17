import streamlit as st
import torch
import numpy as np
import cv2
import supervision as sv
import os
import tempfile
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Drishya",
    page_icon="🖼️",
    layout="centered",
)

# Helper functions - moved to the top before they are used
def show_mask(ax, mask, random_color=False):
    """Display the mask on an axis."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6]) 
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(ax, box):
    """Display the bounding box on an axis."""
    x0, y0, x1, y1 = box
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor='red', facecolor=(0, 0, 0, 0), lw=2))

def apply_color_grading(product_image, target_image, mask, strength=0.5):
    """
    Apply color grading to make the product match the color tone of the target area.
    
    Args:
        product_image: The product image to adjust (numpy array)
        target_image: The target image containing the color tone to match (numpy array)
        mask: Binary mask of the target area (numpy array)
        strength: How strongly to apply the color grading (0.0-1.0)
    
    Returns:
        Color-graded product image
    """
    # Ensure mask is binary and has correct shape
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]
    binary_mask = (mask > 128).astype(np.uint8)
    
    # Get the region of interest from the target image based on the mask
    y_indices, x_indices = np.where(binary_mask == 1)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return product_image  # No adjustment if mask is empty
    
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()
    
    # Extract the target region
    target_region = target_image[y_min:y_max+1, x_min:x_max+1]
    
    # Create a mask for the target region
    local_mask = binary_mask[y_min:y_max+1, x_min:x_max+1]
    
    # Apply mask to target region to only consider pixels within the mask
    masked_target = target_region.copy()
    for c in range(3):  # Process each color channel
        masked_target[:,:,c] = masked_target[:,:,c] * local_mask
    
    # Calculate mean color of the masked target region
    target_pixels = local_mask.sum()
    if target_pixels == 0:
        return product_image  # No pixels to match
    
    target_means = [
        (masked_target[:,:,c].sum() / target_pixels) for c in range(3)
    ]
    
    # Calculate standard deviation of the target region for each channel
    target_std = [0, 0, 0]
    for c in range(3):
        # Calculate squared differences for non-zero mask pixels
        squared_diffs = np.zeros_like(local_mask, dtype=float)
        squared_diffs[local_mask > 0] = ((masked_target[:,:,c][local_mask > 0] - target_means[c]) ** 2)
        target_std[c] = np.sqrt(squared_diffs.sum() / target_pixels)
    
    # Handle alpha channel if present
    has_alpha = product_image.shape[2] == 4
    if has_alpha:
        alpha_channel = product_image[:,:,3].copy()
        product_rgb = product_image[:,:,:3].copy()
    else:
        product_rgb = product_image.copy()
    
    # Calculate mean and std of the product image (only for non-transparent pixels if has alpha)
    if has_alpha:
        # Only consider pixels that aren't fully transparent
        prod_mask = (alpha_channel > 0).astype(float)
        prod_pixels = prod_mask.sum()
        if prod_pixels == 0:
            return product_image  # No pixels to adjust
        
        product_means = [
            (product_rgb[:,:,c] * prod_mask).sum() / prod_pixels for c in range(3)
        ]
        
        product_std = [0, 0, 0]
        for c in range(3):
            squared_diffs = np.zeros_like(prod_mask, dtype=float)
            valid_pixels = prod_mask > 0
            if valid_pixels.sum() > 0:
                squared_diffs[valid_pixels] = ((product_rgb[:,:,c][valid_pixels] - product_means[c]) ** 2)
                product_std[c] = np.sqrt(squared_diffs.sum() / prod_pixels)
    else:
        h, w = product_rgb.shape[:2]
        prod_pixels = h * w
        product_means = [
            product_rgb[:,:,c].sum() / prod_pixels for c in range(3)
        ]
        
        product_std = [0, 0, 0]
        for c in range(3):
            squared_diffs = (product_rgb[:,:,c] - product_means[c]) ** 2
            product_std[c] = np.sqrt(squared_diffs.sum() / prod_pixels)
    
    # Perform color grading by adjusting mean and standard deviation
    graded_product = product_rgb.copy().astype(float)
    
    for c in range(3):
        # Skip channels with zero std to avoid division by zero
        if product_std[c] == 0:
            continue
            
        # Normalize the product channel
        normalized = (graded_product[:,:,c] - product_means[c]) / product_std[c]
        
        # Apply target statistics with the specified strength
        if strength < 1.0:
            # Blend between original and target values
            adj_std = product_std[c] * (1 - strength) + target_std[c] * strength
            adj_mean = product_means[c] * (1 - strength) + target_means[c] * strength
        else:
            adj_std = target_std[c]
            adj_mean = target_means[c]
            
        # Apply the adjustment
        graded_product[:,:,c] = normalized * adj_std + adj_mean
    
    # Clip values to valid range
    graded_product = np.clip(graded_product, 0, 255).astype(np.uint8)
    
    # Reattach alpha channel if needed
    if has_alpha:
        graded_product_with_alpha = np.zeros((graded_product.shape[0], graded_product.shape[1], 4), dtype=np.uint8)
        graded_product_with_alpha[:,:,:3] = graded_product
        graded_product_with_alpha[:,:,3] = alpha_channel
        return graded_product_with_alpha
    
    return graded_product

def replace_product_in_image(ad_image, new_product, mask, scale_factor=1.0):
    """
    Replace a product in an ad image based on a segmentation mask with improved sizing.
    
    Args:
        ad_image: The original advertisement image (numpy array)
        new_product: The new product image to insert (numpy array)
        mask: Binary segmentation mask of the product to replace (numpy array)
        scale_factor: Controls how much of the mask area the product will fill (1.0 = exact fit, >1.0 = larger than mask)
    
    Returns:
        The modified image with the new product inserted
    """
    # Ensure mask is binary
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]
    binary_mask = (mask > 128).astype(np.uint8)
    
    # Get bounding box from mask
    y_indices, x_indices = np.where(binary_mask == 1)
    if len(y_indices) == 0 or len(x_indices) == 0:
        st.error("Invalid mask - no pixels selected")
        return ad_image
        
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()
    
    # Calculate dimensions of the mask area
    mask_height = y_max - y_min + 1
    mask_width = x_max - x_min + 1
    
    # Create output image
    output = ad_image.copy()
    
    # Get product dimensions and aspect ratio
    prod_height, prod_width = new_product.shape[:2]
    prod_aspect_ratio = prod_width / prod_height
    mask_aspect_ratio = mask_width / mask_height
    
    # Calculate the dimensions to preserve aspect ratio
    # Apply scale factor to control how much of the mask the product fills
    # Use the minimum dimension to ensure consistent scaling 
    base_dimension = min(mask_width, mask_height)
    
    # Scale the base dimension
    scaled_dimension = base_dimension * scale_factor
    
    # Calculate new dimensions based on aspect ratio
    if prod_aspect_ratio > 1.0:
        # Product is wider than tall
        resize_width = scaled_dimension * prod_aspect_ratio
        resize_height = scaled_dimension
    else:
        # Product is taller than wide
        resize_width = scaled_dimension
        resize_height = scaled_dimension / prod_aspect_ratio
    
    # Calculate centering offsets
    offset_x = int((mask_width - resize_width) / 2)
    offset_y = int((mask_height - resize_height) / 2)
    
    # Handle transparent images (RGBA)
    if new_product.shape[2] == 4:
        # Extract alpha channel and RGB channels
        alpha = new_product[:, :, 3] / 255.0
        rgb = new_product[:, :, :3]
        
        # Resize both RGB and alpha to the calculated dimensions
        resize_width_int = max(1, int(resize_width))  # Ensure at least 1 pixel
        resize_height_int = max(1, int(resize_height))  # Ensure at least 1 pixel
        resized_rgb = cv2.resize(rgb, (resize_width_int, resize_height_int))
        resized_alpha = cv2.resize(alpha, (resize_width_int, resize_height_int))
        
        # Create a properly sized masks for handling the product overlay
        product_mask = np.zeros((mask_height, mask_width), dtype=np.float32)
        
        # Calculate paste coordinates (allowing overflow when scale > 1.0)
        paste_y_start = offset_y
        paste_y_end = paste_y_start + resize_height_int
        paste_x_start = offset_x
        paste_x_end = paste_x_start + resize_width_int
        
        # Calculate corresponding product coordinates (source coordinates)
        # Start with full product
        prod_y_start = 0
        prod_y_end = resize_height_int
        prod_x_start = 0
        prod_x_end = resize_width_int
        
        # If product would overflow the mask area, adjust the source coordinates
        # but keep the paste coordinates within the mask area
        if paste_y_start < 0:
            prod_y_start = -paste_y_start
            paste_y_start = 0
        if paste_y_end > mask_height:
            prod_y_end = resize_height_int - (paste_y_end - mask_height)
            paste_y_end = mask_height
        if paste_x_start < 0:
            prod_x_start = -paste_x_start
            paste_x_start = 0
        if paste_x_end > mask_width:
            prod_x_end = resize_width_int - (paste_x_end - mask_width)
            paste_x_end = mask_width
        
        # Place the resized alpha into the product mask (only the visible portion)
        if paste_y_end > paste_y_start and paste_x_end > paste_x_start:
            # Ensure we don't go out of bounds for the resized product
            prod_y_end = min(prod_y_end, resize_height_int)
            prod_x_end = min(prod_x_end, resize_width_int)
            
            # Place the visible portion of the alpha mask
            try:
                product_mask[paste_y_start:paste_y_end, paste_x_start:paste_x_end] = resized_alpha[prod_y_start:prod_y_end, prod_x_start:prod_x_end]
            except ValueError as e:
                st.error(f"Error placing alpha mask: {e}")
                return ad_image
        
        # Apply the binary mask to restrict blending only to the segmented region
        product_mask = product_mask * binary_mask[y_min:y_max+1, x_min:x_max+1]
        
        # Create 3-channel alpha for blending
        product_mask_3ch = np.stack([product_mask, product_mask, product_mask], axis=2)
        
        # Get the region of interest in the output image
        roi = output[y_min:y_max+1, x_min:x_max+1]
        
        # Create a properly sized RGB image to blend
        rgb_to_blend = np.zeros((mask_height, mask_width, 3), dtype=np.uint8)
        if paste_y_end > paste_y_start and paste_x_end > paste_x_start:
            try:
                rgb_to_blend[paste_y_start:paste_y_end, paste_x_start:paste_x_end] = resized_rgb[prod_y_start:prod_y_end, prod_x_start:prod_x_end]
            except ValueError as e:
                st.error(f"Error placing RGB image: {e}")
                return ad_image
        
        # Blend using the combined alpha
        blended_roi = roi * (1 - product_mask_3ch) + rgb_to_blend * product_mask_3ch
        
        # Place the blended region back into the output image
        output[y_min:y_max+1, x_min:x_max+1] = blended_roi
    else:
        # For non-transparent images, use a similar approach
        # Resize the product image
        resize_width_int = max(1, int(resize_width))
        resize_height_int = max(1, int(resize_height))
        resized_product = cv2.resize(new_product, (resize_width_int, resize_height_int))
        
        # Create a properly sized product image to blend
        product_to_blend = np.zeros((mask_height, mask_width, 3), dtype=np.uint8)
        
        # Calculate paste coordinates (allowing overflow when scale > 1.0)
        paste_y_start = offset_y
        paste_y_end = paste_y_start + resize_height_int
        paste_x_start = offset_x
        paste_x_end = paste_x_start + resize_width_int
        
        # Calculate corresponding product coordinates (source coordinates)
        # Start with full product
        prod_y_start = 0
        prod_y_end = resize_height_int
        prod_x_start = 0
        prod_x_end = resize_width_int
        
        # If product would overflow the mask area, adjust the source coordinates
        # but keep the paste coordinates within the mask area
        if paste_y_start < 0:
            prod_y_start = -paste_y_start
            paste_y_start = 0
        if paste_y_end > mask_height:
            prod_y_end = resize_height_int - (paste_y_end - mask_height)
            paste_y_end = mask_height
        if paste_x_start < 0:
            prod_x_start = -paste_x_start
            paste_x_start = 0
        if paste_x_end > mask_width:
            prod_x_end = resize_width_int - (paste_x_end - mask_width)
            paste_x_end = mask_width
        
        # Place the resized product into the blend image (only the visible portion)
        if paste_y_end > paste_y_start and paste_x_end > paste_x_start:
            # Ensure we don't go out of bounds for the resized product
            prod_y_end = min(prod_y_end, resize_height_int)
            prod_x_end = min(prod_x_end, resize_width_int)
            
            try:
                product_to_blend[paste_y_start:paste_y_end, paste_x_start:paste_x_end] = resized_product[prod_y_start:prod_y_end, prod_x_start:prod_x_end]
            except ValueError as e:
                st.error(f"Error placing product: {e}")
                return ad_image
        
        # Create a mask for the target region
        product_mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
        if paste_y_end > paste_y_start and paste_x_end > paste_x_start:
            product_mask[paste_y_start:paste_y_end, paste_x_start:paste_x_end] = 1
        
        # Apply the binary mask to restrict blending only to the segmented region
        product_mask = product_mask * binary_mask[y_min:y_max+1, x_min:x_max+1]
        product_mask_3ch = np.stack([product_mask, product_mask, product_mask], axis=2)
        
        # Get the region of interest in the output image
        roi = output[y_min:y_max+1, x_min:x_max+1]
        
        # Blend the new product with the region of interest using the mask
        blended_roi = roi * (1 - product_mask_3ch) + product_to_blend * product_mask_3ch
        
        # Place the blended region back into the output image
        output[y_min:y_max+1, x_min:x_max+1] = blended_roi
    
    return output

# Sidebar setup
st.sidebar.image("logo setu.jpeg", width=50, use_container_width=True)

# App title and description
st.title("Drishya")
st.markdown("""
Product Image Replacement Tool built using Segment Anything Model (SAM) 
1. Upload an advertisement image
2. Draw a bounding box around the product you want to replace
3. Generate segmentation mask
4. Upload a new product image to replace the original
5. Download the final image
""")

@st.cache_resource
def load_model():
    """Load the SAM model from a local file"""
    # Check if CUDA is available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Path to the local model file (in the same directory as the app)
    checkpoint_path = "/Users/akshatmajila/SAM-Setu/sam_vit_h_4b8939.pth"
    
    # Check if the model file exists
    if not os.path.isfile(checkpoint_path):
        st.error(f"Model file not found at {checkpoint_path}. Please make sure the SAM model file is in the same directory as this app.")
        st.stop()
    
    # Load the model
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device=device)
    mask_predictor = SamPredictor(sam)
    
    return mask_predictor, device

# Load the model
mask_predictor, device = load_model()
st.success("Model loaded successfully")

# Create session state for storing data between reruns
if 'generated_masks' not in st.session_state:
    st.session_state.generated_masks = None
if 'best_mask_idx' not in st.session_state:
    st.session_state.best_mask_idx = None
if 'box' not in st.session_state:
    st.session_state.box = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'binary_mask' not in st.session_state:
    st.session_state.binary_mask = None

# Upload an image
st.sidebar.header("Upload Image")
uploaded_ad_file = st.sidebar.file_uploader("Upload 4o Generated Image Here", type=["jpg", "jpeg", "png"], key="ad_image")

if uploaded_ad_file is not None:
    # Read the image
    image = Image.open(uploaded_ad_file)
    image_np = np.array(image)
    
    # Save original image to session state
    st.session_state.original_image = image_np.copy()
    
    # Convert image to RGB if it's RGBA
    if image_np.shape[-1] == 4:  # RGBA
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    
    # Display the uploaded image
    st.header("Step 1: Define Bounding Box")
    st.image(image_np, caption="Uploaded Advertisement Image", use_container_width=True)
    
    # Get image dimensions
    height, width = image_np.shape[:2]
    
    # Create columns for sliders
    col1, col2 = st.columns(2)
    
    # Bounding box inputs
    with col1:
        st.subheader("X-axis Controls")
        x_min = st.slider("X Min", 0, width-1, 200)
        x_max = st.slider("X Max", x_min+1, width, 800)
    
    with col2:
        st.subheader("Y-axis Controls")
        y_min = st.slider("Y Min", 0, height-1, 200)
        y_max = st.slider("Y Max", y_min+1, height, 800)
    
    # Convert the image to RGB for SAM
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB) if len(image_np.shape) == 3 else cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    
    # Display image with bounding box
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image_rgb)
    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)
    ax.axis('off')
    st.pyplot(fig)
    
    # Process the image with SAM
    if st.button("Generate Segmentation Masks"):
        with st.spinner("Processing image with SAM..."):
            # Define the bounding box
            box = np.array([x_min, y_min, x_max, y_max])
            st.session_state.box = box
            
            # Set the image for the predictor
            mask_predictor.set_image(image_rgb)
            
            # Generate masks
            masks, scores, logits = mask_predictor.predict(
                box=box,
                multimask_output=True
            )
            
            # Save generated masks to session state
            st.session_state.generated_masks = masks
            st.session_state.best_mask_idx = np.argmax(scores)
        
        st.header("Step 2: View Segmentation Results")
        
        # Display the masks
        st.subheader("Generated Masks")
        fig, axes = plt.subplots(1, len(masks), figsize=(15, 5))
        if len(masks) == 1:
            axes = [axes]
            
        for i, (mask, score) in enumerate(zip(masks, scores)):
            axes[i].imshow(image_rgb)
            show_mask(axes[i], mask, random_color=False)
            axes[i].title.set_text(f"Mask {i+1} (Score: {score:.3f})")
            axes[i].axis('off')
        st.pyplot(fig)
        
        # Display the best mask applied to the image
        st.subheader("Best Mask Applied to Image")
        best_mask_idx = np.argmax(scores)
        
        # Create a visualization of the mask on the image
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(image_rgb)
        show_mask(ax, masks[best_mask_idx], random_color=False)
        show_box(ax, box)
        ax.axis('off')
        st.pyplot(fig)
        
        # Create a binary mask image
        binary_mask = masks[best_mask_idx].astype(np.uint8) * 255
        st.session_state.binary_mask = binary_mask
        st.image(binary_mask, caption="Binary Mask (Best Score)", use_container_width=True)
        
        # Option to download the mask
        mask_pil = Image.fromarray(binary_mask)
        buf = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        mask_pil.save(buf.name)
        
        with open(buf.name, 'rb') as f:
            st.download_button(
                label="Download Mask",
                data=f,
                file_name="segmentation_mask.png",
                mime="image/png"
            )

# Replace lines ~458-520 with this correctly indented version:

# Product replacement section (only shown after mask generation)
if st.session_state.binary_mask is not None:
    st.header("Step 3: Replace Product")
    
    # Upload replacement product image
    uploaded_product = st.file_uploader("Upload New Product Image (preferably with transparent background)", type=["png", "jpg", "jpeg"], key="product_image")
    
    # Scale factor slider - now properly indented
    scale_factor = st.slider(
        "Product Scale", 
        min_value=0.5, 
        max_value=2.0,
        value=1.0, 
        step=0.05,
        help="Control product size relative to mask (1.0 = exact fit, >1.0 = larger than mask)"
    )

    st.subheader("Color Grading Options")
    enable_color_grading = st.checkbox("Enable Color Grading", value=True, 
                                      help="Apply color adjustment to match product color tone with the background")

    color_grade_strength = st.slider(
        "Color Grade Strength", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="How strongly to apply the color grading (0 = original colors, 1 = full match)"
    )

    # Color grading method selection
    grading_method = st.radio(
        "Color Grading Method",
        ["Match Target Area", "Match Entire Image"],
        help="Choose whether to match colors with just the target area or the entire image"
    )

    if uploaded_product is not None:
        # Read the new product image
        new_product_img = Image.open(uploaded_product)
        new_product_np = np.array(new_product_img)
        
        # Display the new product image
        st.image(new_product_np, caption="New Product Image", width=300)
        
        # Perform the replacement
        if st.button("Replace Product"):
            with st.spinner("Replacing product..."):
                # Ensure the original image is in the correct format
                original_img = st.session_state.original_image
                if original_img.shape[-1] == 4:  # RGBA
                    original_img = cv2.cvtColor(original_img, cv2.COLOR_RGBA2RGB)
                
                # Ensure new product is in the correct format
                if len(new_product_np.shape) == 2:  # Grayscale
                    new_product_np = cv2.cvtColor(new_product_np, cv2.COLOR_GRAY2RGB)
                
                # Apply color grading if enabled
                graded_product = new_product_np.copy()
                if enable_color_grading:
                    if grading_method == "Match Target Area":
                        # Use the mask area for color matching
                        graded_product = apply_color_grading(
                            new_product_np, 
                            original_img, 
                            st.session_state.binary_mask, 
                            color_grade_strength
                        )
                    else:  # Match Entire Image
                        # Create a full-image mask for color matching with the entire image
                        full_mask = np.ones(original_img.shape[:2], dtype=np.uint8) * 255
                        graded_product = apply_color_grading(
                            new_product_np,
                            original_img,
                            full_mask,
                            color_grade_strength
                        )
                    
                    # Display the color-graded product
                    st.subheader("Color-Graded Product")
                    st.image(graded_product, caption="Product After Color Grading", width=300)
                
                # Replace the product
                result_image = replace_product_in_image(
                    original_img,
                    graded_product,  # Use the color-graded product instead of the original
                    st.session_state.binary_mask,
                    scale_factor
                )
                
                # Display the result
                st.subheader("Final Result")
                st.image(result_image, caption="Advertisement with Replaced Product", use_container_width=True)
                
                # Save the result to a temporary file for download
                result_pil = Image.fromarray(result_image)
                result_buf = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                result_pil.save(result_buf.name)
                
                with open(result_buf.name, 'rb') as f:
                    st.download_button(
                        label="Download Final Image",
                        data=f,
                        file_name="product_replaced_ad.png",
                        mime="image/png"
                    )
    else:
        # If no product image is uploaded, disable the button
        st.button("Replace Product", disabled=True, help="Please upload a product image first")


st.markdown("---")
st.markdown("Created with ❤️ using Streamlit and Meta's Segment Anything Model (SAM)")