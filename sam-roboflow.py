import streamlit as st
import torch
import numpy as np
import cv2
import os
import tempfile
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

# Set page configuration to wide layout
st.set_page_config(
    page_title="Drishya",
    page_icon="🖼️",
    layout="wide",
)

# Password protection
def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == "setuftw":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.text_input(
            "Enter the magic words", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input + error
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct
        return True

if not check_password():
    st.stop()  # Stop execution if password is incorrect

# Helper functions
def show_mask(mask, image):
    """Apply mask on image for visualization."""
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    # Create a visualization with mask overlay
    result = image.copy()
    mask_rgb = (mask_image[:,:,:3] * 255).astype(np.uint8)
    mask_alpha = (mask_image[:,:,3:] * 255).astype(np.uint8)
    
    # Blend where mask exists
    alpha_channel = mask_alpha / 255.0
    for c in range(3):
        result[:,:,c] = result[:,:,c] * (1 - alpha_channel[:,:,0]) + mask_rgb[:,:,c] * alpha_channel[:,:,0]
    
    return result

def create_feathered_mask(mask, feather_amount=10):
    """Create a feathered mask with smooth edges for better blending."""
    # Ensure mask is binary
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]
    binary_mask = (mask > 128).astype(np.uint8)
    
    # Create distance transform from the mask edges
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 3)
    
    # Normalize the distance transform
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
    
    # Create inverse distance transform for outside the mask
    inv_binary_mask = 1 - binary_mask
    inv_dist_transform = cv2.distanceTransform(inv_binary_mask, cv2.DIST_L2, 3)
    cv2.normalize(inv_dist_transform, inv_dist_transform, 0, 1.0, cv2.NORM_MINMAX)
    
    # Create a feathered mask by combining both distance transforms
    feathered_mask = np.ones_like(dist_transform, dtype=np.float32)
    
    # Apply feathering at the boundaries
    feathered_mask = np.where(
        dist_transform > 0,
        np.minimum(1.0, dist_transform * (feather_amount / 2)),
        feathered_mask
    )
    
    feathered_mask = np.where(
        inv_dist_transform < feather_amount,
        np.maximum(0.0, 1.0 - (inv_dist_transform / feather_amount)),
        feathered_mask * binary_mask
    )
    
    return feathered_mask

def apply_color_grading(product_image, target_image, mask, strength=0.5):
    """Apply color grading to make the product match the color tone of the target area."""
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

def replace_product_in_image(ad_image, new_product, mask, scale_factor=1.0, feather_amount=15, use_blending=True):
    """
    Replace a product in an ad image with improved edge blending.
    
    Args:
        ad_image: The original advertisement image (numpy array)
        new_product: The new product image to insert (numpy array)
        mask: Binary segmentation mask of the product to replace (numpy array)
        scale_factor: Controls how much of the mask area the product will fill
        feather_amount: Amount of edge feathering in pixels
        use_blending: Whether to apply edge blending or not
        
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
    
    # Calculate the dimensions to preserve aspect ratio
    base_dimension = min(mask_width, mask_height)
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
    
    # Create a feathered mask for better edge blending if blending is enabled
    feathered_mask = binary_mask[y_min:y_max+1, x_min:x_max+1].astype(np.float32)
    if use_blending:
        feathered_mask = create_feathered_mask(binary_mask[y_min:y_max+1, x_min:x_max+1], feather_amount)
    
    # Handle transparent images (RGBA)
    if new_product.shape[2] == 4:
        # Extract alpha channel and RGB channels
        alpha = new_product[:, :, 3] / 255.0
        rgb = new_product[:, :, :3]
        
        # Resize both RGB and alpha to the calculated dimensions
        resize_width_int = max(1, int(resize_width))
        resize_height_int = max(1, int(resize_height))
        resized_rgb = cv2.resize(rgb, (resize_width_int, resize_height_int))
        resized_alpha = cv2.resize(alpha, (resize_width_int, resize_height_int))
        
        # Create a properly sized masks for handling the product overlay
        product_mask = np.zeros((mask_height, mask_width), dtype=np.float32)
        
        # Calculate paste coordinates
        paste_y_start = offset_y
        paste_y_end = paste_y_start + resize_height_int
        paste_x_start = offset_x
        paste_x_end = paste_x_start + resize_width_int
        
        # Calculate corresponding product coordinates
        prod_y_start = 0
        prod_y_end = resize_height_int
        prod_x_start = 0
        prod_x_end = resize_width_int
        
        # Handle overflow
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
        
        # Place the resized alpha into the product mask
        if paste_y_end > paste_y_start and paste_x_end > paste_x_start:
            prod_y_end = min(prod_y_end, resize_height_int)
            prod_x_end = min(prod_x_end, resize_width_int)
            
            product_mask[paste_y_start:paste_y_end, paste_x_start:paste_x_end] = resized_alpha[prod_y_start:prod_y_end, prod_x_start:prod_x_end]
        
        # Apply the binary mask and then the feathered mask for smooth edges if blending is enabled
        product_mask = product_mask * feathered_mask
        
        # Create 3-channel alpha for blending
        product_mask_3ch = np.stack([product_mask, product_mask, product_mask], axis=2)
        
        # Get the region of interest in the output image
        roi = output[y_min:y_max+1, x_min:x_max+1]
        
        # Create a properly sized RGB image to blend
        rgb_to_blend = np.zeros((mask_height, mask_width, 3), dtype=np.uint8)
        if paste_y_end > paste_y_start and paste_x_end > paste_x_start:
            rgb_to_blend[paste_y_start:paste_y_end, paste_x_start:paste_x_end] = resized_rgb[prod_y_start:prod_y_end, prod_x_start:prod_x_end]
        
        # Basic alpha blending
        blended_roi = roi * (1 - product_mask_3ch) + rgb_to_blend * product_mask_3ch
        
        if use_blending:
            # Edge detection on the original mask to identify boundary regions
            edge_kernel = np.ones((5, 5), np.uint8)
            edge_mask = cv2.dilate(binary_mask[y_min:y_max+1, x_min:x_max+1], edge_kernel) - binary_mask[y_min:y_max+1, x_min:x_max+1]
            edge_mask = np.clip(edge_mask, 0, 1).astype(np.float32)
            
            # Apply guided filtering for improved edge transitions
            try:
                r = 5  # Filter radius
                eps = 0.1  # Regularization parameter
                
                harmonized_blend = blended_roi.copy()
                for c in range(3):
                    harmonized_blend[:,:,c] = cv2.guidedFilter(
                        roi[:,:,c].astype(np.float32), 
                        blended_roi[:,:,c].astype(np.float32),
                        r, eps
                    )
                
                blended_roi = harmonized_blend.astype(np.uint8)
            except:
                # Fallback to simple blurring at the edges
                blur_amount = 3
                edge_blur = cv2.GaussianBlur(edge_mask, (blur_amount*2+1, blur_amount*2+1), 0) * 0.7
                edge_blur_3ch = np.stack([edge_blur, edge_blur, edge_blur], axis=2)
                
                harmonized_blend = blended_roi * (1 - edge_blur_3ch) + cv2.GaussianBlur(blended_roi, (blur_amount*2+1, blur_amount*2+1), 0) * edge_blur_3ch
                blended_roi = harmonized_blend.astype(np.uint8)
        
        # Place the blended region back into the output image
        output[y_min:y_max+1, x_min:x_max+1] = blended_roi
    else:
        # For non-transparent images - similar approach
        # Resize the product image
        resize_width_int = max(1, int(resize_width))
        resize_height_int = max(1, int(resize_height))
        resized_product = cv2.resize(new_product, (resize_width_int, resize_height_int))
        
        # Create a properly sized product image to blend
        product_to_blend = np.zeros((mask_height, mask_width, 3), dtype=np.uint8)
        
        # Calculate paste coordinates
        paste_y_start = offset_y
        paste_y_end = paste_y_start + resize_height_int
        paste_x_start = offset_x
        paste_x_end = paste_x_start + resize_width_int
        
        # Calculate corresponding product coordinates
        prod_y_start = 0
        prod_y_end = resize_height_int
        prod_x_start = 0
        prod_x_end = resize_width_int
        
        # Handle overflow
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
        
        # Place the resized product into the blend image
        if paste_y_end > paste_y_start and paste_x_end > paste_x_start:
            prod_y_end = min(prod_y_end, resize_height_int)
            prod_x_end = min(prod_x_end, resize_width_int)
            
            product_to_blend[paste_y_start:paste_y_end, paste_x_start:paste_x_end] = resized_product[prod_y_start:prod_y_end, prod_x_start:prod_x_end]
        
        # Apply the feathered mask for smooth edges
        product_mask = np.zeros((mask_height, mask_width), dtype=np.float32)
        if paste_y_end > paste_y_start and paste_x_end > paste_x_start:
            product_mask[paste_y_start:paste_y_end, paste_x_start:paste_x_end] = 1
        
        # Combine with feathered mask
        product_mask = product_mask * feathered_mask
        product_mask_3ch = np.stack([product_mask, product_mask, product_mask], axis=2)
        
        # Get the region of interest
        roi = output[y_min:y_max+1, x_min:x_max+1]
        
        # Basic alpha blending
        blended_roi = roi * (1 - product_mask_3ch) + product_to_blend * product_mask_3ch
        
        if use_blending:
            # Edge detection to identify boundary regions
            edge_kernel = np.ones((5, 5), np.uint8)
            edge_mask = cv2.dilate(binary_mask[y_min:y_max+1, x_min:x_max+1], edge_kernel) - binary_mask[y_min:y_max+1, x_min:x_max+1]
            edge_mask = np.clip(edge_mask, 0, 1).astype(np.float32)
            
            # Apply guided filtering for improved edge transitions
            try:
                r = 5  # Filter radius
                eps = 0.1  # Regularization parameter
                
                harmonized_blend = blended_roi.copy()
                for c in range(3):
                    harmonized_blend[:,:,c] = cv2.guidedFilter(
                        roi[:,:,c].astype(np.float32), 
                        blended_roi[:,:,c].astype(np.float32),
                        r, eps
                    )
                
                blended_roi = harmonized_blend.astype(np.uint8)
            except:
                # Fallback to simple blurring at the edges
                blur_amount = 3
                edge_blur = cv2.GaussianBlur(edge_mask, (blur_amount*2+1, blur_amount*2+1), 0) * 0.7
                edge_blur_3ch = np.stack([edge_blur, edge_blur, edge_blur], axis=2)
                
                harmonized_blend = blended_roi * (1 - edge_blur_3ch) + cv2.GaussianBlur(blended_roi, (blur_amount*2+1, blur_amount*2+1), 0) * edge_blur_3ch
                blended_roi = harmonized_blend.astype(np.uint8)
        
        # Place the blended region back into the output image
        output[y_min:y_max+1, x_min:x_max+1] = blended_roi
    
    return output

@st.cache_resource
def load_model():
    """Load the SAM model from a local file"""
    # Check if CUDA is available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Path to the model file - look in current directory first, then try common locations
    model_filename = "sam_vit_b_01ec64.pth"
    possible_paths = [
        model_filename,  # Current directory
        os.path.join("models", model_filename),  # models subdirectory
        os.path.join(os.path.dirname(__file__), model_filename),  # Same directory as script
        os.path.join(os.path.dirname(__file__), "models", model_filename)  # models subdirectory relative to script
    ]
    
    # Try to find the model file
    checkpoint_path = None
    for path in possible_paths:
        if os.path.isfile(path):
            checkpoint_path = path
            break
    
    # If model file not found, show error
    if checkpoint_path is None:
        st.error(f"Model file '{model_filename}' not found. Please upload the SAM model file.")
        # Add file uploader for model
        uploaded_model = st.file_uploader("Upload SAM model file (sam_vit_b_01ec64.pth)", type=["pth"])
        if uploaded_model is not None:
            # Save the uploaded model to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
                tmp_file.write(uploaded_model.getvalue())
                checkpoint_path = tmp_file.name
        else:
            st.stop()
    
    # Load the model with error handling
    model_type = "vit_b"
    try:
        # Try loading with map_location to ensure compatibility
        sam = sam_model_registry[model_type](
            checkpoint=checkpoint_path, 
            map_location=device
        ).to(device=device)
        mask_predictor = SamPredictor(sam)
        return mask_predictor, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("This may be due to a version mismatch between the model and PyTorch. Try uploading a compatible model file.")
        
        # Add file uploader for model
        uploaded_model = st.file_uploader("Upload compatible SAM model file", type=["pth"], key="retry_upload")
        if uploaded_model is not None:
            # Save the uploaded model to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
                tmp_file.write(uploaded_model.getvalue())
                try:
                    # Try loading with safe_load
                    sam = sam_model_registry[model_type](
                        checkpoint=tmp_file.name,
                        map_location=device
                    ).to(device=device)
                    mask_predictor = SamPredictor(sam)
                    return mask_predictor, device
                except Exception as e2:
                    st.error(f"Still unable to load model: {str(e2)}")
                    st.stop()
        st.stop()

# App title and description
st.title("Drishya - Product Image Replacement Tool")
st.markdown("""
A tool that lets you replace products in images using AI segmentation. Follow these steps:
1. Upload an image
2. Draw a box around the product to replace
3. Upload a new product image
4. Adjust settings and download the result
""")

# Create session state for storing data between reruns
if 'generated_mask' not in st.session_state:
    st.session_state.generated_mask = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'box_drawn' not in st.session_state:
    st.session_state.box_drawn = None
if 'mask_displayed' not in st.session_state:
    st.session_state.mask_displayed = False
if 'processing_step' not in st.session_state:
    st.session_state.processing_step = 1  # Track the current step

# Step 1: Image upload
st.header("Step 1: Upload Image")
uploaded_ad_file = st.file_uploader("Upload image with product to replace", type=["jpg", "jpeg", "png"], key="ad_image")

if uploaded_ad_file is not None:
    # Load the SAM model
    with st.spinner("Loading AI model..."):
        mask_predictor, device = load_model()
    
    # Read the image
    image = Image.open(uploaded_ad_file)
    image_np = np.array(image)
    
    # Save original image to session state
    st.session_state.original_image = image_np.copy()
    
    # Convert image to RGB if it's RGBA
    if image_np.shape[-1] == 4:  # RGBA
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    else:
        image_rgb = image_np.copy()
    
    # Step 2: Draw bounding box
    st.header("Step 2: Draw Bounding Box")
    st.markdown("Draw a box around the product you want to replace")
    
    # Canvas for drawing
    # Set up a reasonable canvas size based on image dimensions
    h, w = image_rgb.shape[:2]
    canvas_width = min(800, w)
    canvas_height = int(h * (canvas_width / w))
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.2)",
        stroke_width=2,
        stroke_color="rgba(255, 0, 0, 1)",
        background_image=Image.fromarray(image_rgb),
        height=canvas_height,
        width=canvas_width,
        drawing_mode="rect",
        key="canvas",
        update_streamlit=True,
    )

    # Check if a bounding box was drawn
    if canvas_result.json_data is not None and "objects" in canvas_result.json_data:
        objects = canvas_result.json_data["objects"]
        
        if objects:
            # Get the box coordinates from the first object
            rect = objects[0]
            box_height = rect.get("height", 0) * (h / canvas_height)
            box_width = rect.get("width", 0) * (w / canvas_width)
            box_left = rect.get("left", 0) * (w / canvas_width)
            box_top = rect.get("top", 0) * (h / canvas_height)
            
            # Convert to the [x_min, y_min, x_max, y_max] format needed by SAM
            x_min = max(0, int(box_left))
            y_min = max(0, int(box_top))
            x_max = min(w, int(box_left + box_width))
            y_max = min(h, int(box_top + box_height))
            
            # Save the box coordinates
            st.session_state.box_drawn = [x_min, y_min, x_max, y_max]
            
            # Display the box coordinates
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Box Coordinates: X({x_min}, {x_max}), Y({y_min}, {y_max})")
            
            with col2:
                if st.button("Generate Mask", key="generate_mask"):
                    with st.spinner("Processing image with AI..."):
                        # Set the image for the predictor
                        mask_predictor.set_image(image_rgb)
                        
                        # Generate masks
                        masks, scores, logits = mask_predictor.predict(
                            box=np.array([x_min, y_min, x_max, y_max]),
                            multimask_output=True
                        )
                        
                        # Get best mask by score
                        best_mask_idx = np.argmax(scores)
                        binary_mask = masks[best_mask_idx].astype(np.uint8) * 255
                        
                        # Save to session state
                        st.session_state.generated_mask = binary_mask
                        st.session_state.mask_displayed = True
                        st.session_state.processing_step = 3  # Advance to next step
                        
                        # Force re-run to update the UI
                        st.experimental_rerun()

    # Step 3: Show mask and allow product upload
    if st.session_state.mask_displayed and st.session_state.generated_mask is not None:
        st.header("Step 3: Mask Generated")
        
        # Create two columns to display the original image and the mask side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Image**")
            st.image(image_rgb, use_column_width=True)
        
        with col2:
            st.markdown("**Generated Mask**")
            # Create visualization with mask overlay
            mask_vis = show_mask(st.session_state.generated_mask, image_rgb)
            st.image(mask_vis, use_column_width=True)
        
        # Step 4: Product replacement
        st.header("Step 4: Replace Product")
        
        # Upload replacement product image
        uploaded_product = st.file_uploader("Upload New Product Image", type=["png", "jpg", "jpeg"], key="product_image")
        
        if uploaded_product is not None:
            # Read the new product image
            new_product_img = Image.open(uploaded_product)
            new_product_np = np.array(new_product_img)
            
            # Create a row for the product preview and settings
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**New Product Image**")
                prod_h, prod_w = new_product_np.shape[:2]
                st.image(new_product_np, width=int(prod_w/2), use_column_width=False)
            
            with col2:
                # Scale factor slider
                scale_factor = st.slider(
                    "Product Scale", 
                    min_value=0.5, 
                    max_value=2.0,
                    value=1.0, 
                    step=0.05,
                    help="Control product size relative to mask (1.0 = exact fit, >1.0 = larger than mask)"
                )
                
                # Blending options
                use_blending = st.checkbox("Use Edge Blending", value=True, 
                                         help="Enable advanced edge blending (uncheck to keep product edges as-is)")
                
                # Only show feathering slider if blending is enabled
                feather_amount = 15  # Default value
                if use_blending:
                    feather_amount = st.slider(
                        "Edge Feathering Amount", 
                        min_value=0, 
                        max_value=30, 
                        value=15, 
                        step=1,
                        help="Controls the softness of edges (higher = softer transitions)"
                    )
                
                # Collapsible UI for Color Grading Options
                with st.expander("Color Grading Options"):
                    enable_color_grading = st.checkbox("Enable Color Grading", value=True, 
                                                    help="Apply color adjustment to match product color tone with the background")
                    
                    # Only show strength slider if color grading is enabled
                    color_grade_strength = 0.5  # Default value
                    if enable_color_grading:
                        color_grade_strength = st.slider(
                            "Color Grade Strength", 
                            min_value=0.0, 
                            max_value=1.0, 
                            value=0.5, 
                            step=0.05,
                            help="How strongly to apply the color grading (0 = original colors, 1 = full match)"
                        )
                        
                        grading_method = st.radio(
                            "Color Grading Method",
                            ["Match Target Area", "Match Entire Image"],
                            help="Choose whether to match colors with just the target area or the entire image"
                        )
            
            # Replace button
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
                                st.session_state.generated_mask, 
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
                    
                    # Replace the product with improved edge blending
                    result_image = replace_product_in_image(
                        original_img,
                        graded_product,
                        st.session_state.generated_mask,
                        scale_factor,
                        feather_amount,
                        use_blending
                    )
                    
                    # Display the results side by side
                    st.header("Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Original Image**")
                        st.image(original_img, use_column_width=True)
                    
                    with col2:
                        st.markdown("**Replaced Product**")
                        st.image(result_image, use_column_width=True)
                    
                    # Save the result to a temporary file for download
                    result_pil = Image.fromarray(result_image)
                    result_buf = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                    result_pil.save(result_buf.name)
                    
                    with open(result_buf.name, 'rb') as f:
                        st.download_button(
                            label="Download Final Image",
                            data=f,
                            file_name="product_replaced_image.png",
                            mime="image/png"
                        )

# Add footer
st.markdown("---")
st.markdown("Created with ❤️ using Streamlit and Meta's Segment Anything Model (SAM)")
