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

def replace_product_in_image(ad_image, new_product, mask):
    """
    Replace a product in an ad image based on a segmentation mask.
    
    Args:
        ad_image: The original advertisement image (numpy array)
        new_product: The new product image to insert (numpy array)
        mask: Binary segmentation mask of the product to replace (numpy array)
    
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
    target_height = y_max - y_min + 1
    target_width = x_max - x_min + 1
    
    # Create output image
    output = ad_image.copy()
    
    # Handle transparent images (RGBA)
    if new_product.shape[2] == 4:
        # Extract alpha channel and RGB channels
        alpha = new_product[:, :, 3] / 255.0
        rgb = new_product[:, :, :3]
        
        # Calculate the original product aspect ratio
        orig_height, orig_width = new_product.shape[:2]
        orig_aspect_ratio = orig_width / orig_height
        mask_aspect_ratio = target_width / target_height
        
        # Determine the resize approach to ensure product fills the entire mask
        if orig_aspect_ratio > mask_aspect_ratio:
            # Product is wider than mask - match height and crop width
            resize_height = target_height
            resize_width = int(resize_height * orig_aspect_ratio)
            offset_x = (resize_width - target_width) // 2
            offset_y = 0
        else:
            # Product is taller than mask - match width and crop height
            resize_width = target_width
            resize_height = int(resize_width / orig_aspect_ratio)
            offset_x = 0
            offset_y = (resize_height - target_height) // 2
        
        # Resize both RGB and alpha to the calculated dimensions
        resized_rgb = cv2.resize(rgb, (resize_width, resize_height))
        resized_alpha = cv2.resize(alpha, (resize_width, resize_height))
        
        # Crop the resized image to fit the mask dimensions
        if resize_width >= target_width and resize_height >= target_height:
            cropped_rgb = resized_rgb[offset_y:offset_y + target_height, offset_x:offset_x + target_width]
            cropped_alpha = resized_alpha[offset_y:offset_y + target_height, offset_x:offset_x + target_width]
        else:
            # In case the resized image is somehow smaller than target (shouldn't happen)
            # Create padding around the product to fill the entire mask
            cropped_rgb = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            cropped_alpha = np.zeros((target_height, target_width), dtype=np.float32)
            
            paste_y = max(0, (target_height - resize_height) // 2)
            paste_x = max(0, (target_width - resize_width) // 2)
            
            paste_height = min(resize_height, target_height)
            paste_width = min(resize_width, target_width)
            
            cropped_rgb[paste_y:paste_y + paste_height, paste_x:paste_x + paste_width] = resized_rgb[:paste_height, :paste_width]
            cropped_alpha[paste_y:paste_y + paste_height, paste_x:paste_x + paste_width] = resized_alpha[:paste_height, :paste_width]
        
        # Create 3-channel alpha for blending
        cropped_alpha_3ch = np.stack([cropped_alpha, cropped_alpha, cropped_alpha], axis=2)
        
        # Get the region of interest in the output image
        roi = output[y_min:y_max+1, x_min:x_max+1]
        
        # Apply mask to restrict alpha blending only to the segmented region
        mask_region = binary_mask[y_min:y_max+1, x_min:x_max+1]
        mask_region_3ch = np.stack([mask_region, mask_region, mask_region], axis=2)
        
        # Combine mask with alpha for precise blending
        combined_alpha = cropped_alpha_3ch * mask_region_3ch
        
        # Blend using the combined alpha
        blended_roi = roi * (1 - combined_alpha) + cropped_rgb * combined_alpha
        
        # Place the blended region back into the output image
        output[y_min:y_max+1, x_min:x_max+1] = blended_roi
    else:
        # For non-transparent images, use a similar approach
        orig_height, orig_width = new_product.shape[:2]
        orig_aspect_ratio = orig_width / orig_height
        mask_aspect_ratio = target_width / target_height
        
        if orig_aspect_ratio > mask_aspect_ratio:
            resize_height = target_height
            resize_width = int(resize_height * orig_aspect_ratio)
            offset_x = (resize_width - target_width) // 2
            offset_y = 0
        else:
            resize_width = target_width
            resize_height = int(resize_width / orig_aspect_ratio)
            offset_x = 0
            offset_y = (resize_height - target_height) // 2
        
        # Resize the product image
        resized_product = cv2.resize(new_product, (resize_width, resize_height))
        
        # Crop to fit the mask dimensions
        if resize_width >= target_width and resize_height >= target_height:
            cropped_product = resized_product[offset_y:offset_y + target_height, offset_x:offset_x + target_width]
        else:
            # Handle edge case with padding
            cropped_product = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            paste_y = max(0, (target_height - resize_height) // 2)
            paste_x = max(0, (target_width - resize_width) // 2)
            paste_height = min(resize_height, target_height)
            paste_width = min(resize_width, target_width)
            cropped_product[paste_y:paste_y + paste_height, paste_x:paste_x + paste_width] = resized_product[:paste_height, :paste_width]
        
        # Create a mask for the target region
        roi_mask = binary_mask[y_min:y_max+1, x_min:x_max+1]
        roi_mask_3ch = np.stack([roi_mask, roi_mask, roi_mask], axis=2)
        
        # Get the region of interest in the output image
        roi = output[y_min:y_max+1, x_min:x_max+1]
        
        # Blend the new product with the region of interest using the mask
        blended_roi = roi * (1 - roi_mask_3ch) + cropped_product * roi_mask_3ch
        
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

# Product replacement section (only shown after mask generation)
if st.session_state.binary_mask is not None:
    st.header("Step 3: Replace Product")
    
    # Upload replacement product image
    uploaded_product = st.file_uploader("Upload New Product Image (preferably with transparent background)", type=["png", "jpg", "jpeg"], key="product_image")
    
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
                
                # Replace the product
                result_image = replace_product_in_image(
                    original_img,
                    new_product_np,
                    st.session_state.binary_mask
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

st.markdown("---")
st.markdown("Created with ❤️ using Streamlit and Meta's Segment Anything Model (SAM)")