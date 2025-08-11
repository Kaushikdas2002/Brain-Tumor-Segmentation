from flask import Flask, request, jsonify, send_file, render_template, after_this_request
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg') # Use Agg backend for Matplotlib in a non-GUI environment
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import tempfile
import shutil
import os
from io import BytesIO
import imageio.v2 as imageio
from tempfile import NamedTemporaryFile
import zipfile # For creating zip files
import base64 # For encoding images to send to frontend

# Initialize Flask app
app = Flask(__name__)

# --- Global Variables (Consider better state management for production) ---
MODEL = None
PREPROCESSED_VOLUME = None
SEGMENTED_MASK = None
REGION_PERCENTAGES = None
LAST_AFFINE = np.eye(4) # Store affine for saving NIfTI. Default to identity.
MAX_SLICES_PER_VIEW = {"axial": 0, "coronal": 0, "sagittal": 0}

# --- Model Loading ---
def load_model_on_startup():
    """Loads the TensorFlow/Keras model when the Flask application starts."""
    global MODEL
    try:
        # IMPORTANT: Replace "BraTS_Trained_ResUNet_seg_Model2.h5"
        # with the actual path to your model file in the Flask environment.
        # Ensure this file is in the same directory as app.py or provide the full path.
        model_path = "BraTS_Trained_ResUNet_seg_Model.h5"
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}. Please ensure it's in the correct location.")
            MODEL = None
            return

        MODEL = tf.keras.models.load_model(model_path, compile=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        MODEL = None # Ensure model is None if loading fails

# --- Preprocessing and Segmentation Logic (Adapted from Streamlit code) ---
IMG_SIZE = 128 # Target size for cropped dimensions

def normalize_volume(volume):
    """Normalizes a single MRI volume using StandardScaler."""
    scaler = StandardScaler()
    # Handle potential all-zero slices or very small std dev
    if volume.std() < 1e-6: # Avoid division by zero or issues with near-zero std dev
        return np.zeros_like(volume, dtype=np.float32)
    flat = volume.astype(np.float32).flatten().reshape(-1, 1) # Ensure float32 for scaler
    scaled = scaler.fit_transform(flat)
    return scaled.reshape(volume.shape).astype(np.float32)


def preprocess_modalities(flair, t2, t1ce, t1):
    """
    Normalizes, stacks, and crops the MRI modalities.
    Input arrays are expected to be raw numpy arrays from nib.get_fdata().
    """
    global LAST_AFFINE # We'll use a default affine if not set by a NIfTI load
                      # In a real app, ensure affine is correctly handled for each input.

    flair_norm = normalize_volume(flair)
    t2_norm = normalize_volume(t2)
    t1ce_norm = normalize_volume(t1ce)
    t1_norm = normalize_volume(t1)

    stacked = np.stack([flair_norm, t2_norm, t1ce_norm, t1_norm], axis=3)
    
    # Original cropping dimensions from Streamlit:
    # flair.shape -> (240, 240, 155)
    # Cropped to 128x128x128 using [56:184, 56:184, 13:141]
    
    # Define crop boundaries
    x_start, x_end = 56, 56 + IMG_SIZE
    y_start, y_end = 56, 56 + IMG_SIZE
    z_start, z_end = 13, 13 + IMG_SIZE

    # Check if input dimensions are sufficient for the crop
    if stacked.shape[0] < x_end or \
       stacked.shape[1] < y_end or \
       stacked.shape[2] < z_end:
        print(f"Warning: Input stack shape {stacked.shape} is too small for the standard crop.")
        # Fallback: Attempt a center crop to IMG_SIZE if possible
        cx, cy, cz = stacked.shape[0]//2, stacked.shape[1]//2, stacked.shape[2]//2
        h_img_size = IMG_SIZE // 2
        
        # Ensure crop indices are within bounds
        crop_x_start = max(0, cx - h_img_size)
        crop_x_end = min(stacked.shape[0], cx + h_img_size + (IMG_SIZE % 2)) # Add 1 if IMG_SIZE is odd
        crop_y_start = max(0, cy - h_img_size)
        crop_y_end = min(stacked.shape[1], cy + h_img_size + (IMG_SIZE % 2))
        crop_z_start = max(0, cz - h_img_size)
        crop_z_end = min(stacked.shape[2], cz + h_img_size + (IMG_SIZE % 2))

        cropped = stacked[crop_x_start:crop_x_end,
                          crop_y_start:crop_y_end,
                          crop_z_start:crop_z_end, :]
        
        # If cropped shape is still not IMG_SIZE x IMG_SIZE x IMG_SIZE, pad it
        pad_x = IMG_SIZE - cropped.shape[0]
        pad_y = IMG_SIZE - cropped.shape[1]
        pad_z = IMG_SIZE - cropped.shape[2]

        if pad_x > 0 or pad_y > 0 or pad_z > 0:
            print(f"Padding cropped image from {cropped.shape[:3]} to {(IMG_SIZE, IMG_SIZE, IMG_SIZE)}")
            # Calculate padding amounts (before and after for centering)
            pad_x_before = pad_x // 2
            pad_x_after = pad_x - pad_x_before
            pad_y_before = pad_y // 2
            pad_y_after = pad_y - pad_y_before
            pad_z_before = pad_z // 2
            pad_z_after = pad_z - pad_z_before

            cropped = np.pad(cropped, 
                             ((pad_x_before, pad_x_after), 
                              (pad_y_before, pad_y_after), 
                              (pad_z_before, pad_z_after), 
                              (0,0)), # No padding for the modality channel
                             mode='constant', constant_values=0)

        if cropped.shape[:3] != (IMG_SIZE, IMG_SIZE, IMG_SIZE):
             raise ValueError(f"Processed (cropped/padded) shape {cropped.shape[:3]} is not {(IMG_SIZE, IMG_SIZE, IMG_SIZE)}. Original input shape was {stacked.shape[:3]}")
    else:
        # Standard crop
        cropped = stacked[x_start:x_end, y_start:y_end, z_start:z_end, :]
    
    return cropped


def segment_image_stack(image_stack):
    """Performs segmentation using the loaded Keras model."""
    if MODEL is None:
        raise ValueError("Model not loaded. Cannot perform segmentation.")
    # Model expects batch_size, H, W, D, Channels
    input_data = np.expand_dims(image_stack, axis=0) 
    prediction = MODEL.predict(input_data)[0]  # Remove batch dimension
    # Prediction shape: (128, 128, 128, 4) -> 4 classes
    prediction_labels = np.argmax(prediction, axis=-1)  # Shape: (128, 128, 128)
    return prediction_labels.astype(np.uint8) # Ensure uint8 for mask

def overlay_segmentation_on_grayscale(base_slice, mask_slice):
    """Overlays segmentation mask colors onto a grayscale base slice."""
    base_slice = base_slice.astype(np.float32) # Ensure float for normalization
    min_val, max_val = np.min(base_slice), np.max(base_slice)
    norm_diff = max_val - min_val

    if norm_diff < 1e-6: # Handle case where slice is uniform (or near uniform)
        base_rgb = np.zeros((*base_slice.shape, 3), dtype=np.uint8)
    else:
        base_normalized = (base_slice - min_val) / norm_diff
        # Use plt.cm.gray to get RGB, then take only RGB channels (ignore alpha)
        base_rgb = (plt.cm.gray(base_normalized)[:, :, :3] * 255).astype(np.uint8)

    # Apply colors for mask regions. Ensure base_rgb is writable.
    overlay = np.copy(base_rgb)
    overlay[mask_slice == 1] = [0, 0, 255]    # Necrotic Core - Blue
    overlay[mask_slice == 2] = [0, 255, 0]    # Edema - Green
    overlay[mask_slice == 3] = [255, 0, 0]    # Enhancing Tumor - Red
    return overlay


def calculate_tumor_percentages(mask_3d):
    """Calculates the percentage of each tumor region in the 3D mask."""
    total_voxels = np.prod(mask_3d.shape)
    if total_voxels == 0: return {} # Avoid division by zero for empty mask

    region_percentages = {}
    labels_info = {
        1: "Necrotic Core",
        2: "Edema",
        3: "Enhancing Tumor"
    }
    for label_val, name in labels_info.items():
        count = np.sum(mask_3d == label_val)
        region_percentages[name] = (count / total_voxels) * 100
    return region_percentages

# --- Utility for image to base64 ---
def BytesIO_to_base64(b_io):
    """Converts BytesIO image data to a base64 string."""
    return base64.b64encode(b_io.getvalue()).decode('utf-8')

# --- Flask Routes ---
@app.route('/')
def landing_page():
    return render_template('landing.html')

@app.route('/upload')
def upload_segment():
    return render_template('index.html')
# @app.route('/')
# def index():
#     """Serves the main HTML page."""
#     return render_template('index.html')

@app.route('/upload_and_segment', methods=['POST'])
def upload_and_segment():
    """Handles file uploads, performs segmentation, and returns results."""
    global PREPROCESSED_VOLUME, SEGMENTED_MASK, REGION_PERCENTAGES, LAST_AFFINE, MAX_SLICES_PER_VIEW
    if MODEL is None:
        return jsonify({"error": "Model not loaded. Please check server logs."}), 500

    files = request.files
    required_files = ['flair', 't2', 't1ce', 't1']
    if not all(k in files for k in required_files):
        missing = [k for k in required_files if k not in files]
        return jsonify({"error": f"Missing one or more NIfTI files: {', '.join(missing)}"}), 400

    temp_files_paths = {}
    try:
        # Create a temporary directory to store uploaded files
        with tempfile.TemporaryDirectory() as tmpdir:
            for key in required_files:
                file = files[key]
                # Sanitize filename or use a default
                filename = file.filename if file.filename else f"{key}_upload.nii.gz"
                # Basic sanitization (replace risky characters)
                filename = "".join(c if c.isalnum() or c in ['.', '_', '-'] else '_' for c in filename)
                path = os.path.join(tmpdir, filename)
                file.save(path)
                temp_files_paths[key] = path
            
            # Load NIfTI data using nibabel
            # Store affine from one of the files (e.g., flair) - assumes all have same affine/orientation
            flair_nii = nib.load(temp_files_paths['flair'])
            flair_data = flair_nii.get_fdata(dtype=np.float32) # Load as float32
            LAST_AFFINE = flair_nii.affine # Store the affine for potential saving later

            t2_data = nib.load(temp_files_paths['t2']).get_fdata(dtype=np.float32)
            t1ce_data = nib.load(temp_files_paths['t1ce']).get_fdata(dtype=np.float32)
            t1_data = nib.load(temp_files_paths['t1']).get_fdata(dtype=np.float32)

        # Preprocess and Segment
        PREPROCESSED_VOLUME = preprocess_modalities(flair_data, t2_data, t1ce_data, t1_data)
        SEGMENTED_MASK = segment_image_stack(PREPROCESSED_VOLUME)
        REGION_PERCENTAGES = calculate_tumor_percentages(SEGMENTED_MASK)

        # Update max slices for each view based on the processed volume
        if PREPROCESSED_VOLUME is not None:
            MAX_SLICES_PER_VIEW["axial"] = PREPROCESSED_VOLUME.shape[2] # Depth
            MAX_SLICES_PER_VIEW["coronal"] = PREPROCESSED_VOLUME.shape[1] # Height
            MAX_SLICES_PER_VIEW["sagittal"] = PREPROCESSED_VOLUME.shape[0] # Width
        else: # Should not happen if preprocessing is successful
            MAX_SLICES_PER_VIEW = {"axial": 0, "coronal": 0, "sagittal": 0}


        return jsonify({
            "message": "Segmentation successful!",
            "max_slices_per_view": MAX_SLICES_PER_VIEW,
            "region_percentages": REGION_PERCENTAGES
        })

    except ValueError as ve: # Catch specific errors like model not loaded or bad crop
        print(f"ValueError during processing: {ve}")
        return jsonify({"error": f"Processing error: {str(ve)}"}), 400
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc() # Print full traceback to server console for debugging
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/get_slice', methods=['GET'])
def get_slice_image():
    """Generates and returns original and segmented slice images as base64 data URLs."""
    if PREPROCESSED_VOLUME is None or SEGMENTED_MASK is None:
        return jsonify({"error": "No data processed yet. Please upload files first."}), 400

    try:
        slice_num_req = int(request.args.get('slice_num', '1')) # 1-indexed from frontend
        modality_idx = int(request.args.get('modality_idx', '0')) # 0:FLAIR, 1:T2, 2:T1CE, 3:T1
        view = request.args.get('view', 'Axial').lower() # axial, coronal, sagittal

        # Validate modality index
        if not 0 <= modality_idx <= 3:
            return jsonify({"error": "Invalid modality index."}), 400

        # Determine max slices for the current view and adjust slice_num (0-indexed internally)
        current_max_slices_for_view = MAX_SLICES_PER_VIEW.get(view, 0)
        if current_max_slices_for_view == 0:
            return jsonify({"error": f"No slices available for {view} view or view is invalid."}), 400
        
        slice_idx = max(0, min(slice_num_req - 1, current_max_slices_for_view - 1))


        original_slice_data, mask_slice_data = None, None
        if view == "axial":
            original_slice_data = PREPROCESSED_VOLUME[:, :, slice_idx, modality_idx]
            mask_slice_data = SEGMENTED_MASK[:, :, slice_idx]
        elif view == "coronal":
            original_slice_data = PREPROCESSED_VOLUME[:, slice_idx, :, modality_idx]
            mask_slice_data = SEGMENTED_MASK[:, slice_idx, :]
        elif view == "sagittal":
            original_slice_data = PREPROCESSED_VOLUME[slice_idx, :, :, modality_idx]
            mask_slice_data = SEGMENTED_MASK[slice_idx, :, :]
        else:
            return jsonify({"error": "Invalid view type specified."}), 400

        # Rotate slices for standard medical image orientation (optional, but common)
        # np.rot90(slice, k=1) rotates 90 degrees counter-clockwise
        original_slice_display = np.rot90(original_slice_data)
        mask_slice_display = np.rot90(mask_slice_data)
        
        # Generate original image (PNG)
        fig_orig, ax_orig = plt.subplots(figsize=(4,4), dpi=100) # Adjust size/DPI as needed
        ax_orig.imshow(original_slice_display, cmap='gray', aspect='auto')
        ax_orig.axis('off')
        img_io_orig = BytesIO()
        fig_orig.savefig(img_io_orig, format='png', bbox_inches='tight', pad_inches=0)
        img_io_orig.seek(0)
        plt.close(fig_orig) # Close figure to free memory

        # Generate segmented image (PNG)
        segmented_overlay_display = overlay_segmentation_on_grayscale(original_slice_display, mask_slice_display)
        fig_seg, ax_seg = plt.subplots(figsize=(5,5), dpi=100)
        ax_seg.imshow(segmented_overlay_display, aspect='auto')
        ax_seg.axis('off')
        img_io_seg = BytesIO()
        fig_seg.savefig(img_io_seg, format='png', bbox_inches='tight', pad_inches=0)
        img_io_seg.seek(0)
        plt.close(fig_seg)

        return jsonify({
            "original_image_url": f"data:image/png;base64,{BytesIO_to_base64(img_io_orig)}",
            "segmented_image_url": f"data:image/png;base64,{BytesIO_to_base64(img_io_seg)}"
        })

    except Exception as e:
        print(f"Error getting slice: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error generating slice image: {str(e)}"}), 500


@app.route('/get_video', methods=['GET'])
def get_segmentation_video():
    """Generates and serves a video of the segmented slices."""
    if PREPROCESSED_VOLUME is None or SEGMENTED_MASK is None:
        return jsonify({"error": "No data processed yet. Please upload files first."}), 400

    try:
        modality_idx = int(request.args.get('modality_idx', '0'))
        view = request.args.get('view', 'Axial').lower()

        if not 0 <= modality_idx <= 3:
            return jsonify({"error": "Invalid modality index for video."}), 400

        selected_volume_for_video = PREPROCESSED_VOLUME[:, :, :, modality_idx]
        seg_mask_3d_for_video = SEGMENTED_MASK
        frames = []

        num_slices_in_view = MAX_SLICES_PER_VIEW.get(view, 0)
        if num_slices_in_view == 0:
            return jsonify({"error": f"No slices available for {view} view video or view is invalid."}), 400

        for i in range(num_slices_in_view):
            fig, ax = plt.subplots(figsize=(3, 3), dpi=150)  # Smaller DPI for video frames
            base_slice, mask_slice = None, None

            if view == "axial":
                base_slice = selected_volume_for_video[:, :, i]
                mask_slice = seg_mask_3d_for_video[:, :, i]
            elif view == "coronal":
                base_slice = selected_volume_for_video[:, i, :]
                mask_slice = seg_mask_3d_for_video[:, i, :]
            elif view == "sagittal":
                base_slice = selected_volume_for_video[i, :, :]
                mask_slice = seg_mask_3d_for_video[i, :, :]

            # Rotate for display
            base_slice_display = np.rot90(base_slice)
            mask_slice_display = np.rot90(mask_slice)

            rgb_overlay = overlay_segmentation_on_grayscale(base_slice_display, mask_slice_display)
            ax.imshow(rgb_overlay)
            ax.text(5, 10, f"Slice {i + 1}", fontsize=8, color='yellow',
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))
            ax.axis("off")

            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            frames.append(imageio.imread(buf))  # Read PNG from buffer
            plt.close(fig)  # Crucial to close figures

        # Create video in a temporary file
        with NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
            video_path_local = temp_video_file.name
            imageio.mimsave(video_path_local, frames, fps=4, macro_block_size=1, quality=10, codec='libx264')

        @after_this_request
        def remove_file(response):
            try:
                os.remove(video_path_local)
            except Exception as e_clean:
                print(f"Error cleaning up temporary video file {video_path_local}: {e_clean}")
            return response

        return send_file(
            video_path_local,
            as_attachment=True,
            download_name=f"segmentation_{view}_{modality_idx}.mp4",
            mimetype='video/mp4'
        )

    except Exception as e:
        print(f"Error generating video: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error generating video: {str(e)}"}), 500



@app.route('/download_mask', methods=['GET'])
def download_segmentation_mask():
    """Serves the full 3D segmented mask as a NIfTI file inside a ZIP archive."""
    if SEGMENTED_MASK is None:
        return jsonify({"error": "No segmentation mask available to download."}), 400

    try:
        # Ensure affine is valid
        affine_to_use = LAST_AFFINE if isinstance(LAST_AFFINE, np.ndarray) else np.eye(4)

        # Convert to uint8 and create NIfTI image
        mask_to_save = SEGMENTED_MASK.astype(np.uint8)
        seg_img = nib.Nifti1Image(mask_to_save, affine=affine_to_use)

        # Create temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            nifti_filename = "segmented_mask.nii"
            nifti_path = os.path.join(tmpdir, nifti_filename)
            nib.save(seg_img, nifti_path)

            zip_path = os.path.join(tmpdir, "segmentation_mask.zip")
            shutil.make_archive(zip_path.replace(".zip", ""), 'zip', tmpdir, nifti_filename)
            # Read ZIP into memory
            with open(zip_path, 'rb') as f:
                zip_bytes = BytesIO(f.read())
            zip_bytes.seek(0)

            return send_file(
                zip_bytes,
                as_attachment=True,
                download_name="segmentation_mask_nifti.zip",
                mimetype='application/zip'
            )

    except Exception as e:
        return jsonify({"error": f"Error preparing mask for download: {str(e)}"}), 500

# --- Main Execution ---
if __name__ == '__main__':
    load_model_on_startup() # Load the Keras model when Flask starts
    # Make sure to set host='0.0.0.0' to make it accessible on your network
    # Debug should be False in production
    app.run(debug=True, host='0.0.0.0', port=5000)
