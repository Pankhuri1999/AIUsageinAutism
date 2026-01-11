import cv2
import numpy as np
from quickdraw import QuickDrawData
from skimage.metrics import structural_similarity as ssim
import os
import h5py

# Try different import methods
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    try:
        from keras.models import load_model
        TF_AVAILABLE = False
    except ImportError:
        print("Error: Please install tensorflow or keras")
        raise

# --- Simple words for drawing game ---
SIMPLE_WORDS = [
    "cat", "dog", "tree", "house", "car", "sun", "moon", "star",
    "apple", "banana", "circle", "square", "triangle", "heart",
    "fish", "bird", "flower", "cloud", "mountain", "boat",
    "airplane", "bicycle", "cup", "spoon", "fork", "pencil",
    "book", "clock", "key", "umbrella", "rainbow", "butterfly"
]

# Path to QuickDraw model
MODEL_PATH = "QuickDraw.h5"  # Change this to your model path

def get_quickdraw_reference(category):
    """Get a reference drawing from QuickDraw dataset."""
    qd = QuickDrawData()
    if category not in qd.drawing_names:
        raise ValueError(f"Invalid category: {category}")
    return qd.get_drawing(category)

def render_drawing_to_image(drawing, size=(28, 28)):
    """Render QuickDraw drawing to image."""
    img = np.ones((255, 255), dtype=np.uint8) * 255
    for stroke in drawing.strokes:
        for i in range(len(stroke) - 1):
            x1, y1 = stroke[i]
            x2, y2 = stroke[i + 1]
            cv2.line(img, (x1, y1), (x2, y2), color=0, thickness=2)
    img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img_resized

def normalize_drawing(img):
    """Normalize drawing to remove style differences."""
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    coords = np.column_stack(np.where(binary == 0))
    if len(coords) == 0:
        return img
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    cropped = binary[y_min:y_max+1, x_min:x_max+1]
    
    h, w = cropped.shape
    max_dim = max(h, w)
    if max_dim == 0:
        return img
    
    pad = max_dim // 10
    square_size = max_dim + 2 * pad
    square = np.ones((square_size, square_size), dtype=np.uint8) * 255
    
    y_offset = (square_size - h) // 2
    x_offset = (square_size - w) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = cropped
    
    normalized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    return normalized

def preprocess_image_for_model(img):
    """Preprocess image for QuickDraw model prediction."""
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Apply threshold to get binary image
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Normalize the drawing
    normalized = normalize_drawing(thresh)
    
    # Resize to 28x28 (standard QuickDraw model input size)
    processed = cv2.resize(normalized, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Convert to float32 and normalize to 0-1 range
    processed = processed.astype(np.float32) / 255.0
    
    # Reshape for model input: (1, 28, 28, 1)
    processed = np.reshape(processed, (1, 28, 28, 1))
    
    return processed, normalized

def keras_predict(model, image):
    """Predict using the QuickDraw model."""
    pred_probab = model.predict(image, verbose=0)[0]
    pred_class_idx = np.argmax(pred_probab)
    confidence = pred_probab[pred_class_idx]
    return confidence, pred_class_idx, pred_probab

def load_quickdraw_model(model_path):
    """Load the QuickDraw model with multiple fallback methods."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"📦 Loading model from {model_path}...")
    
    # Method 1: Try loading with compile=False
    try:
        print("   Trying method 1: load_model with compile=False...")
        model = load_model(model_path, compile=False)
        print("✅ Model loaded successfully (method 1)")
        return model
    except Exception as e1:
        print(f"   Method 1 failed: {str(e1)[:100]}")
    
    # Method 2: Try loading with safe_mode=False (for older models)
    try:
        print("   Trying method 2: load_model with safe_mode=False...")
        if TF_AVAILABLE:
            model = load_model(model_path, compile=False, safe_mode=False)
        else:
            model = load_model(model_path, compile=False)
        print("✅ Model loaded successfully (method 2)")
        return model
    except Exception as e2:
        print(f"   Method 2 failed: {str(e2)[:100]}")
    
    # Method 3: Try using tf.keras.models.load_model directly
    if TF_AVAILABLE:
        try:
            print("   Trying method 3: tf.keras.models.load_model...")
            model = tf.keras.models.load_model(model_path, compile=False)
            print("✅ Model loaded successfully (method 3)")
            return model
        except Exception as e3:
            print(f"   Method 3 failed: {str(e3)[:100]}")
    
    # Method 4: Try loading weights only (if model architecture is known)
    try:
        print("   Trying method 4: Loading with custom_objects...")
        # This might work if we bypass some compatibility checks
        import json
        with h5py.File(model_path, 'r') as f:
            model_config = f.attrs.get('model_config')
            if model_config:
                if isinstance(model_config, bytes):
                    model_config = model_config.decode('utf-8')
                model_config = json.loads(model_config)
        
        # Try to reconstruct model
        if TF_AVAILABLE:
            model = tf.keras.models.model_from_json(model_config)
            model.load_weights(model_path)
            print("✅ Model loaded successfully (method 4 - weights only)")
            return model
    except Exception as e4:
        print(f"   Method 4 failed: {str(e4)[:100]}")
    
    # If all methods fail, provide helpful error message
    raise RuntimeError(
        f"❌ Could not load model with any method. "
        f"The model file may be incompatible with your Keras/TensorFlow version. "
        f"Try:\n"
        f"  1. Downgrade: pip install tensorflow==2.10.0\n"
        f"  2. Or upgrade: pip install --upgrade tensorflow\n"
        f"  3. Or use a model saved with your current version"
    )

def get_category_from_index(index, category_list):
    """Get category name from model prediction index."""
    # This mapping depends on how your model was trained
    # You may need to adjust this based on your model's class order
    if index < len(category_list):
        return category_list[index]
    return "unknown"

def create_comparison_display(user_img, ref_img, original_img, category, predicted_category, 
                              confidence, model_confidence, score_details):
    """Create a unified display showing comparison results."""
    display_h, display_w = 800, 1200
    unified = np.ones((display_h, display_w, 3), dtype=np.uint8) * 240
    
    # Resize components
    img_w, img_h = 300, 300
    orig_w, orig_h = 300, 300
    
    # Original image
    orig_resized = cv2.resize(original_img, (orig_w, orig_h))
    unified[50:50+orig_h, 50:50+orig_w] = orig_resized
    cv2.putText(unified, "Your Uploaded Image", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # User normalized image
    user_colored = cv2.cvtColor(user_img, cv2.COLOR_GRAY2BGR)
    user_resized = cv2.resize(user_colored, (img_w, img_h))
    unified[50:50+img_h, 400:400+img_w] = user_resized
    cv2.putText(unified, "Your Drawing (Normalized)", (400, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Reference image
    ref_colored = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
    ref_resized = cv2.resize(ref_colored, (img_w, img_h))
    unified[50:50+img_h, 750:750+img_w] = ref_resized
    cv2.putText(unified, f"QuickDraw: {category.upper()}", (750, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Results panel
    results_x = 50
    results_y = 400
    results_w = 1000
    results_h = 350
    
    cv2.rectangle(unified, (results_x, results_y), (results_x + results_w, results_y + results_h), 
                 (200, 200, 200), -1)
    cv2.rectangle(unified, (results_x, results_y), (results_x + results_w, results_y + results_h), 
                 (0, 0, 0), 2)
    
    y_pos = results_y + 40
    line_height = 30
    
    # Title
    cv2.putText(unified, "PREDICTION RESULTS", (results_x + 20, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    y_pos += line_height + 10
    
    # Model prediction
    cv2.putText(unified, f"Model Predicted: {predicted_category.upper()}", 
               (results_x + 20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    y_pos += line_height
    
    cv2.putText(unified, f"Model Confidence: {model_confidence:.2%}", 
               (results_x + 20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    y_pos += line_height
    
    # Expected category
    cv2.putText(unified, f"Expected Category: {category.upper()}", 
               (results_x + 20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    y_pos += line_height
    
    # Match result
    if predicted_category.lower() == category.lower():
        match_text = "✅ MATCH!"
        match_color = (0, 255, 0)
    else:
        match_text = "❌ NO MATCH"
        match_color = (0, 0, 255)
    
    cv2.putText(unified, match_text, (results_x + 20, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, match_color, 2)
    y_pos += line_height + 10
    
    # Similarity scores
    cv2.putText(unified, "Similarity Scores:", (results_x + 20, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    y_pos += line_height
    
    cv2.putText(unified, f"  SSIM: {score_details['ssim']:.2%}", 
               (results_x + 20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    y_pos += line_height - 5
    
    cv2.putText(unified, f"  Contour: {score_details['contour']:.2%}", 
               (results_x + 20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    y_pos += line_height - 5
    
    cv2.putText(unified, f"  Histogram: {score_details['histogram']:.2%}", 
               (results_x + 20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    y_pos += line_height - 5
    
    cv2.putText(unified, f"  Combined: {confidence:.2%}", 
               (results_x + 20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Title
    cv2.putText(unified, "QuickDraw Model Comparison", (50, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return unified

def compare_drawings_ssim(user_img, ref_img):
    """Compare using Structural Similarity Index."""
    user = user_img.astype(np.float32) / 255.0
    ref = ref_img.astype(np.float32) / 255.0
    score, _ = ssim(user, ref, full=True, data_range=1.0)
    return score

def compare_drawings_contour(user_img, ref_img):
    """Compare using contour matching."""
    user_contours, _ = cv2.findContours(user_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ref_contours, _ = cv2.findContours(ref_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(user_contours) == 0 or len(ref_contours) == 0:
        return 0.0
    
    user_largest = max(user_contours, key=cv2.contourArea)
    ref_largest = max(ref_contours, key=cv2.contourArea)
    match_score = cv2.matchShapes(user_largest, ref_largest, cv2.CONTOURS_MATCH_I2, 0)
    similarity = 1.0 / (1.0 + match_score)
    return similarity

def compare_drawings_histogram(user_img, ref_img):
    """Compare using histogram correlation."""
    user_hist = cv2.calcHist([user_img], [0], None, [256], [0, 256])
    ref_hist = cv2.calcHist([ref_img], [0], None, [256], [0, 256])
    correlation = cv2.compareHist(user_hist, ref_hist, cv2.HISTCMP_CORREL)
    return max(0.0, correlation)

def compare_drawings_combined(user_img, ref_img):
    """Combine multiple comparison methods for better accuracy."""
    ssim_score = compare_drawings_ssim(user_img, ref_img)
    contour_score = compare_drawings_contour(user_img, ref_img)
    hist_score = compare_drawings_histogram(user_img, ref_img)
    combined = (0.4 * ssim_score + 0.4 * contour_score + 0.2 * hist_score)
    
    return combined, {
        'ssim': ssim_score,
        'contour': contour_score,
        'histogram': hist_score
    }

# ============================================
# MAIN EXECUTION
# ============================================
print("=" * 60)
print("🎨 QuickDraw Model Comparison")
print("=" * 60)

# Step 1: Load the model
try:
    model = load_quickdraw_model(MODEL_PATH)
except Exception as e:
    print(f"\n❌ Error loading model: {e}")
    print(f"\n💡 Troubleshooting tips:")
    print(f"   1. Check if MODEL_PATH is correct: {MODEL_PATH}")
    print(f"   2. Try installing compatible TensorFlow version:")
    print(f"      pip install tensorflow==2.10.0")
    print(f"   3. Or try upgrading:")
    print(f"      pip install --upgrade tensorflow")
    print(f"   4. The model file may need to be re-saved with your current version")
    raise

# Step 2: Get category
print("\nAvailable categories:")
for i, word in enumerate(SIMPLE_WORDS, 1):
    print(f"  {i:2d}. {word}")

category_input = input("\nEnter category name or number: ").strip().lower()

try:
    category_num = int(category_input)
    if 1 <= category_num <= len(SIMPLE_WORDS):
        category = SIMPLE_WORDS[category_num - 1]
    else:
        category = SIMPLE_WORDS[0]
except ValueError:
    if category_input in SIMPLE_WORDS:
        category = category_input
    else:
        category = SIMPLE_WORDS[0]

print(f"\n🎨 Selected Category: {category.upper()}")
print("=" * 60)

# Step 3: Get image path
image_path = input("\nEnter the path to your image file: ").strip().strip('"')

if not os.path.exists(image_path):
    print(f"❌ Image file not found: {image_path}")
    exit()

# Step 4: Load and preprocess image
try:
    print(f"\n📂 Loading image: {image_path}")
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"Could not read image from: {image_path}")
    
    print("📊 Preprocessing image for model...")
    processed_img, normalized_img = preprocess_image_for_model(original_img)
    print("✅ Image processed successfully")
except Exception as e:
    print(f"❌ Error processing image: {e}")
    raise

# Step 5: Get reference drawing
try:
    print(f"🔍 Fetching QuickDraw reference for '{category}'...")
    drawing = get_quickdraw_reference(category)
    reference_img = render_drawing_to_image(drawing)
    reference_img = normalize_drawing(reference_img)
    print("✅ Reference drawing loaded successfully")
except Exception as e:
    print(f"❌ Error loading QuickDraw reference: {e}")
    raise

# Step 6: Model prediction
print("\n🤖 Running model prediction...")
model_confidence, pred_class_idx, all_predictions = keras_predict(model, processed_img)
predicted_category = get_category_from_index(pred_class_idx, SIMPLE_WORDS)

print(f"   Model predicted: {predicted_category.upper()}")
print(f"   Confidence: {model_confidence:.2%}")

# Step 7: Traditional comparison methods
print("\n📊 Running similarity analysis...")
overall_score, score_details = compare_drawings_combined(normalized_img, reference_img)

# Step 8: Display results
print("\n" + "=" * 60)
print("📈 PREDICTION RESULTS")
print("=" * 60)

print(f"\n🤖 Model Prediction:")
print(f"   • Predicted Category: {predicted_category.upper()}")
print(f"   • Confidence: {model_confidence:.2%}")
print(f"   • Expected Category: {category.upper()}")

if predicted_category.lower() == category.lower():
    print(f"   ✅ MATCH! Model correctly identified the drawing!")
else:
    print(f"   ❌ NO MATCH - Model predicted '{predicted_category}' but expected '{category}'")

print(f"\n📊 Similarity Scores:")
print(f"   • SSIM Score: {score_details['ssim']:.2%}")
print(f"   • Contour Match Score: {score_details['contour']:.2%}")
print(f"   • Histogram Correlation: {score_details['histogram']:.2%}")
print(f"   • Combined Similarity: {overall_score:.2%}")

# Show top 3 predictions
print(f"\n🔝 Top 3 Model Predictions:")
sorted_indices = np.argsort(all_predictions)[::-1][:3]
for i, idx in enumerate(sorted_indices, 1):
    cat_name = get_category_from_index(idx, SIMPLE_WORDS)
    conf = all_predictions[idx]
    print(f"   {i}. {cat_name.upper()}: {conf:.2%}")

# Step 9: Create visual comparison
print("\n🖼️  Displaying visual comparison...")
comparison_display = create_comparison_display(normalized_img, reference_img, original_img, 
                                              category, predicted_category, overall_score,
                                              model_confidence, score_details)

cv2.imshow("QuickDraw Model Comparison - Press any key to close", comparison_display)

print("\n✅ Analysis complete! Press any key in the image window to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("📋 SUMMARY")
print("=" * 60)
print(f"Model Prediction: {predicted_category.upper()} ({model_confidence:.2%})")
print(f"Expected: {category.upper()}")
print(f"Match: {'✅ YES' if predicted_category.lower() == category.lower() else '❌ NO'}")
print(f"Similarity Score: {overall_score:.2%}")
print("=" * 60)
