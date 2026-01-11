import cv2
import numpy as np
from quickdraw import QuickDrawData
from skimage.metrics import structural_similarity as ssim
import time
import random
import os

# ============================================
# SET YOUR IMAGE PATH HERE
# ============================================
IMAGE_PATH = "path/to/your/image.jpg"  # Change this to your image path
# Example: IMAGE_PATH = "C:/Users/YourName/Pictures/drawing.jpg"
# Example: IMAGE_PATH = "./my_drawing.png"
# Example: IMAGE_PATH = "drawing.jpg"  # If image is in same folder

# --- Simple words for drawing game ---
SIMPLE_WORDS = [
    "cat", "dog", "tree", "house", "car", "sun", "moon", "star",
    "apple", "banana", "circle", "square", "triangle", "heart",
    "fish", "bird", "flower", "cloud", "mountain", "boat",
    "airplane", "bicycle", "cup", "spoon", "fork", "pencil",
    "book", "clock", "key", "umbrella", "rainbow", "butterfly"
]

def get_random_word():
    """Get a random simple word from the list."""
    return random.choice(SIMPLE_WORDS)

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

def preprocess_uploaded_image(image_path):
    """Preprocess uploaded image file."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to get binary image
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Normalize the drawing
    normalized = normalize_drawing(thresh)
    
    return normalized, img

def get_contour_statistics(img):
    """Get detailed contour statistics from an image."""
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 5
    filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_contour_area]
    
    external_contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_external = [c for c in external_contours if cv2.contourArea(c) >= min_contour_area]
    
    total_points = sum(len(c) for c in filtered_contours)
    external_points = sum(len(c) for c in filtered_external)
    total_area = sum(cv2.contourArea(c) for c in filtered_contours)
    
    return {
        'all_contours': filtered_contours,
        'external_contours': filtered_external,
        'total_contours': len(filtered_contours),
        'external_contour_count': len(filtered_external),
        'total_points': total_points,
        'external_points': external_points,
        'total_area': total_area,
        'hierarchy': hierarchy
    }

def compare_contours_detailed(user_img, ref_img):
    """Compare contours between two images with detailed statistics."""
    user_stats = get_contour_statistics(user_img)
    ref_stats = get_contour_statistics(ref_img)
    
    user_contours = user_stats['external_contours']
    ref_contours = ref_stats['external_contours']
    
    if len(user_contours) == 0 or len(ref_contours) == 0:
        return {
            'similarity': 0.0,
            'matched_contours': 0,
            'match_percentage': 0.0,
            'user_total': 0,
            'ref_total': 0,
            'user_stats': user_stats,
            'ref_stats': ref_stats
        }
    
    match_scores = []
    matched_pairs = []
    
    for i, user_contour in enumerate(user_contours):
        best_match_score = float('inf')
        best_match_idx = -1
        
        for j, ref_contour in enumerate(ref_contours):
            match_score = cv2.matchShapes(user_contour, ref_contour, cv2.CONTOURS_MATCH_I2, 0)
            if match_score < best_match_score:
                best_match_score = match_score
                best_match_idx = j
        
        if best_match_idx != -1:
            match_scores.append(best_match_score)
            matched_pairs.append((i, best_match_idx, best_match_score))
    
    if len(match_scores) > 0:
        similarities = [1.0 / (1.0 + score) for score in match_scores]
        avg_similarity = np.mean(similarities)
    else:
        avg_similarity = 0.0
    
    good_matches = sum(1 for score in match_scores if 1.0 / (1.0 + score) > 0.5)
    total_possible_matches = min(len(user_contours), len(ref_contours))
    match_percentage = (good_matches / total_possible_matches * 100) if total_possible_matches > 0 else 0.0
    
    return {
        'similarity': avg_similarity,
        'matched_contours': good_matches,
        'total_compared': len(matched_pairs),
        'match_percentage': match_percentage,
        'user_total': len(user_contours),
        'ref_total': len(ref_contours),
        'user_stats': user_stats,
        'ref_stats': ref_stats,
        'match_scores': match_scores,
        'matched_pairs': matched_pairs
    }

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

def visualize_contours(img, stats, title="Contours"):
    """Visualize contours on image."""
    vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis_img, stats['external_contours'], -1, (0, 255, 0), 1)
    cv2.drawContours(vis_img, stats['all_contours'], -1, (255, 0, 0), 1)
    return vis_img

def create_comparison_display(user_img, ref_img, original_img, category, user_stats, ref_stats, 
                              contour_comparison, overall_score, score_details):
    """Create a unified display showing comparison results."""
    display_h, display_w = 800, 1200
    unified = np.ones((display_h, display_w, 3), dtype=np.uint8) * 240
    
    # Resize components
    img_w, img_h = 300, 300
    orig_w, orig_h = 300, 300
    
    # Original image
    orig_resized = cv2.resize(original_img, (orig_w, orig_h))
    unified[50:50+orig_h, 50:50+orig_w] = orig_resized
    cv2.putText(unified, "Your Image", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
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
    
    # Contour visualizations
    user_vis = visualize_contours(user_img, user_stats)
    ref_vis = visualize_contours(ref_img, ref_stats)
    
    user_vis_resized = cv2.resize(user_vis, (img_w, img_h))
    ref_vis_resized = cv2.resize(ref_vis, (img_w, img_h))
    
    unified[400:400+img_h, 50:50+img_w] = user_vis_resized
    cv2.putText(unified, "Your Contours", (50, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    unified[400:400+img_h, 400:400+img_w] = ref_vis_resized
    cv2.putText(unified, "Reference Contours", (400, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Results panel
    results_x = 750
    results_y = 400
    results_w = 400
    results_h = 300
    
    cv2.rectangle(unified, (results_x, results_y), (results_x + results_w, results_y + results_h), 
                 (200, 200, 200), -1)
    cv2.rectangle(unified, (results_x, results_y), (results_x + results_w, results_y + results_h), 
                 (0, 0, 0), 2)
    
    y_pos = results_y + 30
    line_height = 25
    
    cv2.putText(unified, f"Category: {category.upper()}", (results_x + 10, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    y_pos += line_height
    
    cv2.putText(unified, f"Overall Score: {overall_score:.2%}", (results_x + 10, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    y_pos += line_height
    
    cv2.putText(unified, f"SSIM: {score_details['ssim']:.2%}", (results_x + 10, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    y_pos += line_height
    
    cv2.putText(unified, f"Contour: {score_details['contour']:.2%}", (results_x + 10, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    y_pos += line_height
    
    cv2.putText(unified, f"Histogram: {score_details['histogram']:.2%}", (results_x + 10, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    y_pos += line_height
    
    cv2.putText(unified, f"Contours Matched: {contour_comparison['matched_contours']}/{contour_comparison['total_compared']}",
               (results_x + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    y_pos += line_height
    
    cv2.putText(unified, f"Match %: {contour_comparison['match_percentage']:.1f}%",
               (results_x + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Title
    cv2.putText(unified, "Image Comparison with QuickDraw Dataset", (50, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return unified

# ============================================
# MAIN EXECUTION
# ============================================
print("=" * 60)
print("🎨 Image Comparison with QuickDraw Dataset")
print("=" * 60)

# Get category from user
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

print(f"\n🎨 Comparing with: {category.upper()}")
print("=" * 60)

# Load and preprocess uploaded image
try:
    print(f"\n📂 Loading image: {IMAGE_PATH}")
    user_img, original_img = preprocess_uploaded_image(IMAGE_PATH)
    print("✅ Image loaded and preprocessed successfully")
except Exception as e:
    print(f"❌ Error loading image: {e}")
    print(f"   Make sure IMAGE_PATH is set correctly at the top of the code!")
    raise

# Get reference drawing from QuickDraw
try:
    print(f"🔍 Fetching QuickDraw reference for '{category}'...")
    drawing = get_quickdraw_reference(category)
    reference_img = render_drawing_to_image(drawing)
    reference_img = normalize_drawing(reference_img)
    print("✅ Reference drawing loaded successfully")
except Exception as e:
    print(f"❌ Error loading QuickDraw reference: {e}")
    raise

# Get contour statistics
print("\n📊 Analyzing contours...")
user_stats = get_contour_statistics(user_img)
ref_stats = get_contour_statistics(reference_img)

# Compare contours in detail
contour_comparison = compare_contours_detailed(user_img, reference_img)

# Get overall similarity scores
overall_score, score_details = compare_drawings_combined(user_img, reference_img)

# Display results
print("\n" + "=" * 60)
print("📈 CONTOUR ANALYSIS RESULTS")
print("=" * 60)

print(f"\n📐 Your Image Contours:")
print(f"   • Total Contours (all): {user_stats['total_contours']}")
print(f"   • External Contours: {user_stats['external_contour_count']}")
print(f"   • Total Contour Points: {user_stats['total_points']}")
print(f"   • External Contour Points: {user_stats['external_points']}")
print(f"   • Total Contour Area: {user_stats['total_area']:.2f} pixels²")

print(f"\n📐 QuickDraw Reference Contours:")
print(f"   • Total Contours (all): {ref_stats['total_contours']}")
print(f"   • External Contours: {ref_stats['external_contour_count']}")
print(f"   • Total Contour Points: {ref_stats['total_points']}")
print(f"   • External Contour Points: {ref_stats['external_points']}")
print(f"   • Total Contour Area: {ref_stats['total_area']:.2f} pixels²")

print(f"\n🔄 Contour Comparison:")
print(f"   • Contours Compared: {contour_comparison['total_compared']}")
print(f"   • Matched Contours (similarity > 0.5): {contour_comparison['matched_contours']}")
print(f"   • Match Percentage: {contour_comparison['match_percentage']:.2f}%")
print(f"   • Contour Similarity Score: {contour_comparison['similarity']:.2%}")

print("\n" + "=" * 60)
print("🎯 OVERALL SIMILARITY SCORES")
print("=" * 60)
print(f"\n📊 Combined Similarity Score: {overall_score:.2%}")
print(f"   • SSIM Score: {score_details['ssim']:.2%}")
print(f"   • Contour Match Score: {score_details['contour']:.2%}")
print(f"   • Histogram Correlation: {score_details['histogram']:.2%}")

# Create visual comparison
print("\n🖼️  Displaying visual comparison...")
comparison_display = create_comparison_display(user_img, reference_img, original_img, category,
                                              user_stats, ref_stats, contour_comparison,
                                              overall_score, score_details)

cv2.imshow("Image Comparison Results - Press any key to close", comparison_display)

print("\n✅ Analysis complete! Press any key in the image window to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("📋 SUMMARY")
print("=" * 60)
print(f"Overall Similarity: {overall_score:.2%}")
print(f"Contour Match: {contour_comparison['matched_contours']}/{contour_comparison['total_compared']} ({contour_comparison['match_percentage']:.1f}%)")
print(f"Your Contours: {user_stats['external_contour_count']} | Reference Contours: {ref_stats['external_contour_count']}")
print("=" * 60)
