import cv2
import numpy as np
from quickdraw import QuickDrawData
from skimage.metrics import structural_similarity as ssim
import time
import random

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
    # Convert to binary
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Find bounding box of the drawing
    coords = np.column_stack(np.where(binary == 0))
    if len(coords) == 0:
        return img
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Crop to bounding box
    cropped = binary[y_min:y_max+1, x_min:x_max+1]
    
    # Add padding and resize to standard size
    h, w = cropped.shape
    max_dim = max(h, w)
    if max_dim == 0:
        return img
    
    # Create square canvas with padding
    pad = max_dim // 10
    square_size = max_dim + 2 * pad
    square = np.ones((square_size, square_size), dtype=np.uint8) * 255
    
    # Center the drawing
    y_offset = (square_size - h) // 2
    x_offset = (square_size - w) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = cropped
    
    # Resize to standard size
    normalized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    
    return normalized

def preprocess_canvas(canvas):
    """Preprocess user canvas drawing."""
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Normalize the drawing
    normalized = normalize_drawing(thresh)
    
    return normalized

def detect_yellow_object(frame):
    """Detect yellow colored object in the frame."""
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range for yellow color in HSV
    # Yellow hue is around 20-30 in OpenCV HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    
    # Create mask for yellow color
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # Get the largest contour (assuming it's the yellow object)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate centroid
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy), mask, largest_contour
        else:
            return None, mask, None
    else:
        return None, mask, None

def get_contour_statistics(img):
    """Get detailed contour statistics from an image."""
    # Find all contours
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out very small contours (noise)
    min_contour_area = 5
    filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_contour_area]
    
    # Get external contours only (for comparison)
    external_contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_external = [c for c in external_contours if cv2.contourArea(c) >= min_contour_area]
    
    # Calculate total contour points
    total_points = sum(len(c) for c in filtered_contours)
    external_points = sum(len(c) for c in filtered_external)
    
    # Calculate total contour area
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
    # Get contour statistics for both images
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
    
    # Compare contours using shape matching
    match_scores = []
    matched_pairs = []
    
    # For each user contour, find best matching reference contour
    for i, user_contour in enumerate(user_contours):
        best_match_score = float('inf')
        best_match_idx = -1
        
        for j, ref_contour in enumerate(ref_contours):
            # Use Hu moments for shape matching
            match_score = cv2.matchShapes(user_contour, ref_contour, cv2.CONTOURS_MATCH_I2, 0)
            
            if match_score < best_match_score:
                best_match_score = match_score
                best_match_idx = j
        
        if best_match_idx != -1:
            match_scores.append(best_match_score)
            matched_pairs.append((i, best_match_idx, best_match_score))
    
    # Calculate similarity from match scores (lower score = more similar)
    if len(match_scores) > 0:
        # Convert match scores to similarity (0-1 scale)
        # Lower match_score means more similar, so we invert it
        similarities = [1.0 / (1.0 + score) for score in match_scores]
        avg_similarity = np.mean(similarities)
    else:
        avg_similarity = 0.0
    
    # Calculate how many contours were matched
    # A match is considered good if similarity > 0.5
    good_matches = sum(1 for score in match_scores if 1.0 / (1.0 + score) > 0.5)
    
    # Calculate percentage of contours matched
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
    # Find contours
    user_contours, _ = cv2.findContours(user_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ref_contours, _ = cv2.findContours(ref_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(user_contours) == 0 or len(ref_contours) == 0:
        return 0.0
    
    # Get largest contours
    user_largest = max(user_contours, key=cv2.contourArea)
    ref_largest = max(ref_contours, key=cv2.contourArea)
    
    # Match shapes
    match_score = cv2.matchShapes(user_largest, ref_largest, cv2.CONTOURS_MATCH_I2, 0)
    
    # Convert to similarity (lower match_score = more similar)
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
    
    # Weighted combination
    combined = (0.4 * ssim_score + 0.4 * contour_score + 0.2 * hist_score)
    
    return combined, {
        'ssim': ssim_score,
        'contour': contour_score,
        'histogram': hist_score
    }

def visualize_contours(img, stats, title="Contours"):
    """Visualize contours on image."""
    vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Draw all external contours
    cv2.drawContours(vis_img, stats['external_contours'], -1, (0, 255, 0), 1)
    
    # Draw all contours (including internal)
    cv2.drawContours(vis_img, stats['all_contours'], -1, (255, 0, 0), 1)
    
    return vis_img

def main():
    """Main function to compare air drawing with QuickDraw shapes."""
    print("=" * 60)
    print("🎨 Air Drawing Comparison with QuickDraw Dataset")
    print("=" * 60)
    
    # Get category from user
    print("\nAvailable categories:")
    for i, word in enumerate(SIMPLE_WORDS, 1):
        print(f"  {i:2d}. {word}")
    
    category_input = input("\nEnter category name or number: ").strip().lower()
    
    # Check if input is a number
    try:
        category_num = int(category_input)
        if 1 <= category_num <= len(SIMPLE_WORDS):
            category = SIMPLE_WORDS[category_num - 1]
        else:
            print(f"Invalid number. Using default: {SIMPLE_WORDS[0]}")
            category = SIMPLE_WORDS[0]
    except ValueError:
        # Input is a word
        if category_input in SIMPLE_WORDS:
            category = category_input
        else:
            print(f"Category '{category_input}' not found. Using default: {SIMPLE_WORDS[0]}")
            category = SIMPLE_WORDS[0]
    
    print(f"\n🎨 Draw: {category.upper()}")
    print("=" * 60)
    
    # Get reference drawing from QuickDraw
    try:
        print(f"🔍 Fetching QuickDraw reference for '{category}'...")
        drawing = get_quickdraw_reference(category)
        reference_img = render_drawing_to_image(drawing)
        reference_img = normalize_drawing(reference_img)
        print("✅ Reference drawing loaded successfully")
    except Exception as e:
        print(f"❌ Error loading QuickDraw reference: {e}")
        return
    
    print("\nInstructions:")
    print("  • Show a YELLOW colored object to the camera")
    print("  • Move the yellow object to draw in the air")
    print("  • Press 's' to submit your drawing")
    print("  • Press 'c' to clear canvas")
    print("  • Press 'q' to quit")
    print("  • Press 'n' for a new random word\n")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    time.sleep(2)  # Allow camera to warm up
    
    canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    prev_center = None
    drawing_active = False
    drawing_points = []  # Track all drawing points
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Detect yellow object
        center, mask, contour = detect_yellow_object(frame)
        
        # Draw mask for debugging (optional)
        mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        if center is not None:
            # Draw yellow object position on frame
            cv2.circle(frame, center, 15, (0, 255, 255), -1)
            cv2.circle(frame, center, 20, (0, 255, 255), 2)
            
            # Draw contour on frame
            if contour is not None:
                cv2.drawContours(frame, [contour], -1, (0, 255, 255), 2)
            
            # Draw on canvas
            if prev_center:
                cv2.line(canvas, prev_center, center, (0, 0, 0), 5)
                drawing_active = True
                drawing_points.append(center)
            
            prev_center = center
        else:
            prev_center = None
        
        # Add text overlay
        cv2.putText(frame, f"Draw: {category.upper()}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Draw: {category.upper()}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        cv2.putText(frame, "Yellow Object Detector", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display frames
        cv2.imshow("Air Drawing - Webcam Feed (Yellow Object)", frame)
        cv2.imshow("Yellow Object Mask", mask_display)
        cv2.imshow("Air Canvas - Press 's' to Submit", canvas)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            if drawing_active:
                break
            else:
                print("Please draw something first!")
        elif key == ord('c'):
            canvas[:] = 255
            drawing_active = False
            prev_center = None
            drawing_points = []
            print("Canvas cleared!")
        elif key == ord('n'):
            # Get new random word
            category = get_random_word()
            print(f"\n🎨 New word: {category.upper()}")
            canvas[:] = 255
            drawing_active = False
            prev_center = None
            drawing_points = []
            try:
                drawing = get_quickdraw_reference(category)
                reference_img = render_drawing_to_image(drawing)
                reference_img = normalize_drawing(reference_img)
            except ValueError as e:
                print(f"Error: {e}")
        elif key == ord('q'):
            print("Exiting.")
            cap.release()
            cv2.destroyAllWindows()
            return
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Check if drawing has enough content
    if len(drawing_points) < 10:
        print("\n⚠️  Drawing too short! Please draw more.")
        return
    
    # Calculate drawing area coverage
    if len(drawing_points) > 0:
        points_array = np.array(drawing_points)
        x_range = points_array[:, 0].max() - points_array[:, 0].min()
        y_range = points_array[:, 1].max() - points_array[:, 1].min()
        coverage = (x_range * y_range) / (640 * 480)
        if coverage < 0.01:  # Less than 1% of canvas
            print("\n⚠️  Drawing too small! Please draw larger.")
            return
    
    # Process user drawing
    print("\n📊 Processing your drawing...")
    user_img = preprocess_canvas(canvas)
    
    # Check if user drawing has enough content
    user_pixels = np.sum(user_img == 0)
    if user_pixels < 50:  # Too few black pixels
        print("\n⚠️  Drawing too sparse! Please draw more clearly.")
        return
    
    # Get contour statistics
    print("📊 Analyzing contours...")
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
    
    print(f"\n📐 User Drawing Contours:")
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
    
    # Visualize
    print("\n🖼️  Displaying visualizations...")
    
    # Create visualization images
    user_vis = visualize_contours(user_img, user_stats, "User Drawing")
    ref_vis = visualize_contours(reference_img, ref_stats, "Reference")
    
    # Resize for display
    display_size = (400, 400)
    user_display = cv2.resize(user_vis, display_size)
    ref_display = cv2.resize(ref_vis, display_size)
    
    # Create side-by-side comparison
    comparison = np.hstack([user_display, ref_display])
    
    # Add text labels
    cv2.putText(comparison, "Your Drawing", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(comparison, f"QuickDraw: {category.upper()}", (420, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Add contour count labels
    cv2.putText(comparison, f"Contours: {user_stats['external_contour_count']}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(comparison, f"Contours: {ref_stats['external_contour_count']}", (420, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show images
    cv2.imshow("Contour Comparison - Press any key to close", comparison)
    
    # Also show normalized images side by side
    norm_comparison = np.hstack([user_img, reference_img])
    norm_comparison_resized = cv2.resize(norm_comparison, (560, 140))
    cv2.putText(norm_comparison_resized, "Your Drawing (Normalized)", (10, 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(norm_comparison_resized, f"Reference: {category.upper()}", (290, 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imshow("Normalized Comparison - Press any key to close", norm_comparison_resized)
    
    print("\n✅ Analysis complete! Press any key in the image windows to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    print(f"Overall Similarity: {overall_score:.2%}")
    print(f"Contour Match: {contour_comparison['matched_contours']}/{contour_comparison['total_compared']} ({contour_comparison['match_percentage']:.1f}%)")
    print(f"User Contours: {user_stats['external_contour_count']} | Reference Contours: {ref_stats['external_contour_count']}")
    print("=" * 60)

if __name__ == "__main__":
    main()