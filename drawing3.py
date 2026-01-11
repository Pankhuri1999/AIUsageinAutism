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

# Drawing colors
COLORS = {
    'black': (0, 0, 0),
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'yellow': (0, 255, 255),
    'purple': (255, 0, 255),
    'orange': (0, 165, 255)
}
COLOR_NAMES = list(COLORS.keys())
current_color = 'black'
current_color_index = 0

# Drawing modes
DRAW_MODE = 0
ERASE_MODE = 1
current_mode = DRAW_MODE

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

def preprocess_canvas(canvas):
    """Preprocess user canvas drawing."""
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    normalized = normalize_drawing(thresh)
    return normalized

def detect_yellow_object(frame):
    """Detect yellow colored object in the frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy), mask, largest_contour
    return None, mask, None

def calculate_focal_point(drawing_points, canvas_size):
    """Calculate focal point for large drawings."""
    if len(drawing_points) < 10:
        return None
    
    points_array = np.array(drawing_points)
    x_min, y_min = points_array.min(axis=0)
    x_max, y_max = points_array.max(axis=0)
    
    # Calculate bounding box
    width = x_max - x_min
    height = y_max - y_min
    
    # If drawing area is large (>40% of canvas), focus on center of drawing
    canvas_area = canvas_size[0] * canvas_size[1]
    drawing_area = width * height
    area_ratio = drawing_area / canvas_area
    
    if area_ratio > 0.4:
        focal_x = (x_min + x_max) // 2
        focal_y = (y_min + y_max) // 2
        return (focal_x, focal_y), (width, height)
    
    return None, None

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

def create_unified_frame(frame, canvas, mask, reference_img, category, center, focal_point, 
                        drawing_points, show_mask=True):
    """Create a single unified frame with all elements."""
    h, w = frame.shape[:2]
    
    # Create main display frame (1280x720 for better layout)
    display_h, display_w = 720, 1280
    unified = np.ones((display_h, display_w, 3), dtype=np.uint8) * 240
    
    # Resize components
    cam_w, cam_h = 480, 360
    canvas_w, canvas_h = 400, 300
    ref_w, ref_h = 200, 200
    mask_w, mask_h = 200, 150
    
    # Resize frame
    frame_resized = cv2.resize(frame, (cam_w, cam_h))
    
    # Resize canvas
    canvas_resized = cv2.resize(canvas, (canvas_w, canvas_h))
    
    # Resize reference
    ref_resized = cv2.resize(reference_img, (ref_w, ref_h))
    ref_colored = cv2.cvtColor(ref_resized, cv2.COLOR_GRAY2BGR)
    
    # Resize mask if needed
    if show_mask:
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_resized = cv2.resize(mask_colored, (mask_w, mask_h))
    
    # Layout: Top row - Camera, Canvas, Reference
    # Bottom row - Mask, Info panel
    
    # Top row
    y_offset_top = 50
    unified[y_offset_top:y_offset_top+cam_h, 10:10+cam_w] = frame_resized
    
    # Draw focal point on camera if exists
    if focal_point[0] is not None:
        fx, fy = focal_point[0]
        # Scale focal point to camera view
        scale_x = cam_w / w
        scale_y = cam_h / h
        fx_scaled = int(fx * scale_x)
        fy_scaled = int(fy * scale_y)
        cv2.circle(unified, (10 + fx_scaled, y_offset_top + fy_scaled), 20, (255, 0, 255), 3)
        cv2.putText(unified, "FOCUS", (10 + fx_scaled - 30, y_offset_top + fy_scaled - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    # Canvas
    canvas_x = 10 + cam_w + 10
    unified[y_offset_top:y_offset_top+canvas_h, canvas_x:canvas_x+canvas_w] = canvas_resized
    
    # Draw focal point on canvas if exists
    if focal_point[0] is not None:
        fx, fy = focal_point[0]
        scale_x = canvas_w / canvas.shape[1]
        scale_y = canvas_h / canvas.shape[0]
        fx_scaled = int(fx * scale_x)
        fy_scaled = int(fy * scale_y)
        cv2.circle(unified, (canvas_x + fx_scaled, y_offset_top + fy_scaled), 15, (255, 0, 255), 2)
    
    # Reference image
    ref_x = canvas_x + canvas_w + 10
    unified[y_offset_top:y_offset_top+ref_h, ref_x:ref_x+ref_w] = ref_colored
    
    # Bottom row
    y_offset_bottom = y_offset_top + canvas_h + 20
    
    # Mask
    if show_mask:
        unified[y_offset_bottom:y_offset_bottom+mask_h, 10:10+mask_w] = mask_resized
    
    # Info panel
    info_x = 10 + mask_w + 10 if show_mask else 10
    info_y = y_offset_bottom
    info_w = 400
    info_h = 150
    
    # Draw info panel background
    cv2.rectangle(unified, (info_x, info_y), (info_x + info_w, info_y + info_h), (200, 200, 200), -1)
    cv2.rectangle(unified, (info_x, info_y), (info_x + info_w, info_y + info_h), (0, 0, 0), 2)
    
    # Info text
    line_height = 20
    y_pos = info_y + 25
    
    cv2.putText(unified, f"Category: {category.upper()}", (info_x + 10, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    y_pos += line_height
    
    cv2.putText(unified, f"Color: {current_color.upper()}", (info_x + 10, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[current_color], 2)
    y_pos += line_height
    
    mode_text = "ERASE" if current_mode == ERASE_MODE else "DRAW"
    cv2.putText(unified, f"Mode: {mode_text}", (info_x + 10, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    y_pos += line_height
    
    if center:
        cv2.putText(unified, f"Tracking: ACTIVE", (info_x + 10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(unified, f"Tracking: INACTIVE", (info_x + 10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    y_pos += line_height
    
    if focal_point[0]:
        cv2.putText(unified, f"Focal Point: ACTIVE", (info_x + 10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    # Color palette (QuickDraw_V1 inspired)
    palette_x = info_x + info_w + 10
    palette_y = y_offset_bottom
    palette_size = 30
    spacing = 5
    
    for i, (name, color) in enumerate(COLORS.items()):
        px = palette_x
        py = palette_y + i * (palette_size + spacing)
        cv2.rectangle(unified, (px, py), (px + palette_size, py + palette_size), color, -1)
        if i == current_color_index:
            cv2.rectangle(unified, (px - 2, py - 2), (px + palette_size + 2, py + palette_size + 2), (0, 0, 0), 3)
        cv2.putText(unified, name[:3], (px + palette_size + 5, py + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Instructions
    inst_x = 10
    inst_y = display_h - 80
    cv2.putText(unified, "Controls: 's'=Submit | 'c'=Clear | 'e'=Erase | 'd'=Draw | 'n'=New Word | 'q'=Quit",
               (inst_x, inst_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(unified, "Color Change: Move yellow object to color palette area and hold",
               (inst_x, inst_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Title
    cv2.putText(unified, "QuickDraw Air Drawing - Unified View", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return unified

def main():
    """Main function to compare air drawing with QuickDraw shapes."""
    global current_color, current_color_index, current_mode
    
    print("=" * 60)
    print("🎨 Air Drawing Comparison with QuickDraw Dataset")
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
    
    print(f"\n🎨 Draw: {category.upper()}")
    print("=" * 60)
    
    # Get reference drawing
    try:
        drawing = get_quickdraw_reference(category)
        reference_img = render_drawing_to_image(drawing)
        reference_img = normalize_drawing(reference_img)
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    print("\nInstructions:")
    print("  • Show a YELLOW colored object to the camera")
    print("  • Move the yellow object to draw in the air")
    print("  • Press 'e' to toggle eraser mode")
    print("  • Press 'd' to toggle draw mode")
    print("  • Press 's' to submit your drawing")
    print("  • Press 'c' to clear canvas")
    print("  • Press 'q' to quit")
    print("  • Press 'n' for a new random word\n")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    time.sleep(2)
    
    canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255
    prev_center = None
    drawing_active = False
    drawing_points = []
    last_color_change_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Detect yellow object
        center, mask, contour = detect_yellow_object(frame)
        
        # Color selection (QuickDraw_V1 feature)
        # If yellow object is in palette area (right side), change color
        if center:
            cx, cy = center
            h, w = frame.shape[:2]
            # Check if in right 20% of screen (palette area)
            if cx > w * 0.8:
                current_time = time.time()
                if current_time - last_color_change_time > 1.0:  # Change color every 1 second
                    current_color_index = (current_color_index + 1) % len(COLOR_NAMES)
                    current_color = COLOR_NAMES[current_color_index]
                    last_color_change_time = current_time
        
        # Calculate focal point for large drawings
        focal_point, focal_size = calculate_focal_point(drawing_points, canvas.shape[:2])
        unified_frame = create_unified_frame(frame, canvas, mask, reference_img, category,
                                   center, (focal_point, focal_size), drawing_points)
        
        # Draw on canvas
        if center is not None:
            if prev_center:
                if current_mode == DRAW_MODE:
                    cv2.line(canvas, prev_center, center, COLORS[current_color], 5)
                else:  # ERASE_MODE
                    cv2.line(canvas, prev_center, center, (255, 255, 255), 10)
                drawing_active = True
                drawing_points.append(center)
            prev_center = center
        else:
            prev_center = None
        
        # Create unified frame
        unified_frame = create_unified_frame(frame, canvas, mask, reference_img, category,
                                           center, (focal_point, focal_size), drawing_points)
        
        cv2.imshow("QuickDraw Air Drawing - Unified View", unified_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            if drawing_active:
                break
        elif key == ord('c'):
            canvas[:] = 255
            drawing_active = False
            prev_center = None
            drawing_points = []
        elif key == ord('e'):
            current_mode = ERASE_MODE
            print("Eraser mode activated")
        elif key == ord('d'):
            current_mode = DRAW_MODE
            print("Draw mode activated")
        elif key == ord('n'):
            category = get_random_word()
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
            cap.release()
            cv2.destroyAllWindows()
            return
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Validation
    if len(drawing_points) < 10:
        print("\n⚠️  Drawing too short!")
        return
    
    # Process and analyze
    print("\n📊 Processing your drawing...")
    user_img = preprocess_canvas(canvas)
    
    user_pixels = np.sum(user_img == 0)
    if user_pixels < 50:
        print("\n⚠️  Drawing too sparse!")
        return
    
    # Get contour statistics
    print("📊 Analyzing contours...")
    user_stats = get_contour_statistics(user_img)
    ref_stats = get_contour_statistics(reference_img)
    contour_comparison = compare_contours_detailed(user_img, reference_img)
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
    
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    print(f"Overall Similarity: {overall_score:.2%}")
    print(f"Contour Match: {contour_comparison['matched_contours']}/{contour_comparison['total_compared']} ({contour_comparison['match_percentage']:.1f}%)")
    print(f"User Contours: {user_stats['external_contour_count']} | Reference Contours: {ref_stats['external_contour_count']}")
    print("=" * 60)

if __name__ == "__main__":
    main()
