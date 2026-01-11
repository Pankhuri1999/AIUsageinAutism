import cv2
import numpy as np
from quickdraw import QuickDrawData
from skimage.metrics import structural_similarity as ssim
import time
import random
from collections import deque

# --- Simple words for drawing game ---
SIMPLE_WORDS = [
    "cat", "dog", "tree", "house", "car", "sun", "moon", "star",
    "apple", "banana", "circle", "square", "triangle", "heart",
    "fish", "bird", "flower", "cloud", "mountain", "boat",
    "airplane", "bicycle", "cup", "spoon", "fork", "pencil",
    "book", "clock", "key", "umbrella", "rainbow", "butterfly"
]

# Drawing color - fixed to blue
DRAWING_COLOR = (255, 0, 0)  # Blue in BGR

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

def detect_blue_object(frame):
    """Detect blue colored object - multiple detection methods."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Method 1: QuickDraw_V1 style (narrower range)
    a1, b1, c1 = 120, 100, 15
    Lower1 = np.array([a1-c1, 50+b1, 50])
    Upper1 = np.array([a1+c1, 255, 255])
    mask1 = cv2.inRange(hsv, Lower1, Upper1)
    
    # Method 2: Wider blue range
    Lower2 = np.array([100, 50, 50])
    Upper2 = np.array([130, 255, 255])
    mask2 = cv2.inRange(hsv, Lower2, Upper2)
    
    # Method 3: Light blue range
    Lower3 = np.array([90, 50, 50])
    Upper3 = np.array([110, 255, 255])
    mask3 = cv2.inRange(hsv, Lower3, Upper3)
    
    # Combine all masks
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.bitwise_or(mask, mask3)
    
    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Find contours
    cnts, heir = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    center = None
    circle_info = None
    
    if len(cnts) > 0:
        # Sort by area and try largest ones
        cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)
        
        for cnt in cnts_sorted:
            area = cv2.contourArea(cnt)
            # Lower threshold - accept smaller objects
            if area > 100:  # Reduced from 200
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                    ((x, y), radius) = cv2.minEnclosingCircle(cnt)
                    circle_info = (int(x), int(y), int(radius))
                    break  # Use first valid detection
    
    return center, mask, None, circle_info

def calculate_focal_point(drawing_points, canvas_size):
    """Calculate focal point for large drawings."""
    if len(drawing_points) < 10:
        return None, None
    
    points_array = np.array(drawing_points)
    x_min, y_min = points_array.min(axis=0)
    x_max, y_max = points_array.max(axis=0)
    
    width = x_max - x_min
    height = y_max - y_min
    
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

def create_unified_frame(frame, canvas, mask, reference_img, category, center, circle_info, 
                        pts, focal_point, show_mask=True):
    """Create a single unified frame with drawing overlaid on camera feed."""
    h, w = frame.shape[:2]
    
    # Create main display frame (1280x720)
    display_h, display_w = 720, 1280
    unified = np.ones((display_h, display_w, 3), dtype=np.uint8) * 240
    
    # Resize components
    cam_w, cam_h = 640, 480
    canvas_w, canvas_h = 400, 300
    ref_w, ref_h = 200, 200
    mask_w, mask_h = 200, 150
    
    # Main camera feed with drawing overlay
    frame_with_drawing = frame.copy()
    
    # Draw the trail on camera feed
    if len(pts) > 1:
        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            if current_mode == DRAW_MODE:
                cv2.line(frame_with_drawing, pts[i - 1], pts[i], DRAWING_COLOR, 4)
            else:
                cv2.line(frame_with_drawing, pts[i - 1], pts[i], (255, 255, 255), 6)
    
    # Draw circle and center point
    if center is not None:
        if circle_info is not None:
            x, y, radius = circle_info
            cv2.circle(frame_with_drawing, (x, y), radius, (0, 255, 255), 2)
        cv2.circle(frame_with_drawing, center, 8, (0, 0, 255), -1)
        cv2.putText(frame_with_drawing, "BLUE DETECTED", (center[0] - 60, center[1] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        # Show message when not detected
        cv2.putText(frame_with_drawing, "NO BLUE DETECTED", (w//2 - 100, h//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame_with_drawing, "Show blue object to camera", (w//2 - 120, h//2 + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Resize frame with drawing
    frame_resized = cv2.resize(frame_with_drawing, (cam_w, cam_h))
    
    # Resize canvas
    canvas_resized = cv2.resize(canvas, (canvas_w, canvas_h))
    
    # Resize reference
    ref_resized = cv2.resize(reference_img, (ref_w, ref_h))
    ref_colored = cv2.cvtColor(ref_resized, cv2.COLOR_GRAY2BGR)
    
    # Resize mask
    if show_mask:
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_resized = cv2.resize(mask_colored, (mask_w, mask_h))
    
    # Layout
    y_offset_top = 50
    
    # Camera feed with drawing overlay
    unified[y_offset_top:y_offset_top+cam_h, 10:10+cam_w] = frame_resized
    
    # Focal point
    if focal_point[0] is not None:
        fx, fy = focal_point[0]
        scale_x = cam_w / w
        scale_y = cam_h / h
        fx_scaled = int(fx * scale_x)
        fy_scaled = int(fy * scale_y)
        cv2.circle(unified, (10 + fx_scaled, y_offset_top + fy_scaled), 20, (255, 0, 255), 3)
    
    # Canvas
    canvas_x = 10 + cam_w + 10
    unified[y_offset_top:y_offset_top+canvas_h, canvas_x:canvas_x+canvas_w] = canvas_resized
    
    # Reference
    ref_x = canvas_x + canvas_w + 10
    unified[y_offset_top:y_offset_top+ref_h, ref_x:ref_x+ref_w] = ref_colored
    
    # Bottom row
    y_offset_bottom = y_offset_top + canvas_h + 20
    
    # Mask
    if show_mask:
        unified[y_offset_bottom:y_offset_bottom+mask_h, 10:10+mask_w] = mask_resized
        cv2.putText(unified, "MASK (white=detected)", (10, y_offset_bottom - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Info panel
    info_x = 10 + mask_w + 10 if show_mask else 10
    info_y = y_offset_bottom
    info_w = 400
    info_h = 150
    
    cv2.rectangle(unified, (info_x, info_y), (info_x + info_w, info_y + info_h), (200, 200, 200), -1)
    cv2.rectangle(unified, (info_x, info_y), (info_x + info_w, info_y + info_h), (0, 0, 0), 2)
    
    line_height = 20
    y_pos = info_y + 25
    
    cv2.putText(unified, f"Category: {category.upper()}", (info_x + 10, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    y_pos += line_height
    
    cv2.putText(unified, f"Color: BLUE", (info_x + 10, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, DRAWING_COLOR, 2)
    y_pos += line_height
    
    mode_text = "ERASE" if current_mode == ERASE_MODE else "DRAW"
    cv2.putText(unified, f"Mode: {mode_text}", (info_x + 10, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    y_pos += line_height
    
    if center:
        cv2.putText(unified, f"Tracking: ACTIVE", (info_x + 10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += line_height
        cv2.putText(unified, f"Points: {len(pts)}", (info_x + 10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    else:
        cv2.putText(unified, f"Tracking: INACTIVE", (info_x + 10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_pos += line_height
        cv2.putText(unified, f"Show blue object!", (info_x + 10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Instructions
    inst_x = 10
    inst_y = display_h - 80
    cv2.putText(unified, "Controls: 's'=Submit | 'c'=Clear | 'e'=Erase | 'd'=Draw | 'n'=New Word | 'q'=Quit",
               (inst_x, inst_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(unified, "Check MASK window - white areas = detected blue. Draw on camera feed!",
               (inst_x, inst_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Title
    cv2.putText(unified, "QuickDraw Air Drawing - Draw on Camera Feed", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return unified

def main():
    """Main function to compare air drawing with QuickDraw shapes."""
    global current_mode
    
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
    print("  • Show a BLUE colored object to the camera")
    print("  • Check the MASK window (bottom left) - white = detected blue")
    print("  • Move the blue object to draw - you'll see your drawing on camera feed!")
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
    
    # Use deque for point tracking
    pts = deque(maxlen=512)
    blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
    
    drawing_active = False
    
    print("Starting detection...")
    print("TIP: Check the MASK window - if you see white areas when showing blue object, detection is working!")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Detect blue object
        center, mask, contour, circle_info = detect_blue_object(frame)
        
        # Track points
        if center is not None:
            pts.appendleft(center)
            drawing_active = True
        else:
            # Keep drawing even if briefly lost
            if len(pts) > 0:
                # Allow small gaps
                pass
        
        # Draw on blackboard
        if len(pts) > 1:
            for i in range(1, len(pts)):
                if pts[i - 1] is None or pts[i] is None:
                    continue
                if current_mode == DRAW_MODE:
                    cv2.line(blackboard, pts[i - 1], pts[i], (255, 255, 255), 7)
                else:
                    cv2.line(blackboard, pts[i - 1], pts[i], (0, 0, 0), 10)
        
        # Calculate focal point
        drawing_points = [p for p in pts if p is not None]
        focal_point, focal_size = calculate_focal_point(drawing_points, frame.shape[:2])
        
        # Create unified frame
        unified_frame = create_unified_frame(frame, blackboard, mask, reference_img, category,
                                           center, circle_info, pts, (focal_point, focal_size))
        
        cv2.imshow("QuickDraw Air Drawing - Draw on Camera Feed", unified_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            if drawing_active and len(pts) > 10:
                break
            else:
                print("⚠️  Please draw something first! Check MASK window to verify blue detection.")
        elif key == ord('c'):
            blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
            pts = deque(maxlen=512)
            drawing_active = False
            print("Canvas cleared!")
        elif key == ord('e'):
            current_mode = ERASE_MODE
            print("Eraser mode activated")
        elif key == ord('d'):
            current_mode = DRAW_MODE
            print("Draw mode activated")
        elif key == ord('n'):
            category = get_random_word()
            blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
            pts = deque(maxlen=512)
            drawing_active = False
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
    drawing_points = [p for p in pts if p is not None]
    if len(drawing_points) < 10:
        print("\n⚠️  Drawing too short!")
        return
    
    # Process and analyze
    print("\n📊 Processing your drawing...")
    user_img = preprocess_canvas(blackboard)
    
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
