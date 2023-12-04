import cv2
import numpy as np

# Draw outline around the pocket (goal)
def draw_pockets(image):
    pocket_coordinates = [
        ((1, 110), (75, 30)),  # Top left pocket
        ((985, 85), (940, 15)),  # Top middle pocket
        ((1850, 108), (1917, 28)),  # Top right pocket
        ((1, 980), (70, 1055)),  # Bottom left pocket
        ((938, 1000), (993, 1073)),  # Bottom middle pocket
        ((1860, 978), (1918, 1055))  # Bottom right pocket
    ]

    for top_left, bottom_right in pocket_coordinates:
        cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 4)

# Draw circle around the ball
def draw_circles(contours, image, min_diameter=23, max_diameter=30):
    result_image = image.copy()
    
    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] == 0:
            continue

        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])

        radius = max(min_diameter // 2, min(max_diameter // 2, int(np.sqrt(cv2.contourArea(contour) / np.pi))))

        cv2.circle(result_image, (cX, cY), radius, (0, 255, 0), 2)

    return result_image
    
def filter_ctrs(ctrs, alpha=1.0, min_s=100, max_s=5000):
    filtered_ctrs = []

    for x in range(len(ctrs)):
        rot_rect = cv2.minAreaRect(ctrs[x])
        w = rot_rect[1][0]
        h = rot_rect[1][1]
        area = cv2.contourArea(ctrs[x])

        if (h * alpha < w) or (w * alpha < h):
            continue

        if (area < min_s) or (area > max_s):
            continue

        filtered_ctrs.append(ctrs[x])

    return filtered_ctrs

def find_ctrs_color(ctrs, input_img):
    K = np.ones((3, 3), np.uint8)
    output = input_img.copy()
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, np.uint8)

    for i in range(len(ctrs)):
        M = cv2.moments(ctrs[i])
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])

        mask[...] = 0
        cv2.drawContours(mask, ctrs, i, 255, -1)
        mask = cv2.erode(mask, K, iterations=3)

        output = cv2.circle(output, (cX, cY), 20, cv2.mean(input_img, mask), -1)

    return output

def filter_ctrs_circularity(ctrs, circularity_threshold=0.7):
    filtered_ctrs = []

    for contour in ctrs:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter * perimeter)

        if circularity > circularity_threshold:
            filtered_ctrs.append(contour)

    return filtered_ctrs

def process_frame(frame):
    transformed_blur = cv2.GaussianBlur(frame, (5, 5), 2)
    blur_RGB = cv2.cvtColor(transformed_blur, cv2.COLOR_BGR2RGB)

    lower = np.array([50, 120, 30])
    upper = np.array([70, 255, 255])

    hsv = cv2.cvtColor(blur_RGB, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask_closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    _, mask_inv = cv2.threshold(mask_closing, 5, 255, cv2.THRESH_BINARY_INV)

    masked_img = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # Shadow filtering
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh_gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    mask_inv_no_shadow = cv2.bitwise_and(mask_inv, thresh_gray)

    # Improved edge detection using Canny (50,150)
    edges = cv2.Canny(mask_inv_no_shadow, 50, 150)

    # Dilation to connect edges
    dilated_edges = cv2.dilate(edges, None, iterations=2)

    # Find contours in the dilated edges
    ctrs, hierarchy = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on circularity
    ctrs_filtered_circularity = filter_ctrs_circularity(ctrs)
    detected_objects = draw_circles(ctrs_filtered_circularity, frame, min_diameter=25, max_diameter=30)
    # 23, 35

    ctrs_filtered = filter_ctrs(ctrs)
    detected_objects_filtered = draw_circles(ctrs_filtered, frame, min_diameter=23, max_diameter=30)

    ctrs_color = find_ctrs_color(ctrs_filtered, frame)
    ctrs_color = cv2.addWeighted(ctrs_color, 0.5, frame, 0.5, 0)

    return detected_objects

# Read video file
video_path = "snooker_video.mp4"
cap = cv2.VideoCapture(video_path)

cv2.namedWindow('output', cv2.WINDOW_NORMAL)
cv2.moveWindow('output', 0, 0)  

font = cv2.FONT_HERSHEY_SIMPLEX
player1_score = 0
player2_score = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Process the frame
    result = process_frame(frame)

    # Draw pockets on the frame
    draw_pockets(result)

    # Resize the video frame to a common width (1600 pixels) for combining
    output_frame_resized = cv2.resize(result, (1600, 1080))

    # Create a combined frame for video and scoreboard
    combined_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    combined_frame[:, :1600] = output_frame_resized  # Display video frame
    combined_frame[:, 1600:] = (255, 255, 255)  # White background for scoreboard
    cv2.putText(combined_frame, f'Player 1: {player1_score}', (1650, 100), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(combined_frame, f'Player 2: {player2_score}', (1650, 200), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('output', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q        
        break

cap.release()
cv2.destroyAllWindows()