import cv2
import numpy as np

def draw_pockets(image):
    pocket_coordinates = [
        (60, 60), (960, 60), (1860, 60),  # Top row
        (60, 1020), (960, 1020), (1860, 1020)  # Bottom row
    ]

    for pocket_coord in pocket_coordinates:
        cv2.rectangle(image, (pocket_coord[0] - 20, pocket_coord[1] - 20), (pocket_coord[0] + 20, pocket_coord[1] + 20), (255, 0, 0), 2)

def draw_circles(contours, image, min_diameter=25, max_diameter=30):
    result_image = image.copy()

    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] == 0:
            continue

        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])

        # Create an empty mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Draw the current contour on the mask
        cv2.drawContours(mask, [contour], 0, 255, thickness=cv2.FILLED)

        # Extract the average color inside the contour
        average_color = cv2.mean(image, mask)

        # Calculate the radius
        radius = max(min_diameter // 2, min(max_diameter // 2, int(np.sqrt(cv2.contourArea(contour) / np.pi))))

        # Use the average color for the outline
        cv2.circle(result_image, (cX, cY), radius, average_color[:3], 2)

    return result_image

# Function to identify potted balls
def hit_cue_ball(frame, player, reds_potted, colored_potted, player_score):
    # Ball detection logic 
    potted_balls = ["red", "yellow"] 

    for ball in potted_balls:
        if ball == "red":
            reds_potted += 1
            player_score += 1

            if colored_potted < 1:
                break
            else:
                colored_potted = 0

        elif ball in ["yellow", "green", "brown", "blue", "pink", "black"]:
            colored_potted += 1

            # Update player_score based on the color
            color_scores = {"yellow": 2, "green": 3, "brown": 4, "blue": 5, "pink": 6, "black": 7}
            player_score += color_scores[ball]

            # Check if it's the end of the player's turn
            if colored_potted > 1 or reds_potted < 1:
                print(f"End {player}'s turn (incorrect sequence)")
                reds_potted = 0
            else:
                reds_potted = 0

        else:
            # No valid balls potted, end current player's turn (foul)
            print(f"End {player}'s turn (foul)")
            reds_potted = 0
            colored_potted = 0

    return reds_potted, colored_potted, player_score
    
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

    ctrs, hierarchy = cv2.findContours(mask_inv_no_shadow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detected_objects = draw_circles(ctrs, frame, min_diameter=23, max_diameter=35)

    ctrs_filtered = filter_ctrs(ctrs)
    detected_objects_filtered = draw_circles(ctrs_filtered, frame, min_diameter=23, max_diameter=35)

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


