import cv2
import numpy as np

class Ball:
    def __init__(self, color):
        self.color = color
        self.position = None
        self.label = None

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

# Detect circle and filter unwanted contour
def filter_ctrs_circularity_and_size(ctrs, circularity_threshold=0.7, min_contour_size=68, max_contour_size=1000):
    filtered_ctrs = []

    for contour in ctrs:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter * perimeter)

        if circularity > circularity_threshold and min_contour_size < area < max_contour_size:
            filtered_ctrs.append(contour)

    return filtered_ctrs

# Draw circle arount the ball, detect color and count the amount of ball based on color
def draw_circles_with_color_assignment_and_count(ctrs, image, hsv_image, min_diameter=23, max_diameter=30):
    result_image = image.copy()
    balls = []
    ball_counts = {color: 0 for color in ['red', 'black', 'pink', 'brown', 'yellow', 'white', 'blue', 'green']}

    for contour in ctrs:
        M = cv2.moments(contour)
        if M['m00'] == 0:
            continue

        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])

        # Extract the region of interest (ROI) around the ball
        roi = hsv_image[cY - 15:cY + 15, cX - 15:cX + 15]

        # Calculate the average HSV values in the ROI
        avg_hsv = np.mean(roi, axis=(0, 1))

        # Define color ranges 
        color_ranges = {
            'red': ((0, 100, 100), (30, 255, 255)),
            'black': ((0, 0, 0), (180, 255, 50)),
            'pink': ((0, 50, 150), (100, 100, 255)),
            'brown': ((5, 50, 30), (40, 255, 150)),
            'yellow': ((28, 120, 150), (40, 255, 255)),
            'white': ((80, 50, 50), (100, 200, 255)),
            'blue': ((80, 50, 50), (140, 255, 255)),
            'green' : ((40, 50, 30), (80, 255, 255)),
        }

        ball_color = 'unknown'

        # Check the color of the ball by comparing with each color range
        for color, (lower, upper) in color_ranges.items():
            if all(lower[i] <= avg_hsv[i] <= upper[i] for i in range(3)):
                ball_color = color
                ball_counts[color] += 1  # Increment the count for the detected color
                break

        radius = max(min_diameter // 2, min(max_diameter // 2, int(np.sqrt(cv2.contourArea(contour) / np.pi))) if min_diameter <= np.sqrt(cv2.contourArea(contour) / np.pi) <= max_diameter else 0)
        cv2.circle(result_image, (cX, cY), radius, (0, 255, 0), 2)

        ball_found = False
        for ball in balls:
            # Check if the contour is close to an existing ball
            if abs(ball.position[0] - cX) < 15 and abs(ball.position[1] - cY) < 15:
                ball.position = (cX, cY)
                ball.color = ball_color
                ball_found = True
                break

        if not ball_found:
            new_ball = Ball(color=ball_color)
            new_ball.position = (cX, cY)
            balls.append(new_ball)

    # Add color labels and counts to the image
    for ball in balls:
        if ball.position is not None:
            if ball.color != 'unknown':
                cv2.putText(result_image, f'{ball.color} ({ball_counts[ball.color]})', (ball.position[0] - 10, ball.position[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(result_image, 'unknown', (ball.position[0] - 10, ball.position[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    return result_image, ball_counts

# Assuming player 1 starts the game
current_player = 1
player1_score = 0
player2_score = 0

# Function to detect pocketing of balls and update scores
def update_scores(ctrs_filtered_circularity_size):
    global current_player, player1_score, player2_score

    pocket_coordinates = [
        ((1, 110), (75, 30)),  # Top left pocket
        ((985, 85), (940, 15)),  # Top middle pocket
        ((1850, 108), (1917, 28)),  # Top right pocket
        ((1, 980), (70, 1055)),  # Bottom left pocket
        ((938, 1000), (993, 1073)),  # Bottom middle pocket
        ((1860, 978), (1918, 1055))  # Bottom right pocket
    ]

    for contour in ctrs_filtered_circularity_size:
        # Find the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)
        cX = x + w // 2
        cY = y + h // 2

        for pocket in pocket_coordinates:
            if pocket[0][0] <= cX <= pocket[1][0] and pocket[0][1] <= cY <= pocket[1][1]:
                # Update scores based on the current player
                if current_player == 1:
                    player1_score += 1
                else:
                    player2_score += 1

                # Switch to the next player
                current_player = 2 if current_player == 1 else 1
                break  # Exit loop once a pocket is found
            
# Read video file
video_path = "snooker_video.mp4"
cap = cv2.VideoCapture(video_path)

cv2.namedWindow('output', cv2.WINDOW_NORMAL)
cv2.moveWindow('output', 0, 0)

font = cv2.FONT_HERSHEY_SIMPLEX
player1_score = 0
player2_score = 0

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

    # Filter contours based on circularity and size
    ctrs_filtered_circularity_size = filter_ctrs_circularity_and_size(ctrs, min_contour_size=68, max_contour_size=1000)

    # Call the modified function to draw circles and get ball counts
    detected_objects, _ = draw_circles_with_color_assignment_and_count(ctrs_filtered_circularity_size, frame, hsv, min_diameter=25, max_diameter=30)

    return detected_objects, ctrs_filtered_circularity_size

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Process the frame
    detected_objects, ctrs_filtered_circularity_size = process_frame(frame)

    # Draw pockets on the frame
    draw_pockets(detected_objects)

    # Call function to update scores based on detected balls (detected contours)
    update_scores(ctrs_filtered_circularity_size)  # Pass the detected contours here

    # Resize the video frame to a common width (1600 pixels) for combining
    output_frame_resized = cv2.resize(detected_objects, (1600, 720))

    # Create a combined frame for video and scoreboard
    combined_frame = np.zeros((720, 1920, 3), dtype=np.uint8)
    combined_frame[:, :1600] = output_frame_resized  # Display video frame
    combined_frame[:, 1600:] = (0, 130, 0)  # Green background for scoreboard

    # Add a rectangle around the texts
    cv2.rectangle(combined_frame, (1645, 10), (1855, 130), (0, 0, 0), 2)

    # Add a rectangle around the ball count text
    cv2.rectangle(combined_frame, (1645, 145), (1855, 600), (0, 0, 0), 2)

    # Add text to display scores on the combined frame
    cv2.putText(combined_frame, f'SCORE', (1650, 35), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(combined_frame, f'Player 1: {player1_score}', (1650, 70), font, 0.7, (255, 0, 0), 1, cv2.LINE_AA)  
    cv2.putText(combined_frame, f'Player 2: {player2_score}', (1650, 105), font, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(combined_frame, f'Red Ball: 1 point', (1650, 165), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(combined_frame, f'Yellow Ball: 2 point', (1650, 195), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA) 
    cv2.putText(combined_frame, f'Green Ball: 3 point', (1650, 225), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA) 
    cv2.putText(combined_frame, f'Brown Ball: 4 point', (1650, 255), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(combined_frame, f'Blue Ball: 5 point', (1650, 285), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(combined_frame, f'Pink Ball: 6 point', (1650, 315), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(combined_frame, f'Black Ball: 7 point', (1650, 345), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Display the combined frame with ball colors and labels
    cv2.imshow('output', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()