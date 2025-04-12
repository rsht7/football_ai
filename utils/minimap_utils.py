import cv2
import numpy as np




def create_homography_minimap(frame, H, ball_positions):
    """Applies homography and plots ball trajectory on the minimap."""
    if H is None or len(ball_positions) == 0:
        return None  # Return None if homography matrix is missing or no ball data

    # Define the output minimap size (adjust as needed)
    height, width = 600, 800  
    warped = cv2.warpPerspective(frame, H, (width, height))  # Get bird's-eye view

    # Resize the minimap
    minimap = cv2.resize(warped, (250, 180))

    # Convert ball positions to numpy array for transformation
    ball_positions = np.array(ball_positions, dtype=np.float32).reshape(-1, 1, 2)

    # Transform ball positions using homography
    transformed_positions = cv2.perspectiveTransform(ball_positions, H)

    # Convert to integer coordinates
    transformed_positions = transformed_positions.reshape(-1, 2).astype(int)

    # Draw the ball path on the minimap
    for i in range(1, len(transformed_positions)):
        cv2.line(minimap, tuple(transformed_positions[i - 1]), tuple(transformed_positions[i]), (0, 0, 255), 2)

    return minimap


def overlay_minimap(frame, minimap, opacity=0.7, position=(50, 50)):
    """Overlays minimap onto the main video frame."""
    if minimap is None:
        return frame  # Skip if minimap isn't available

    x_offset, y_offset = position  # Bottom-right position
    h, w, _ = minimap.shape

    # Extract region of interest (ROI)
    roi = frame[y_offset:y_offset + h, x_offset:x_offset + w]

    # Blend the minimap with the ROI
    blended = cv2.addWeighted(roi, 1 - opacity, minimap, opacity, 0)

    # Replace original frame region with blended image
    frame[y_offset:y_offset + h, x_offset:x_offset + w] = blended

    return frame