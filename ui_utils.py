import cv2

def draw_buttons(frame):
    # Define button properties
    
    return frame

def check_button_click(event, x, y, flags, param):
    frame = param['frame']
    button_width = 100
    button_height = 40

    # Bottom button position
    bottom_button_position = (frame.shape[1] - button_width - 20, frame.shape[0] - button_height - 20)
    if bottom_button_position[0] <= x <= bottom_button_position[0] + button_width and bottom_button_position[1] <= y <= bottom_button_position[1] + button_height:
        print("Save button clicked!")
        # Add functionality to save the image or perform an action

    # Right button position
    right_button_position = (frame.shape[1] - button_width - 20, 20)
    if right_button_position[0] <= x <= right_button_position[0] + button_width and right_button_position[1] <= y <= right_button_position[1] + button_height:
        print("Alert button clicked!")
        # Add functionality to trigger alert
        param['email_alert']("Test Alert", "This is a test alert message.", "test@example.com")
