import cv2
import math
import mediapipe as mp
from pynput.mouse import Button, Controller
import pyautogui

mouse=Controller()

cap = cv2.VideoCapture(0)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 

(screen_width, screen_height) = pyautogui.size()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

tipIds = [4, 8, 12, 16, 20]

pinch=False

# Define a function to count fingers
def countFingers(image, hand_landmarks, handNo=0):

	global pinch

	if hand_landmarks:
		# Get all Landmarks of the FIRST Hand VISIBLE
		landmarks = hand_landmarks[handNo].landmark

		# Count Fingers        
		fingers = []

		for lm_index in tipIds:
			# Get Finger Tip and Bottom y Position Value
			finger_tip_y = landmarks[lm_index].y 
			finger_bottom_y = landmarks[lm_index - 2].y

			thunmb_tip_x = int((landmarks[4].x) * width)
			finger_tip_x = int((landmarks[8].x) * width)

			thunmb_tip_y = int((landmarks[4].y) * height)
			finger_tip_y = int((landmarks[8].y) * height)


			# Check if ANY FINGER is OPEN or CLOSED
			if lm_index !=4:
				if finger_tip_y < finger_bottom_y:
					fingers.append(1)


				if finger_tip_y > finger_bottom_y:
					fingers.append(0)

		totalFingers = fingers.count(1)

		# PINCH

		# Draw a LINE between FINGER TIP and THUMB TIP
		cv2.line(image,(finger_tip_x,finger_tip_y),(thunmb_tip_x,thunmb_tip_y),(255,0,0),1.5)
		centerpoint_x = int((finger_tip_x + thunmb_tip_x)/2)
		centerpoint_y = int((finger_tip_y + thunmb_tip_y)/2)

		# Draw a CIRCLE on CENTER of the LINE between FINGER TIP and THUMB TIP
		cv2.circle(image,(centerpoint_x,centerpoint_y),2,(0,0,255),2)

		# Calculate DISTANCE between FINGER TIP and THUMB TIP
		dist = math.sqrt( ((finger_tip_x-thunmb_tip_x)**2) + ((finger_tip_y-thunmb_tip_y)**2) )

		# Set Mouse Position on the Screen Relative to the Output Window Size	
		mouse_pos_x = (centerpoint_x / width)*screen_width
		mouse_pos_y = (centerpoint_y/height)*screen_height

		mouse_pos = (mouse_pos_x,mouse_pos_y)

		# Check PINCH Formation Conditions
		if(dist <= 40):
			if(pinch == False):
				pinch = True
				mouse.press(Button.left)
		if(dist > 40):
			if(pinch == True):
				pinch = False
				mouse.release(Button.left)


# Define a function to 
def drawHandLanmarks(image, hand_landmarks):

    # Darw connections between landmark points
    if hand_landmarks:

      for landmarks in hand_landmarks:
               
        mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)



while True:
	success, image = cap.read()
	
	image = cv2.flip(image, 1)

	# Detect the Hands Landmarks 
	results = hands.process(image)

	# Get landmark position from the processed result
	hand_landmarks = results.multi_hand_landmarks

	# Draw Landmarks
	drawHandLanmarks(image, hand_landmarks)

	# Get Hand Fingers Position        
	countFingers(image, hand_landmarks)

	cv2.imshow("Media Controller", image)

	# Quit the window on pressing Sapcebar key
	key = cv2.waitKey(1)
	if key == 27:
		break

cv2.destroyAllWindows()
