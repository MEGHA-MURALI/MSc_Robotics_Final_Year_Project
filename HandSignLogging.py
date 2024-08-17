#Reference: https://github.com/kinivi/hand-gesture-recognition-mediapipe

import cv2
import mediapipe as mp
import copy
import itertools
import csv
import sys


# This function is taken from the github project mentioned at top
# Calculate the landmark coordinates relative to the captured frame size
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

# This function is taken from the github project mentioned at top
# Normalize the landmark distances measurements from wrist (0th landmark)
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

# Takes the hand gesture id, landmark list and writing it in to the keypoints.csv file
def logging_csv(number, landmark_list):
    csv_path = 'dataset/keypoint.csv'
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([number, *landmark_list])
        # Printing the content written in to the list to confirm the capture
        print("Saved: ",str(number),"-",landmark_list)
    return




# Copying mediapipe instances of hand skeleton drawing and hand detection model instances to custom variables
mp_drawing = mp.solutions.drawing_utils
#mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# initializing video capturing device 'default webcam'
cap = cv2.VideoCapture(0)

# To store number of images taken for each gesture category. Will be incremented dynamically by 1 when a record is saved
sign_count = 0

#Name of the hand gesture passed as an argument
signName = sys.argv[1]

# Initializing hand detection instance with custom parameters
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5,max_num_hands=1) as hands:
    # While loop will run as long as the camera is in a functional state
    while cap.isOpened():
        # Reading image from the camera
        success, image = cap.read()

        # Flipping captured image horizontally to get selfie view
        image = cv2.flip(image, 1)

        # If the camera is functional but did not capture an image, the while loop will keep restarting from here
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False

        # Converting OpenCV native BGR color scheme to mediapipe compatible RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generating hand landmark list
        results = hands.process(image)

        # Creating a completely independent copy of the image
        # (Simple assignment creates only bindings between target and object)
        debug_image = copy.deepcopy(image)

        # Converting image back to BRG format of OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Waits 1 millisecond to check key presses and returns it to variable k
        # Can be used to control the speed of the while loop execution
        k = cv2.waitKey(1)

        # Rest of the code will continue only if the results contains landmarks of a hand or hands.
        # If not that means no hands are present in the frame.
        if results.multi_hand_landmarks is not None:
            # Iterate through each landmark detected in the hand
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Write to the dataset file after checking which key is pressed on the keyboard
                if k == ord('0'):
                    # Writing content in to CSV file as a data record
                    logging_csv(signName,pre_processed_landmark_list)
                    # Incrementing record count for gesture 0
                    sign_count += 1


            # Drawing skeleton on the frame
            for hand_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)

        font = cv2.FONT_HERSHEY_SIMPLEX
        origin = (50, 50)
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2

        image = cv2.putText(image,
                            str(sign_count)+" Records captured for "+str(signName),
                            origin,
                            font,
                            fontScale,
                            color,
                            thickness,
                            cv2.LINE_AA)


        # Displaying frame with the landmark skeleton overlay
        cv2.imshow('MediaPipe Hand Landmark', image) 

        # Exits the main loop if 'Esc' pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Releasing handle to camera before terminating the programme
cap.release()
