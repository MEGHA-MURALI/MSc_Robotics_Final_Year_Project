import cv2
import mediapipe as mp
import copy
import itertools
import pickle

# This function is taken from the github project mentioned in the report
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

# This function is taken from the github project mentioned in the report
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


# Copying mediapipe instances of hand skeleton drawing and hand detection model instances to custom variables

mp_drawing = mp.solutions.drawing_utils
#mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


# initializing video capturing device 'default webcam'
cap = cv2.VideoCapture(0)

# Used to extract correct gesture name form 'sign_labels' list
signNum = -1

# Initializing hand detection instance with custom parameters
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7,max_num_hands=1) as hands:
    #model = pickle.load(open('model/finalized_model.sav', 'rb'))
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

            
            # Drawing skeleton on the frame
            for hand_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)
                
            # Predicting current gesture
            model = pickle.load(open('models/finalized_model_keypoint.sav', 'rb'))
            result = model.predict([pre_processed_landmark_list])
            print(result)
            image = cv2.putText(image, str(result), (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                2, cv2.LINE_AA)

        # Displaying frame with the landmark skeleton overlay
        cv2.imshow('MediaPipe Hands', image)

        # Exits the main loop if 'Esc' pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Releasing handle to camera before terminating the programme
cap.release()
