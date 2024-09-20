#Reference: https://github.com/kinivi/hand-gesture-recognition-mediapipe

import cv2
import mediapipe as mp
import copy
import itertools
import csv
import playsound
import pickle
import os
from google.cloud import texttospeech
from pydub import AudioSegment
from pydub.playback import play
import io
import threading


lock = threading.Lock()

response = None
prev_result = None
the_same  = True

# This function is taken from the github project mentioned at top
# Calculate the landmark coordinates relative to the captured frame size
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    #print("=============================")
    #c = 0
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        #if c==0:
        #    print("landmark_x:landmark_y = ",str(landmark_x),":",str(landmark_y))
        #    c=c+1
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
    
    #print(temp_landmark_list)

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


def get_current_landmarks_both_hands(left_landmark_list, right_landmark_list):

    temp_left = [] #Considered as the main list to be returned. So righ hand deteils will be appended to this.
    temp_right = []

    for i in range(0,len(left_landmark_list)):
        temp_left.append(left_landmark_list[i])
        temp_right.append(right_landmark_list[i])

    for i in range(0,len(temp_right)):
        temp_left.append(temp_right[i])

    return temp_left


def sayIt():
    global the_same
    global response

    print("Voice started in BG")
    while True:
        with lock:
            if the_same==False and the_same!=None:
                try:
                    audio_stream = io.BytesIO(response.audio_content)
                    audio = AudioSegment.from_file(audio_stream, format="mp3")
                    play(audio)
                except:
                    pass
    

def main_detection():


    global response
    global prev_result
    prev_result = ""
    global the_same 
    the_same = None
    
    #print(type(response))


    #t1 = threading.Thread(target=sayIt)
    #t1.start()

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'textspeech-436005-bbe68d981d0b.json'
    client = texttospeech.TextToSpeechClient()   

    #synthesis_input = texttospeech.SynthesisInput(text="أ")


    voice = texttospeech.VoiceSelectionParams(
        language_code="ar-XA", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )




    # Copying mediapipe instances of hand skeleton drawing and hand detection model instances to custom variables
    mp_drawing = mp.solutions.drawing_utils
    #mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # initializing video capturing device 'default webcam'
    cap = cv2.VideoCapture(0)

    # To store number of images taken for each gesture category. Will be incremented dynamically by 1 when a record is saved
    sign_count = 0

    # Initializing hand detection instance with custom parameters
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7,max_num_hands=2) as hands:
    
    
        file1 = open('myfile.csv','a')
    
    
        model_signs = pickle.load(open('model/finalized_model_hyp_onlysigns.sav', 'rb'))
        model_numbers = pickle.load(open('model/numbers_model_iter2.sav', 'rb'))

        # While loop will run as long as the camera is in a functional state
        while cap.isOpened():

            #global response
            response = None

            pre_processed_landmark_list = ""

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

            Dict = {"Alif": 'أ', "Ba": 'ب', "Ta": 'ت'}

            # Rest of the code will continue only if the results contains landmarks of a hand or hands.
            # If not that means no hands are present in the frame.
            if results.multi_hand_landmarks is not None:
                if len(results.multi_handedness) == 1: 
                    #print("=========  1 Hand  ========")

                    # Iterate through each landmark detected in the hand
                    for hand_landmarks in results.multi_hand_landmarks:
                        hand = results.multi_hand_landmarks[0]

                        landmark_list = calc_landmark_list(debug_image,hand)
                        
                        pre_processed_landmark_list = pre_process_landmark(landmark_list)
                        
                        sign_count += 1
                        
                        probabilities = model_signs.predict_proba([pre_processed_landmark_list]) #This array can be used to print the probability of how sure the model is about the sign detected
                        #print(probabilities.size())
                        result = model_signs.predict([pre_processed_landmark_list])

                        msg = str(result)+": "+ str(probabilities.max())

                        image = cv2.putText(image, str(result), (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2, cv2.LINE_AA)

                        #print(type(str(result[0])))
                        if results != None and results != prev_result:
                            synthesis_input = texttospeech.SynthesisInput(text=result[0])
                            
                            if prev_result==result:
                                the_same = True
                            else:
                                the_same = False
                                with lock:
                                    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

                                print("New API response stored")
                                
                            prev_result = result



                    # Drawing skeleton on the frame
                    for hand_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS)
                            #mp_drawing_styles.get_default_hand_landmarks_style(),
                            #mp_drawing_styles.get_default_hand_connections_style())
                        
                elif len(results.multi_handedness) == 2: 
                    #print("========= 2 Hands ========")
                    # Iterate through each landmark detected in the hand
                    for hand_landmarks in results.multi_hand_landmarks:
                        left_hand = results.multi_hand_landmarks[0]
                        right_hand = results.multi_hand_landmarks[1]

                        left_landmark_list = calc_landmark_list(debug_image, left_hand)
                        right_landmark_list = calc_landmark_list(debug_image, right_hand)

                        left_pre_processed_landmark_list = pre_process_landmark(left_landmark_list)
                        right_pre_processed_landmark_list = pre_process_landmark(right_landmark_list)
                        
                        sign_count += 1

                        pre_processed_landmark_list = get_current_landmarks_both_hands(left_pre_processed_landmark_list,right_pre_processed_landmark_list)
                        
                        probabilities = model_numbers.predict_proba([pre_processed_landmark_list]) #This array can be used to print the probability of how sure the model is about the sign detected
                        #print(probabilities.size())
                        result = model_numbers.predict([pre_processed_landmark_list])

                        msg = str(result)+": "+ str(probabilities.max())

                        image = cv2.putText(image, str(result), (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                    2, cv2.LINE_AA)


                    # Drawing skeleton on the frame
                    for hand_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS)
                            #mp_drawing_styles.get_default_hand_landmarks_style(),
                            #mp_drawing_styles.get_default_hand_connections_style())
                        
            
            # Displaying frame with the landmark skeleton overlay
            cv2.imshow('MediaPipe Hands', image)

            # Exits the main loop if 'Esc' pressed
            if cv2.waitKey(1) & 0xff == ord('q'):
                break

    # Releasing handle to camera before terminating the programme
    cap.release()


if __name__ =="__main__":
    t1 = threading.Thread(target=main_detection, args=())
    t2 = threading.Thread(target=sayIt, args=())

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print("Done!")