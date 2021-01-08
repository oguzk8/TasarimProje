from imageai.Detection import VideoObjectDetection
import cv2

"""
-----VERSIONS-----
tensorflow==1.13.1
keras==2.2.4
imageai==2.1.5
h5py==2.10.0
"""

FOCAL_LENGTH = 160.54
KNOWN_WIDTH = 75
distance = 100

camera = cv2.VideoCapture(0)

detector = VideoObjectDetection()

model_path = "models/yolo.h5"
input_path = "input/car1.jpg"
output_path = "output/newimage.jpg"

detector.setModelTypeAsYOLOv3()
detector.setModelPath(model_path)
detector.loadModel()
# detection = detector.detectObjectsFromImage(input_path, output_path)


def distance_to_camera(imgWidth):
    return KNOWN_WIDTH * FOCAL_LENGTH / imgWidth


def forFrame(frame_number, output_array, output_count, returned_frame):
    print("FOR FRAME ", frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
    print("------------END OF A FRAME --------------")
    #to do: go through all of the detected cars and find the closest one
    global distance
    if len(output_array) != 0:
        distance = distance_to_camera((output_array[0]["box_points"][2] - output_array[0]["box_points"][0])*0.2645833333) #temporary
    print(distance)
    cv2.putText(returned_frame, str(distance), (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("image", returned_frame)
    cv2.waitKey(50)

"""
#this code is for general detection for testing
video_path = detector.detectObjectsFromVideo(camera_input=camera,
    output_file_path="camera_detected_video", frames_per_second=20, log_progress=True, minimum_percentage_probability=30,
                                             per_frame_function=forFrame, save_detected_video=False, return_detected_frame=True)
"""


#actual detection only for cars
video_path = detector.detectCustomObjectsFromVideo(custom_objects=detector.CustomObjects(car=True), camera_input=camera,
    output_file_path="camera_detected_video", frames_per_second=20, log_progress=True, minimum_percentage_probability=30,
                                             per_frame_function=forFrame, save_detected_video=False, return_detected_frame=True)
