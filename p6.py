import cv2
import numpy as np

count_frame = cv2.CAP_PROP_FRAME_COUNT
number_of_frame = np.random.uniform(size=25)

def get_background(addr):
    cap = cv2.VideoCapture(addr)
    frames_id = cap.get(count_frame) * number_of_frame
    frames_list = []

    while True:

        for frame_id in frames_id:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            frames_list.append(frame)

        median_frame = np.median(frames_list, axis=0).astype(dtype=np.uint8)
        cv2.imshow('frame', median_frame)

        if cv2.waitKey(25) & 0xFF == 27:
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    get_background('cars.mp4')