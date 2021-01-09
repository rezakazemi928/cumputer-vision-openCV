import cv2


def background_sub_mog2(addr):
    cap = cv2.VideoCapture(addr)
    subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=255, detectShadows=True)

    while True:
        ret, frame = cap.read()
        mask = subtractor.apply(frame)

        if ret == True:
            cv2.imshow('mask', mask)
            key = cv2.waitKey(25) & 0xFF

            if key == 27:
                break
        else:
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    background_sub_mog2('cars.mp4')