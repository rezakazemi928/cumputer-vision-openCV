import cv2


def background_subtractor_gmg(addr):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cap = cv2.VideoCapture(addr)
    subtractor = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=120, decisionThreshold=0.9)

    while True:
        ret, frame = cap.read()

        if ret == True:
            mask = subtractor.apply(frame)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            cv2.imshow('frame', mask)

            if cv2.waitKey(25) & 0xFF == 27:
                break
        else:
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    background_subtractor_gmg('cars.mp4')