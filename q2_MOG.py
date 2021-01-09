import cv2


def background_sub_mog(addr):
    cap = cv2.VideoCapture(addr)
    subtractor = cv2.bgsegm.createBackgroundSubtractorMOG(history=200, nmixtures=5, backgroundRatio=0.7, noiseSigma=0)

    while True:
        ret, frame = cap.read()
        if ret == True:
            mask = subtractor.apply(frame)
            cv2.imshow('Frame', mask)

            if cv2.waitKey(20) & 0xFF == 27:
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    background_sub_mog('cars.mp4')