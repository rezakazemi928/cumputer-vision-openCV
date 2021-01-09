import cv2


def background_sub_mog2(addr):
    cap = cv2.VideoCapture(addr)
    subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=255, detectShadows=True)

    while True:
        ret, frame = cap.read()
        mask = subtractor.apply(frame)

        if ret == True:
            bgr_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            out_frame = cv2.bitwise_or(bgr_mask, frame)
            cv2.imshow('mask', out_frame)
            key = cv2.waitKey(25) & 0xFF

            if key == 27:
                break
        else:
            break

    cv2.destroyAllWindows()
    cap.release()


def background_sub_mog(addr):
    cap = cv2.VideoCapture(addr)
    subtractor = cv2.bgsegm.createBackgroundSubtractorMOG(history=200, nmixtures=25, backgroundRatio=0.7, noiseSigma=0)

    while True:
        ret, frame = cap.read()
        if ret == True:
            mask = subtractor.apply(frame)
            bgr_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            out_frame = cv2.bitwise_or(frame, bgr_mask)
            cv2.imshow('Frame', out_frame)

            if cv2.waitKey(20) & 0xFF == 27:
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def background_sub_gmg(addr):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cap = cv2.VideoCapture(addr)
    subtractor = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=120, decisionThreshold=0.9)

    while True:
        ret, frame = cap.read()

        if ret == True:
            mask = subtractor.apply(frame)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            bgr_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            out_frame = cv2.bitwise_or(frame, bgr_mask)
            cv2.imshow('frame', out_frame)

            if cv2.waitKey(25) & 0xFF == 27:
                break
        else:
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    while True:
        menu = """
        1 --> MOG
        2 --> MOG2
        3 --> GMG
        Any num to Exit.
        ----------------
        """
        print(menu)
        choice = int(input('which sub_method you want to choose: '))

        if choice == 1:
            background_sub_mog('cars.mp4')

        elif choice == 2:
            background_sub_mog2('cars.mp4')

        elif choice == 3:
            background_sub_gmg('cars.mp4')

        else:
            break