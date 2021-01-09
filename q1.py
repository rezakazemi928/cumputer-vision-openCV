import cv2
import numpy as np

color = np.random.randint(0 , 255 , (100 , 3))


def optical_points_LK_SH(addr, feature_parameter, lk_parametes):
    cap = cv2.VideoCapture(addr)
    ret, previous_frame = cap.read()
    previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    point0 = cv2.goodFeaturesToTrack(previous_frame_gray, mask=None, **feature_parameter)
    mask = np.zeros_like(previous_frame)

    while True:
        if ret == True:
            ret, frame = cap.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            point1, st, err = cv2.calcOpticalFlowPyrLK(previous_frame_gray, frame_gray, point0, None, **lk_parametes)
            new_point = point1[st == 1]
            old_point = point0[st == 1]

            for index, (new, old) in enumerate(zip(new_point, old_point)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (a, b), (c, d), color[index].tolist(), 2)
                frame = cv2.circle(frame, (a, b), 5, color[index].tolist(), -1)

            img = cv2.add(frame, mask)
            cv2.imshow('frame', img)

            if cv2.waitKey(25) & 0xFF == 27:
                break
            else:
                previous_frame_gray = frame_gray.copy()
                point0 = new_point.reshape(-1, 1, 2)
        else:
            break

    cv2.destroyAllWindows()
    cap.release()

def optical_points_LK_SH_version_2(addr, feature_parameter, lk_parametes):
    cap = cv2.VideoCapture(addr)

    while True:
        ret, previous_frame = cap.read()
        previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        point0 = cv2.goodFeaturesToTrack(previous_frame_gray, mask=None, **feature_parameter)

        if ret == True:
            mask = np.zeros_like(previous_frame)
            ret, frame = cap.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            point1, st, err = cv2.calcOpticalFlowPyrLK(previous_frame_gray, frame_gray, point0, None, **lk_parametes)
            new_point = point1[st == 1]
            old_point = point0[st == 1]

            for index, (new, old) in enumerate(zip(new_point, old_point)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (a, b), (c, d), color[index].tolist(), 2)
                frame = cv2.circle(frame, (a, b), 5, color[index].tolist(), -1)

            img = cv2.add(frame, mask)
            cv2.imshow('frame', img)

            if cv2.waitKey(25) & 0xFF == 27:
                break
            else:
                previous_frame_gray = frame_gray.copy()
                point0 = new_point.reshape(-1, 1, 2)
        else:
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=15)

    lk_params = dict(winSize=(10, 10),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    optical_points_LK_SH('cars.mp4', feature_params, lk_params)
    optical_points_LK_SH_version_2('cars.mp4' , feature_params , lk_params)