import cv2
import numpy as np


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Could not open camera.')
        return

    subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)
    count = 0
    line_position = 200  # vertical position of counting line

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = subtractor.apply(frame)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            centroid_y = y + h // 2
            if centroid_y < line_position < centroid_y + 5:
                count += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (255, 0, 0), 2)
        cv2.putText(frame, f'Count: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2)

        cv2.imshow('Counter', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
