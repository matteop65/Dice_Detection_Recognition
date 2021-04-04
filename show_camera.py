import cv2

img = cv2.VideoCapture(0)


# def camera_frame():
while True:
    _, frame = img.read()
    cv2.imshow("image",frame)
    if cv2.waitKey(1) & 0xFF == 'q':
        img.release()
        cv2.destroyAllWindows()
        break