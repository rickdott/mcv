import cv2

cam = cv2.VideoCapture(0)
cv2.namedWindow("preview")
images = 0

# Read webcam frames until escape is pressed, saving frame when spacebar is pressed
while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed")
        break
    cv2.imshow("preview", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # Escape
        print("Quitting...")
        break
    elif k%256 == 32:
        # Spacebar
        fname = f'img_{images}.png'
        cv2.imwrite(fname, frame)
        print(f'{images + 1} saved.')
        images += 1

# Clean up
cam.release()
cv2.destroyAllWindows()