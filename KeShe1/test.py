import numpy as np
import cv2
import time

r, g, b = 255, 255, 255

x = np.array([[[r, g, b] for _ in range(1080)] for _ in range(720)]).astype(np.uint8)

print(x)


while cv2.waitKey(1) & 0xFF != 27:
    cv2.imshow("?", cv2.transpose(x))
    time.sleep(0.1)

cv2.destroyAllWindows()
