import cv2 as cv
import ultralytics
import math
import numpy as np
import win32gui, win32ui, win32con, win32api
import time
from ultralytics.yolo.utils.plotting import Annotator

# The following is the machine learning aimbot project by Kallum Doughty 25084869
# This code is a modified version of the code by Ben Johnson, 2020 found at https://learncodebygaming.com/blog/fast-window-capture
# The code is used to capture the game window making use of the openCV library and the win32 library
# The code has been modified to work with the ultralytics yolov8 object detection model
# The code that was used will be highlighted accordingly where it is present

window_name = "Counter-Strike: Global Offensive - Direct3D 9"

model = ultralytics.YOLO("best.pt", "v8") # Load the weights in the file 'best.pt' and use the yolov8 model

# Disable opencv logging
cv.setLogLevel(0)

# The following code is from Ben Johnson, 2020 found at https://learncodebygaming.com/blog/fast-window-capture
# Capture the video data from the selected application window
class WindowCapture:

    # properties
    w = 0
    h = 0
    hwnd = None
    cropped_x = 0
    cropped_y = 0
    offset_x = 0
    offset_y = 0

    # constructor
    def __init__(self, window_name):
        # find the handle for the window we want to capture
        self.hwnd = win32gui.FindWindow(None, window_name)
        if not self.hwnd:
            raise Exception('Window not found: {}'.format(window_name))

        # get the window size
        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.w = window_rect[2] - window_rect[0]
        self.h = window_rect[3] - window_rect[1]

        # account for the window border and titlebar and cut them off
        border_pixels = 8
        titlebar_pixels = 30
        self.w = self.w - (border_pixels * 2)
        self.h = self.h - titlebar_pixels - border_pixels
        self.cropped_x = border_pixels
        self.cropped_y = titlebar_pixels

        # set the cropped coordinates offset so we can translate screenshot
        # images into actual screen positions
        self.offset_x = window_rect[0] + self.cropped_x
        self.offset_y = window_rect[1] + self.cropped_y

    def get_screenshot(self):

        # get the window image data
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (self.w, self.h), dcObj, (self.cropped_x, self.cropped_y), win32con.SRCCOPY)

        # convert the raw data into a format opencv can read
        #dataBitMap.SaveBitmapFile(cDC, 'debug.bmp')
        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (self.h, self.w, 4)

        # free resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        # drop the alpha channel, or cv.matchTemplate() will throw an error like:
        #   error: (-215:Assertion failed) (depth == CV_8U || depth == CV_32F) && type == _templ.type() 
        #   && _img.dims() <= 2 in function 'cv::matchTemplate'
        img = img[...,:3]

        # make image C_CONTIGUOUS to avoid errors that look like:
        #   File ... in draw_rectangles
        #   TypeError: an integer is required (got type tuple)
        # see the discussion here:
        # https://github.com/opencv/opencv/issues/14866#issuecomment-580207109
        img = np.ascontiguousarray(img)

        return img
# End of code from Ben Johnson, 2020 found at https://learncodebygaming.com/blog/fast-window-capture

# Calculate the closest enemy to the center of the window
def closest_enemy_coords(boxes, wincap):
    # Get the coordinates of the enemy closest to the center of the window
    closest_pos = None
    closest_dist = None
    for box in boxes:
        # Get the coordinates of the enemy
        b = box.xyxy[0]
        # Get the distance between the enemy and the center of the window
        dist = math.sqrt((b[0] - wincap.w / 2) ** 2 + (b[1] - wincap.h / 2) ** 2)
        # If the enemy is closer than the previous closest enemy, set the closest enemy to the current enemy
        if closest_dist is None or dist < closest_dist:
            # Aim at the CENTER of the box
            center = (b[0] + (b[2] - b[0]) / 2, b[1] + (b[3] - b[1]) / 2)

            # Head is at the top center of the box
            # Instead of using the direct top we want about 90% of the height
            head = (b[0] + (b[2] - b[0]) / 2, b[1] + (b[3] - b[1]) * 0.1)
            closest_pos = head
            closest_dist = dist
    return closest_pos


wincap = WindowCapture(window_name) # Using Ben Johnson's WindowCapture class to capture the window
loop_time = time.time()
while(True):

    # get an updated image of the game
    screenshot = wincap.get_screenshot()

    # detect objects in the image
    detected = model.predict(screenshot, imgsz = 480) # Using the YOLOv8 model to detect enemies - imgsz must be a multiple of 32

    # loop through the detections and draw them on the image
    for detection in detected:
        annotator = Annotator(screenshot)
        
        boxes = detection.boxes

        for box in boxes:
            b = box.xyxy[0]
            c = box.cls
            annotator.box_label(b, model.names[int(c)], color = (0, 0, 255))


    screenshot = annotator.result()

    closest_pos = None

    # Closest enemy to the centre of the window
    if boxes is not None:
        closest_pos = closest_enemy_coords(boxes, wincap)

    # Calculate the amount of pixels to move in the X and Y axis to aim at the closest enemy
    if closest_pos is not None:
        # Center of the window in pixels
        centre_x = 800 / 2.0
        centre_y = 600 / 2.0

        # Get the distance between the center of the window and the enemy in pixels
        distance_x = closest_pos[0] - centre_x
        distance_y = closest_pos[1] - centre_y

        # Calculate the new mouse position
        scale_x = 1
        scale_y = 1

        aim_x = float( distance_x * scale_x )
        aim_y = float( distance_y * scale_y )

        print( "{}, {}", aim_x, aim_y )
        
        ## Draw pos text on screen
        cv.putText(screenshot, 'X: ' + str(aim_x), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv.putText(screenshot, 'Y: ' + str(aim_y), (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Move the mouse to aim at the closest enemy
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(aim_x), int(aim_y))
    
# Another instance of code by Ben Johnson, 2020 found at https://learncodebygaming.com/blog/fast-window-capture

    cv.imshow('Machine Learning Aimbot', screenshot)

    # debug the loop rate
    #print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time.time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break
# End of code by Ben Johnson, 2020 found at https://learncodebygaming.com/blog/fast-window-capture