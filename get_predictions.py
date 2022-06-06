import cv2
import numpy as np
import pyautogui
import imutils
import cv2
import time

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
thickness              = 1
lineType               = 2
cnt = 0
model_name = "mobilenet_v2"

fire_detection_fire_no_fire_winter = tf.keras.models.load_model(f'./3_models_series/mobilenet_v2/{model_name}_2_fire_or_no_fire_winter.h5')
fire_detection_fire_neutral_smoke = tf.keras.models.load_model(f'./3_models_series/mobilenet_v2/{model_name}_3_fire_smoke_neutral.h5')
fire_detection_fire_no_fire = tf.keras.models.load_model(f'./3_models_series/mobilenet_v2/{model_name}_2_fire_or_no_fire.h5')

while True:
    cnt = cnt + 1
    time.sleep(1)
    print("started_new_frame")
    image = pyautogui.screenshot()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    print("got the screenshot")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('my_patch.png',0)

    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc= cv2.minMaxLoc(result)

    height, width = template.shape[:2]
    height_resize, width_resize = image.shape[:2]

    top_left = max_loc
    bottom_right = (top_left[0] + width, top_left[1] + height)
    cv2.rectangle(image, top_left, bottom_right, (0,0,255),5)

    top_left_to_crop = (int(top_left[0] - width*2), top_left[1] + height*2)

    bottom_right_to_crop = (int(top_left[0] + width*0.7), top_left[1] + height*9)
    cv2.rectangle(image, top_left_to_crop, bottom_right_to_crop, (0,255,255),5)
    
    try:
        crop_img = image[top_left[1] + height*2:top_left[1] + height*9, int(top_left[0] - width*2):int(top_left[0] + width*0.7)]
        img_rgb = cv2.resize(image, (int(width_resize/2), int(height_resize/2)))

        #cv2.imwrite(f"img_rgb_{cnt}.png", img_rgb)
        #cv2.imwrite(f"crop_img_{cnt}.png", crop_img)
        
        image =cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB)
        image =cv2.resize(image,(224,224))
        
        pred_fire_detection_fire_no_fire_winter.predict(image)
        pred_fire_detection_fire_neutral_smoke.predict(image)
        pred_fire_detection_fire_no_fire.predict(image)
        
        text_prediction_results = f"pred_fire_detection_fire_no_fire_winter = {pred_fire_detection_fire_no_fire_winter}," 
                                  f"pred_fire_detection_fire_neutral_smoke ={pred_fire_detection_fire_neutral_smoke}"
                                  f"pred_fire_detection_fire_no_fire= {pred_fire_detection_fire_no_fire}"
                                  
        cv2.putText(crop_img,text_prediction_results, 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
        
        cv2.imwrite(f"crop_img_{cnt}.png", crop_img)
   
    except Exception as e:
        print(e)



