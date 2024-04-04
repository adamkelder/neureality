import os
import time
import cv2
import numpy as np
import pandas as pd
from pytorchyolo import detect, models


names_path = 'config/coco.names'
with open(names_path) as f:
    names = f.read().splitlines()
img_dir = 'images'


def func_profile(func, *args, **kwargs):
    t0 = time.time()
    output = func(*args, **kwargs)
    dt = (time.time() - t0) * 1000  # ms
    return output, dt


def print_profiles(profiles):
    op_names = [x[0] for x in profiles[0]]
    data = []
    for i, op_name in enumerate(op_names):  # assuming ops are always in the same order
        values = [profile[i][1] for profile in profiles]
        stats = {
            'op_name': op_name,
            'mean': np.mean(values),
            'std': np.mean(values),
            'min': np.min(values),
            'max': np.max(values),
        }
        data.append(stats)
    df = pd.DataFrame(data)
    print(df)


def draw_boxes(bgr, boxes):
    for box in boxes:
        pt1 = tuple(int(round(x)) for x in box[0:2])
        pt2 = tuple(int(round(x)) for x in box[2:4])
        cv2.rectangle(bgr, pt1, pt2, (0, 255, 0), thickness=2)
        cv2.putText(bgr, f'{names[int(box[5])]} {box[4]:.2f}', pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                    thickness=2)


def main():
    # Load the YOLO model
    model = models.load_model(
        "config/yolov3-tiny.cfg",
        "weights/yolov3-tiny.weights")

    profiles = []
    img_list = [os.path.join(img_dir, p) for p in sorted(os.listdir(img_dir)) if p.endswith('.jpg')]
    for img_path in img_list:
        profile = []
        model.profile = []  # reset model profile

        # preprocessing
        # Load the image as a numpy array
        bgr, dt = func_profile(cv2.imread, img_path)
        profile.append(['preproc_imread', dt])
        # resize
        bgr, dt = func_profile(cv2.resize, bgr, dsize=(416, 416))
        profile.append(['preproc_resize', dt])
        # Convert OpenCV bgr to rgb
        img, dt = func_profile(cv2.cvtColor, bgr, cv2.COLOR_BGR2RGB)
        profile.append(['preproc_cvtColor', dt])

        # Runs the YOLO model on the image
        profile_model, boxes = detect.detect_image(model, img)
        profile += profile_model
        # postprocessing
        # draw boxes
        _, dt = func_profile(draw_boxes, bgr, boxes)
        profile.append(['postproc_draw_boxes', dt])

        profiles.append(profile)

        # cv2.imshow('win', bgr)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
    print_profiles(profiles)


if __name__ == '__main__':
    main()
