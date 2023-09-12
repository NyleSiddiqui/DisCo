import cv2
import os
from natsort import natsorted

# image_folder = '/home/siddiqui/DisCo/exp/ntu60-perframe-20eps/eval_step_1107/pred_gs7.5_scale-cond1.0-ref1.0'
image_folder = '/home/kzhai/DisCo/exp/ntu60-fixpose/eval_step_6999/pred_gs7.5_scale-cond1.0-ref1.0'
images = [img for img in os.listdir(image_folder)]

for video in os.listdir('/home/c3-0/datasets/NTU_RGBD_120/nturgb+d_rgb/'):
    # if int(video[5:8]) != 1 and int(video[1:4]) == 1:
    video = video[:20]
    vid_images = [img for img in images if video in img]
    if len(vid_images) > 5:
        vid_images = natsorted(vid_images)
        frame = cv2.imread(os.path.join(image_folder, vid_images[0]))
        height, width, layers = frame.shape
        video = cv2.VideoWriter(f'{video}.avi', 0, 10, (width,height))

        for img in vid_images:
            video.write(cv2.imread(os.path.join(image_folder, img)))

        video.release()
        exit()

