import cv2
import os
from natsort import natsorted

image_folder = "/home/siddiqui/DisCo/exp/pretrained-ft-continue2/eval_step_79999/pred_gs7.5_scale-cond1.0-ref1.0/"
images = [img for img in os.listdir(image_folder)]

for video in os.listdir('/home/c3-0/datasets/NTU_RGBD_120/nturgb+d_rgb/'):
    # if int(video[5:8]) != 1 and int(video[1:4]) == 1:
    video = video[:20]
    vid_images = [img for img in images if video in img]
    print(video, len(vid_images))
    if len(vid_images) > 5:
        vid_images = natsorted(vid_images)
        print(video, len(vid_images))
        frame = cv2.imread(os.path.join(image_folder, vid_images[0]))
        height, width, layers = frame.shape
        video = cv2.VideoWriter(f'{video}.avi', 0, 10, (width,height))


        for img in vid_images:
            print(img)
            video.write(cv2.imread(os.path.join(image_folder, img)))

        video.release()

