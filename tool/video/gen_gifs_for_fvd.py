import glob
import math
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
import numpy as np
import imageio
import threading
import os
import sys
pythonpath = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print(pythonpath)
sys.path.insert(0, pythonpath)

class Preprocess(threading.Thread):
    def __init__(self, data, begin_idx, end_idx, gif_root, frames_len, fps, use_tqdm=False, overwrite=False):
        threading.Thread.__init__(self)
        self.data = data
        self.begin_idx = begin_idx
        self.end_idx = end_idx
        self.use_tqdm = use_tqdm
        self.gif_root = gif_root
        self.frames_len = frames_len
        self.fps = fps
        self.end_idx =  self.end_idx - self.frames_len
        self.overwrite = overwrite

    def run(self):
        vid = "_".join(os.path.splitext(os.path.basename(self.data[0]))[0].split("_")[:2])
        it = tqdm(range(self.begin_idx, self.end_idx), desc=f"{vid}") if self.use_tqdm else range(self.begin_idx, self.end_idx)
        for idx in it:
            file = self.data[idx]
            image = imageio.imread(file)
            base_name = os.path.splitext(os.path.basename(file))[0]
            if os.path.exists(os.path.join(self.gif_root, f'{base_name}.gif')) and not self.overwrite:
                continue
            images = [image]
            for i in range(1, self.frames_len):
                try:
                    file = self.data[idx+i]
                    image = imageio.imread(file)
                except Exception as e:
                    print(e, f"curr_idx:{idx+i}", f"begin_idx:{self.begin_idx}", f"end_idx:{self.end_idx}", f"data_len: {len(self.data)}")
                images.append(image)
            imageio.mimsave(os.path.join(self.gif_root, f'{base_name}.gif'), images, fps=self.fps)


def compose_gif(img_path, gif_frames, fps, num_workers, overwrite=False):
    if img_path[-1] == "/":
        img_path = img_path[:-1]
    base = os.path.basename(img_path) + "_gif"
    root = os.path.dirname(img_path)
    gif_root = os.path.join(root, base)
    if not os.path.exists(gif_root):
        os.mkdir(gif_root)
    inst_names = glob.glob(f"{img_path}/*.png")
    inst_names = sorted(inst_names)
    print(inst_names)
    exit()
    
    vid2files = defaultdict(list)
    for path in inst_names:
        vid = "_".join(os.path.splitext(os.path.basename(path))[0].split("_")[:2])
        vid2files[vid].append(path)

    for vid, inst_paths in vid2files.items():
        pool = []
        per_num = math.ceil(len(inst_paths) / num_workers)
        cur_idx = 0
        for idx in range(num_workers):
            begin_idx = cur_idx
            end_idx = cur_idx + per_num
            if end_idx > len(inst_paths):
                end_idx = len(inst_paths)
            pool.append(
                Preprocess(
                    inst_paths, begin_idx, end_idx, gif_root,
                    gif_frames, fps, idx==0, overwrite=overwrite))
            cur_idx = end_idx

        for idx in range(num_workers):
            pool[idx].start()

        for idx in range(num_workers):
            pool[idx].join()
            
    return gif_root


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    #parser.add_argument('--root_dir', type=str, default="/home/t-mni/blob/chenfeipremiumblock/data/G")
    parser.add_argument('--root_dir', type=str, default="/home/siddiqui/DisCo")
    parser.add_argument('--path_gen', type=str, default="/home/siddiqui/DisCo/exp/ntu60-CV/eval_step_31250/pred_gs7.5_scale-cond1.0-ref1.0/")
    parser.add_argument('--path_gt', type=str, default="/home/siddiqui/DisCo/exp/ntu60-CV/eval_step_31250/gt/")
    parser.add_argument('--gif_frames', type=int, default=16)
    parser.add_argument('--gif_fps', type=float, default=3)
    parser.add_argument('--overwrite', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()
    print(len(os.listdir("/home/siddiqui/DisCo/exp/ntu60-CV/eval_step_31250/gt/")))
    exit()
    if args.root_dir not in args.path_gen:
        args.path_gen = os.path.join(args.root_dir, args.path_gen)
    if args.root_dir not in args.path_gt:
        args.path_gt = os.path.join(args.root_dir, args.path_gt)
    print(f"compose_gif for pred")
    path_gen_gif = compose_gif(args.path_gen, args.gif_frames,
        args.gif_fps, args.num_workers, overwrite=args.overwrite)
    print(f"compose_gif for gt")
    path_gt_gif = compose_gif(args.path_gt, args.gif_frames,
        args.gif_fps, args.num_workers, overwrite=args.overwrite)
    print(path_gen_gif, path_gt_gif)