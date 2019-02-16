import numpy as np
import pandas as pd
import cv2
import os
import argparse

from threading import Thread, Event
import time

from pytube import YouTube


parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='data/',
                    help='output dir to place images')
parser.add_argument('--start_pos', default=0, type=int,
                    help='start position')
parser.add_argument('--timeout', default=7, type=int,
                    help='max time to load')
args = parser.parse_args()


outdir = args.outdir 
start_pos = args.start_pos
timeout = args.timeout


def download_video(yt, res, outdir, name='vid'):
    """
    yt: youtube object
    res: resolution; eg '144p'
    outdir: output directory
    name: name of video
    """
    
    try:
        # filter by resolution and to mp4
        stream = yt.streams.filter(res=res, mime_type='video/mp4').all()[0]

        # download the stream
        stream.download(output_path='data/', filename=name)
    except:
        print("Connection Error Download Vid", res)
        return False

    return True
    
def get_frames(cap, pos):
    """
    cap: cv2.VideoCapture
    pos: array of frame positions
    
    Returns:
    frames: list of np arrays
    success: bool, false if fail
    """
    
    frames = []
    success = True
    
    # Save frames in good positions
    for i in pos:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            success = False
            
    return frames, success

def get_yt(link):
    
    print(link)
    try: 
        #object creation using YouTube which was imported in the beginning 
        yt = YouTube(link, ) 
    except: 
        print("YT Error") 
        return None

    return yt

def run_func_with_timeout(func, args, timeout=5):
    
    stop_it = Event()
    
    # Create a thread that needs to run for 5 seconds
    stuff_doing_thread = Thread(target=func, args=args)
 
    stuff_doing_thread.start()
    stuff_doing_thread.join(timeout=timeout)
    
    res = stuff_doing_thread.isAlive()
    stop_it.set()
    
    return not res

root = 'https://www.youtube.com/watch?v='

urls = pd.read_csv('data/links.csv', header=None)[0].apply(lambda x: x.replace("'", '') )
urls = urls.tolist()[start_pos:]


XDIR = 'data/144px/'
YDIR = 'data/240px/'

# Make sure dirs exist
os.makedirs(outdir, exist_ok=True)
os.makedirs(XDIR, exist_ok=True)
os.makedirs(YDIR, exist_ok=True)

img_count = len(os.listdir(XDIR))
N_FRAMES_ITER = 11


for nvids, url in enumerate(urls):

    try:
    
        link = root + url
        
        yt = get_yt(link)
        
        if yt is None:
            continue

        # Download video for 144 px
        name = 'temp'
        outfile = outdir+name+'.mp4'
        
        success = run_func_with_timeout(download_video, (yt, '144p', outdir, name), timeout=timeout)
        if not success:
            continue 

        # Get 5 frames for OpenCV
        cap = cv2.VideoCapture(outfile)
        totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        # Get positions
        pos = (np.random.rand(N_FRAMES_ITER)*totalFrames-3).astype(np.int) + 1
        pos.sort()

        imgs144, success = get_frames(cap, pos)
        del cap
        if not success:
            continue
        
        # Download 240p video
        success = run_func_with_timeout(download_video, (yt, '240p', outdir, name), timeout=timeout )
        if not success:
            continue 
        
        # Ditto for 240p
        cap = cv2.VideoCapture(outfile)
        imgs240, sucess = get_frames(cap, pos)
        del cap
        if not success:
            continue
        
        # Save images
        for i in range(len(imgs144)):
            cv2.imwrite(XDIR+str(img_count)+'.png', imgs144[i])
            cv2.imwrite(YDIR+str(img_count)+'.png', imgs240[i])
            
            img_count+=1
        
        print("SUCCESS: ", nvids)
        del yt

    except:
        continue
    
    
