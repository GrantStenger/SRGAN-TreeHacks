# Import Dependencies
import numpy as np
import pandas as pd
import cv2
import os
import argparse
from threading import Thread, Event
import time
from pytube import YouTube


def download_video(yt, res, name='vid'):
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

    try:
        #object creation using YouTube which was imported in the beginning
        yt = YouTube(link)
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


def download_and_frame(yt, id_val, target_width, target_height, 
                        timeout=10, N_FRAMES_ITER=15, pos=None):

    # Download video for 240 px
    name = 'temp'
    outfile = 'data/'+name+'.mp4'
    successful = True 

    success = run_func_with_timeout(download_video, (yt, id_val, name), timeout=timeout)
    if not success:
        return []

    # Get 5 frames for OpenCV
    cap = cv2.VideoCapture(outfile)
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    width = cap.get(3)   # float
    height = cap.get(4) # float

    # check if within 
    if abs(height - target_height) > 2 or abs(width - target_width) > 2:
        print(width, height, "no bueno")
        return successful, [] 
    
    # Get positions
    if pos is None:
        pos = (np.random.rand(N_FRAMES_ITER)*totalFrames-3).astype(np.int) + 1
        pos.sort()

    imgs, success = get_frames(cap, pos)
    del cap

    if not success:
        return []
    
    return imgs, pos

           

def main():

    root = 'https://www.youtube.com/watch?v='

    urls = pd.read_csv('data/links.csv', header=None)[0].apply(lambda x: x.replace("'", '') )
    urls = urls.tolist()[start_pos:]
    
    YDIR = outdir+'/240px/'
    Y3DIR = outdir+'/480px/'
   
    # Make sure dirs exist
    os.makedirs(outdir, exist_ok=True)
    os.makedirs('data/', exist_ok=True)
    os.makedirs(YDIR, exist_ok=True)
    os.makedirs(Y3DIR, exist_ok=True)

    img_count = len(os.listdir(YDIR))

    for nvids, url in enumerate(urls):

        try:
            link = root + url
            yt = get_yt(link)

            imgs240, pos = download_and_frame(yt=yt, id_val='240p', target_height=240, 
                                            target_width=426)
            
            if len(imgs240) == 0:
                print("COINTOINE")
                continue 

            imgs480, pos = download_and_frame(yt=yt, id_val='480p', target_height=480, 
                                            target_width=854, pos=pos)

            print(len(imgs480))
            if len(imgs480) == 0:
                print("COINTINE")
                continue

            # Save images
            for i in range(len(imgs240)):
                cv2.imwrite(YDIR+str(img_count)+'.png', imgs240[i])
                cv2.imwrite(Y3DIR+str(img_count)+'.png', imgs480[i])
                img_count+=1

            print("SUCCESS: ", nvids)
            del yt

        except:
            continue


if __name__ == "__main__":

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

    main()
