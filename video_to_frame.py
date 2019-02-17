from pytube import YouTube
import argparse
import cv2
import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# REPLICATE ARGUMENT PARSING IN MAIN #
parser = argparse.ArgumentParser()
parser.add_argument('--video_ext', default='dQw4w9WgXcQ',
                    help='extension in YT url after v=')
parser.add_argument('--res', default='240p',
                    help='desired download resolution')
parser.add_argument('--video_name', default='test_vid',
                    help='name of video')
parser.add_argument('--length', default=60, type=int, help='trimmed length of video')
args = parser.parse_args()

YT_ROOT = 'https://www.youtube.com/watch?v='
VIDEO_URL = args.video_ext
RES = args.res
VIDEO_NAME = args.video_name
LENGTH = args.length


# Function that saves array of frames to video_frames/
# given YouTube video url, res, name, ideal length
def video_to_frames(video_url, res, video_name, length):
    """
    video_url: url of youtube video
    res: resolution; eg '144p'
    video_name: name of video
    length: trim length
    """
    # grab YouTube object from url and download it to data/
    yt = get_yt(video_url)
    if yt is None:
        return
    download_video(yt, res, video_name)

    # trim to first length seconds
    ffmpeg_extract_subclip('data/' + video_name + '.mp4',
                           0, length, targetname='data/' + video_name + '_trimmed.mp4')

    # loads video and extracts frame by frame
    vid_cap = cv2.VideoCapture('data/' + video_name + '_trimmed.mp4')
    success, image = vid_cap.read()
    count = 0
    while success:
        if not os.path.exists("video_frames"):
            os.makedirs("video_frames")
        cv2.imwrite("video_frames/frame%d.jpg" % count, image)  # save frame as JPEG file
        success, image = vid_cap.read()
        print('Read a new frame: ', success)
        count += 1


# Helper function to download YT vid given YouTube Object
def download_video(yt, res, name='vid'):
    """
    yt: youtube object
    res: resolution; eg '144p'
    name: name of video, default vid
    """
    try:
        # filter by resolution and to mp4
        stream = yt.streams.filter(res=RES, mime_type='video/mp4').all()[0]

        if not os.path.exists("data"):
            os.makedirs("data")
        stream.download(output_path='data/', filename=name)

    except:
        print("Connection Error Download Vid", res)
        return False

    return True


# Helper function to return YouTube object given link
def get_yt(link):
    print(link)
    try:
        # object creation using YouTube which was imported in the beginning
        yt = YouTube(link, )
    except:
        print("YT Error")
        return None

    return yt


# Tester code, call from main() of main program
video_to_frames(YT_ROOT+VIDEO_URL, RES, VIDEO_NAME, LENGTH)
