from pytube import YouTube
import argparse
import cv2
import os


parser = argparse.ArgumentParser()
parser.add_argument('--video_ext', default='P1LNrzr8JzM',
                    help='extension in YT url after v=')
parser.add_argument('--res', default='240p',
                    help='desired download resolution')
parser.add_argument('--video_name', default='test_vid',
                    help='name of video')
args = parser.parse_args()

YT_ROOT = 'https://www.youtube.com/watch?v='
VIDEO_URL = args.video_ext
RES = args.res
VIDEO_NAME = args.video_name


# Function that saves array of frames to video_frames/
# given YouTube video url, res, name
def video_to_frames(video_url, res, video_name):
    """
    video_url: url of youtube video
    res: resolution; eg '144p'
    video_name: name of video
    """
    # grab YouTube object from url and download it to data/
    yt = get_yt(video_url)
    if yt is None:
        return
    download_video(yt, res, video_name)

    # loads video and extracts frame by frame
    vid_cap = cv2.VideoCapture('data/' + VIDEO_NAME + '.mp4')
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
    name: name of video
    """
    try:
        # filter by resolution and to mp4
        stream = yt.streams.filter(res=RES, mime_type='video/mp4').all()[0]
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


video_to_frames(YT_ROOT+VIDEO_URL, RES, VIDEO_NAME)
