# Import Dependencies
import argparse
import cv2
import numpy as np

def main():
    # Create list of all images
    images = []
    for f in os.listdir(FIELDS.dir_path):
        if f.endswith(FIELDS.ext):
            images.append(f)

    # Determine the width and height from the first image
    image_path = os.path.join(FIELDS.dir_path, images[0])
    frame = cv2.imread(image_path)
    cv2.imshow('video', frame)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))

    for image in images:

        image_path = os.path.join(FIELDS.dir_path, image)
        frame = cv2.imread(image_path)

        out.write(frame) # Write out frame to video

        cv2.imshow('video',frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
            break

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()

    print("The output video is {}".format(output))

if __name__ == "__main__":

    # Construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", default=str, help="path to the directory containing the images")
    parser.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
    parser.add_argument('--video_name', default='usc_village', help='name of video')
    parser.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")

    # Parses known arguments
    FLAGS, unparsed = argparse.parse_known_args()

    main()












# Function that returns array of frames given YouTube video url, res, name
def video_to_frames(video_url, res, video_name, length=20):
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

    # trim everything but first 'length' seconds of video
    ffmpeg_extract_subclip('data/' + video_name + '.mp4', 0, 20,
                           targetname='trimmed_' + video_name + '.mp4')


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
