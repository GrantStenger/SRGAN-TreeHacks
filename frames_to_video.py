# Import Dependencies
import argparse
import cv2
import numpy as np
import os


# REPLICATE ARGUMENT PARSING IN MAIN #
parser = argparse.ArgumentParser()
parser.add_argument('--frame_source',
                    help='location of frames')
parser.add_argument('--output', required=False,
                    default='output.mp4', help='output video filename')
parser.add_argument('--length', default=10, type=int,
                    help='length of final video')

args = parser.parse_args()

FRAMES_DIR = args.frame_source
OUTPUT = args.output
LENGTH = args.length


# Function that takes frames and uses CV to mush them into video
def frames_to_video(frames_dir, outfile, length):
    """
    frame_dir: dir of frame .jpeg files
    output: name of .mp4 output video file
    """
    # Create list of all images
    total_images = len([name for name in os.listdir(frames_dir)])

    # these many frames per second
    frame_rate = total_images/length

    # Determine the width and height from the first image
    image_path = frames_dir + '/frame0.jpg'
    frame = cv2.imread(image_path)
    cv2.imshow('video', frame)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outfile, fourcc, frame_rate, (width, height))

    for temp in range(0, total_images):

        image_path = os.path.join(frames_dir + '/frame' + str(temp) + '.jpg')
        frame = cv2.imread(image_path)
        out.write(frame) # Write out frame to video
        cv2.imshow('video', frame)

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()

    print("The output video is {}".format(outfile))


frames_to_video(FRAMES_DIR, OUTPUT, LENGTH)
