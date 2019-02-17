import subprocess

def run_left_right(lr_vid, hr_vid, output_vid):

    command = "ffmpeg -y -i {0} -vf scale=480:852 output.mp4".format(lr_vid)
    subprocess.run(command, shell=True)

    command = "ffmpeg -y -i output.mp4 -i {1} -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' \
    -map [vid] -c:v libx264 -crf 23 -preset veryfast {2}".format(lr_vid, hr_vid, output_vid)

    subprocess.run(command, shell=True)

