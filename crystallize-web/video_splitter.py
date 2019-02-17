import subprocess

def run_left_right(left_vid, right_vid, output_vid):

    command = "ffmpeg -i {0} -i {1} -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' \
    -map [vid] -c:v libx264 -crf 23 -preset veryfast {2}".format(left_vid, right_vid, output_vid)

    subprocess.run(command, shell=True)
