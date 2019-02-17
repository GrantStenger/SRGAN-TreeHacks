# Import Dependencies
from flask import Flask, render_template, request
from video_to_frame import video_to_frames
from frames_to_video import frames_to_video
from video_splitter import run_left_right


app = Flask(__name__)
app.secret_key = 'YouWillNeverGuessThisSuperSecretKey'


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/video', methods=['POST'])
def video():
    data = request.form
    youtube_url = data['youtube_url']

    # TODO do stuff with url here, plz help
    # video_to_frames(youtube_url, '144p', 'test_vid', 5)
    # video_to_frames(youtube_url, '720p', 'test_vid', 5)
    # frames_to_video('video_frames144p', 'out_vid144p.mp4', 8)
    # frames_to_video('video_frames720p', 'out_vid720p.mp4', 8)
    run_left_right('out_vid144p.mp4', 'out_vid720p.mp4', 'output_vid.mp4')

    # Pass parameters to template similar to youtube_url and look at video.html template
    return render_template('video.html', youtube_url=youtube_url, yt_video=None)


if __name__ == '__main__':
    app.run(debug=True)
