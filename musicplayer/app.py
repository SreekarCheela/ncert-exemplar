import os
import random
from flask import Flask, render_template, send_from_directory, jsonify

app = Flask(__name__)

SONGS_DIRECTORY = 'songs'

@app.route('/')
def index():
    songs = [song for song in os.listdir(SONGS_DIRECTORY) if song.endswith('.mp3')]
    random_song = random.choice(songs)
    return render_template('index.html', random_song=random_song)

@app.route('/random')
def get_random_song():
    songs = [song for song in os.listdir(SONGS_DIRECTORY) if song.endswith('.mp3')]
    random_song = random.choice(songs)
    return jsonify({'random_song': random_song})

@app.route('/songs/<path:filename>')
def download_file(filename):
    return send_from_directory('songs', filename, as_attachment=True, mimetype='audio/mpeg')

if __name__ == '__main__':
    app.run(debug=True)
