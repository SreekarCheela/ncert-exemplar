function playRandom() {
    // Fetch the current random song from the server
    fetch('/random')
        .then(response => response.json())
        .then(data => {
            // Directly set the new random song URL to the audio player source
            var audioPlayer = document.getElementById('audio-player');
            audioPlayer.src = '/songs/' + data.random_song;
            audioPlayer.load();
            audioPlayer.play();

            // Update the displayed song name
            var songName = document.querySelector('strong');
            songName.textContent = data.random_song;
        })
        .catch(error => console.error('Error:', error));
}
