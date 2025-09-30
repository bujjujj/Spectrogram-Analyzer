# Spectrogram-Analyzer
A program that uses ML to learn patterns in spectrograms, and assigns tags/genres to them. Trained on the MagnaTagATune Dataset. Currently using 10k tracks in training.

Run program like this:
Navigate to ".../spectrogram_analyzer_project" in terminal
Run "python -m src.data_processing"
Run "python -m src.training"

Run "python app/app.py"

Something like this should appear:
\\* Serving Flask app 'app'
\\* Running on http://127.0.0.1:5000
Press CTRL+C to quit

Open web browser (i.e Chrome) and navigate to URL shown in terminal
