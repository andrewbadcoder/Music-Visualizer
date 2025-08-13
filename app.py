
from flask import Flask, render_template, request, jsonify
from config import get_config
import numpy as np
import json
import os

# Create Flask app
app = Flask(__name__)

# Load configuration
config_class = get_config()
app.config.from_object(config_class)

# Initialize the app with config
config_class.init_app(app)

class SimpleAudioProcessor:
    def __init__(self):
        # Use configuration values
        self.sample_rate = app.config['SAMPLE_RATE']
        self.frequency_bins = app.config['FREQUENCY_BINS']
        self.smoothing_factor = app.config['SMOOTHING_FACTOR']
        
    def extract_features(self, audio_data=None):
        """Extract audio features for visualization - simplified version"""
        try:
            # For now, generate dummy data for testing
            # Later we'll add real audio processing when librosa works
            
            # Generate realistic-looking spectrum data
            spectrum = []
            for i in range(self.frequency_bins):
                # Create a spectrum that looks like music (more energy in lower frequencies)
                freq_energy = np.random.exponential(1.0) * (1.0 - i/self.frequency_bins) * 100
                spectrum.append(max(0, freq_energy))
            
            # Generate other audio features
            rms = np.random.random() * 0.5
            spectral_centroid = 1000 + np.random.random() * 2000
            tempo = 120 + np.random.random() * 60
            
            # Create magnitude matrix for heatmap visualization
            magnitude_matrix = []
            for i in range(min(64, self.frequency_bins)):  # Reduced size
                row = [max(0, np.random.exponential(1.0) * (1.0 - i/64) * 50) for _ in range(20)]
                magnitude_matrix.append(row)
            
            return {
                'spectrum': spectrum,
                'rms': float(rms),
                'spectral_centroid': float(spectral_centroid),
                'tempo': float(tempo),
                'magnitude_matrix': magnitude_matrix
            }
            
        except Exception as e:
            app.logger.error(f"Error processing audio: {e}")
            return None

# Global audio processor instance
audio_processor = SimpleAudioProcessor()

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_audio():
    """Handle audio file upload - simplified version"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Supported: wav, mp3, flac, ogg, m4a, aac'}), 400
    
    try:
        # For now, just generate dummy features
        # Later we'll add real audio file processing
        features = audio_processor.extract_features()
        
        if features:
            return jsonify({
                'success': True,
                'features': features,
                'duration': 180.0,  # Dummy duration
                'sample_rate': audio_processor.sample_rate,
                'filename': file.filename
            })
        else:
            return jsonify({'error': 'Failed to process audio'}), 500
            
    except Exception as e:
        app.logger.error(f"Audio upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_features')
def get_features():
    """Get current audio features - for polling instead of websockets"""
    features = audio_processor.extract_features()
    if features:
        return jsonify(features)
    else:
        return jsonify({'error': 'Failed to generate features'}), 500

@app.route('/start_processing', methods=['POST'])
def start_processing():
    """Start audio processing"""
    return jsonify({'status': 'Processing started (demo mode)'})

@app.route('/stop_processing', methods=['POST'])
def stop_processing():
    """Stop audio processing"""
    return jsonify({'status': 'Processing stopped'})

if __name__ == '__main__':
    # Get configuration
    config_name = os.environ.get('FLASK_ENV', 'development')
    print(f"Starting Simple Music Visualizer in {config_name} mode...")
    print(f"Sample Rate: {app.config['SAMPLE_RATE']} Hz")
    print(f"Frequency Bins: {app.config['FREQUENCY_BINS']}")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print("\nNote: Running in demo mode without real-time WebSockets")
    print("Visit: http://localhost:5000")
    
    # Run the application
    app.run(debug=app.config['DEBUG'], 
           host='0.0.0.0', 
           port=5000)