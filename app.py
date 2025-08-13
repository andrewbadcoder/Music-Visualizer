# Import necessary modules
import os
from flask import Flask, render_template, request, jsonify
# We need to import SocketIO for real-time communication
from flask_socketio import SocketIO, emit
import logging

# Import the core visualization module
from config import get_config
from visualizer import AudioVisualizer

def create_app():
    """Factory function to create and configure the Flask app."""
    
    # Get the configuration based on the environment variable
    config_obj = get_config()
    
    app = Flask(__name__)
    app.config.from_object(config_obj)

    # Initialize extensions
    # Flask-SocketIO is essential for real-time data streaming
    socketio = SocketIO(app)
    
    # Create a global instance of our real visualizer
    visualizer = AudioVisualizer(config=config_obj)

    # A simple route to serve the front-end
    @app.route('/')
    def index():
        return render_template('index.html')

    # Add a new route for file uploads using a standard POST request
    @app.route('/upload', methods=['POST'])
    def upload_file():
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected for uploading'}), 400
            
        if file and allowed_file(file.filename):
            try:
                # Save the uploaded file to the configured folder
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                
                # Process the uploaded file with our real visualizer
                visualizer.load_audio_file(filepath)
                
                # Use Socket.IO to tell the client the audio is ready
                socketio.emit('audio_ready', {'message': 'Audio file processed and ready for visualization'})
                return jsonify({'message': 'File uploaded and processed successfully'}), 200
            except Exception as e:
                app.logger.error(f"File upload or processing failed: {e}")
                return jsonify({'error': f'Failed to process file: {e}'}), 500
        else:
            return jsonify({'error': 'Allowed file types are ' + ', '.join(app.config['ALLOWED_EXTENSIONS'])}), 400

    def allowed_file(filename):
        """Check if uploaded file has an allowed extension"""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

    # New Socket.IO event handler to send visualization data
    @socketio.on('get_frame')
    def send_frame():
        """
        Responds to a 'get_frame' request from the client by sending the next frame
        of visualization data.
        """
        frame_data = visualizer.get_visualization_frame()
        if frame_data:
            # We emit the data back to the client
            emit('visualization_data', {'data': frame_data})
        else:
            # No more frames, tell the client to stop
            emit('end_of_audio', {'message': 'End of audio stream'})
            
    # Socket.IO event to reset playback for a new session
    @socketio.on('reset')
    def reset_visualizer():
        """Resets the visualizer to start from the beginning."""
        visualizer.reset_playback()
        emit('reset_complete', {'message': 'Visualizer has been reset'})

    return app, socketio

# Run the application
if __name__ == '__main__':
    app, socketio = create_app()
    config_name = os.environ.get('FLASK_ENV', 'development')
    
    print(f"Starting Music Visualizer in {config_name} mode...")
    print(f"Sample Rate: {app.config['SAMPLE_RATE']} Hz")
    print(f"Frequency Bins: {app.config['FREQUENCY_BINS']}")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print("Visit: http://localhost:5000")
    
    # We must use socketio.run() to properly handle WebSocket events
    socketio.run(app, debug=app.config.get('DEBUG', True), host='0.0.0.0')