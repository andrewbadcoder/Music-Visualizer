import librosa
import numpy as np
import os
import logging
from config import Config

# Set up logging for this module
logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioVisualizer:
    """
    Handles audio file loading, analysis, and data preparation for visualization.
    This class is designed to be a singleton for a given audio file,
    holding the processed data and serving it to the frontend.
    """

    def __init__(self, config=Config):
        """Initializes the visualizer with configuration settings."""
        self.config = config
        self.audio_data = None
        self.sample_rate = None
        self.stft_data = None
        self.db_spectrogram = None
        self.current_frame_index = 0
        logger.info("AudioVisualizer initialized.")

    def load_audio_file(self, file_path):
        """
        Loads an audio file and performs initial analysis.
        
        Args:
            file_path (str): The full path to the audio file.
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"Audio file not found at {file_path}")
        
        try:
            # librosa.load returns audio data (a numpy array) and the sample rate
            self.audio_data, self.sample_rate = librosa.load(
                file_path,
                sr=self.config.SAMPLE_RATE
            )
            logger.info(f"Audio file loaded successfully: {file_path}")
            
            # Perform Short-Time Fourier Transform (STFT)
            # This converts the audio signal into the frequency domain.
            self.stft_data = librosa.stft(
                self.audio_data,
                n_fft=self.config.N_FFT,
                hop_length=self.config.HOP_LENGTH
            )
            
            # Convert the magnitude spectrogram to decibels for better visualization.
            # This is a logarithmic scale, which is more representative of human hearing.
            self.db_spectrogram = librosa.amplitude_to_db(
                np.abs(self.stft_data),
                ref=np.max
            )
            logger.info("STFT and decibel spectrogram generated.")
            
        except Exception as e:
            logger.error(f"Error processing audio file: {e}")
            self.audio_data = None
            self.stft_data = None
            self.db_spectrogram = None
            raise

    def get_visualization_frame(self):
        """
        Returns the next frame of data for visualization.
        
        This method will iterate through the spectrogram and return a slice
        of frequency data for the current frame.
        """
        if self.db_spectrogram is None:
            return None

        # Check if we have reached the end of the audio
        if self.current_frame_index >= self.db_spectrogram.shape[1]:
            logger.info("End of audio reached.")
            return None

        # Get the frequency data for the current time frame.
        # We take the first `FREQUENCY_BINS` rows (low frequencies).
        current_frame = self.db_spectrogram[:self.config.FREQUENCY_BINS, self.current_frame_index]
        
        # Increment the frame index for the next call.
        self.current_frame_index += 1
        
        # Scale the data to be between 0 and 1 for easier frontend processing
        # This normalization is crucial for consistent visualization.
        normalized_frame = (current_frame - current_frame.min()) / (current_frame.max() - current_frame.min())
        
        # Convert to a list for JSON serialization
        return normalized_frame.tolist()

    def reset_playback(self):
        """Resets the frame index to start from the beginning."""
        self.current_frame_index = 0
        logger.info("Playback frame index reset.")