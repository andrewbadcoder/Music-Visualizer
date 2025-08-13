import os
from datetime import timedelta

class Config:
    """Base configuration class"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production-2024'
    
    # Audio processing settings
    SAMPLE_RATE = 22050  # Standard sample rate for audio processing
    HOP_LENGTH = 512     # Number of samples between successive frames
    N_FFT = 2048         # Length of the FFT window
    BUFFER_SIZE = 1024   # Audio buffer size for real-time processing
    
    # Visualization settings
    FREQUENCY_BINS = 128    # Number of frequency bins to display
    SMOOTHING_FACTOR = 0.8  # Smoothing for visualization (0-1)
    UPDATE_RATE = 30        # FPS for visualization updates
    
    # File upload settings
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac'}
    
    # Socket.IO settings
    SOCKETIO_CORS_ALLOWED_ORIGINS = "*"
    SOCKETIO_PING_TIMEOUT = 60
    SOCKETIO_PING_INTERVAL = 25
    
    # Security settings
    WTF_CSRF_ENABLED = False  # Disabled for API endpoints, enable for forms
    PERMANENT_SESSION_LIFETIME = timedelta(hours=1)
    
    # Performance settings
    THREADING = True
    PROCESSES = 1
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'app.log'
    
    @staticmethod
    def init_app(app):
        """Initialize application with configuration"""
        # Create upload directory if it doesn't exist
        upload_dir = app.config.get('UPLOAD_FOLDER')
        if upload_dir and not os.path.exists(upload_dir):
            os.makedirs(upload_dir)

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False
    
    # Development-specific audio settings
    SAMPLE_RATE = 22050
    UPDATE_RATE = 30
    
    # Relaxed security for development
    SOCKETIO_CORS_ALLOWED_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:5000"]
    
    # Development logging
    LOG_LEVEL = 'DEBUG'
    
class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    WTF_CSRF_ENABLED = False
    
    # Test-specific settings
    SAMPLE_RATE = 16000  # Lower sample rate for faster tests
    BUFFER_SIZE = 512
    
    # In-memory database for tests
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    # Production security
    SECRET_KEY = os.environ.get('SECRET_KEY')
    WTF_CSRF_ENABLED = True
    
    # Production performance settings
    SAMPLE_RATE = 44100  # Higher quality for production
    UPDATE_RATE = 60     # Higher FPS for smoother visualization
    
    # Strict CORS for production
    SOCKETIO_CORS_ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', '').split(',')
    
    # Production logging
    LOG_LEVEL = 'WARNING'
    LOG_FILE = '/var/log/musicvisualizer/app.log'
    
    @classmethod
    def init_app(cls, app):
        Config.init_app(app)
        
        # Production-specific initialization
        import logging
        from logging.handlers import RotatingFileHandler
        
        # Set up file logging
        if not app.debug and not app.testing:
            if not os.path.exists('/var/log/musicvisualizer'):
                os.makedirs('/var/log/musicvisualizer')
            
            file_handler = RotatingFileHandler(
                '/var/log/musicvisualizer/app.log',
                maxBytes=10240000,  # 10MB
                backupCount=10
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
            ))
            file_handler.setLevel(logging.WARNING)
            app.logger.addHandler(file_handler)
            
            app.logger.setLevel(logging.WARNING)
            app.logger.info('Music Visualizer startup')

class DockerConfig(ProductionConfig):
    """Docker deployment configuration"""
    
    # Docker-specific settings
    HOST = '0.0.0.0'
    PORT = int(os.environ.get('PORT', 5000))
    
    # Use environment variables for Docker secrets
    SECRET_KEY = os.environ.get('SECRET_KEY', 'docker-fallback-key')
    
    # Docker volume paths
    UPLOAD_FOLDER = '/app/uploads'
    LOG_FILE = '/app/logs/app.log'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'docker': DockerConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get configuration based on environment"""
    env = os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])