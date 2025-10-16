#!/bin/bash

# Aviator Predictor Setup Script
# This script sets up the development environment

set -e

echo "🚀 Setting up Aviator Predictor..."

# Check if Python 3.9+ is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then
    echo "✅ Python $PYTHON_VERSION detected"
else
    echo "❌ Python $PYTHON_VERSION is too old. Please install Python 3.9 or higher."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "⚠️  Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data logs models screenshots backups

# Set up environment variables
echo "🔐 Setting up environment variables..."
if [ ! -f ".env" ]; then
    cat > .env << EOF
# Aviator Predictor Environment Configuration
FLASK_ENV=development
FLASK_APP=app.py
DEBUG=True
SECRET_KEY=dev-secret-key-change-in-production
DATABASE_URL=sqlite:///aviator_predictor.db

# Cloud Provider Credentials (Optional)
# AWS_ACCESS_KEY_ID=your_aws_access_key
# AWS_SECRET_ACCESS_KEY=your_aws_secret_key
# AZURE_STORAGE_CONNECTION_STRING=your_azure_connection_string
# GOOGLE_APPLICATION_CREDENTIALS=path/to/gcp/credentials.json

# API Keys (Optional)
# SPRIBE_API_KEY=your_spribe_api_key
# EVOLUTION_API_KEY=your_evolution_api_key
EOF
    echo "✅ Environment file created (.env)"
else
    echo "⚠️  Environment file already exists"
fi

# Initialize database
echo "🗄️  Initializing database..."
python -c "
from app import app
from config.database import init_database
with app.app_context():
    init_database(app)
    print('Database initialized successfully')
"

# Check if Chrome is installed (for Selenium)
echo "🌐 Checking Chrome installation..."
if command -v google-chrome &> /dev/null || command -v chromium-browser &> /dev/null; then
    echo "✅ Chrome/Chromium found"
else
    echo "⚠️  Chrome/Chromium not found. Installing..."
    
    # Detect OS and install Chrome
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v apt-get &> /dev/null; then
            # Ubuntu/Debian
            wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
            echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" | sudo tee /etc/apt/sources.list.d/google-chrome.list
            sudo apt-get update
            sudo apt-get install -y google-chrome-stable
        elif command -v yum &> /dev/null; then
            # CentOS/RHEL
            sudo yum install -y https://dl.google.com/linux/direct/google-chrome-stable_current_x86_64.rpm
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install --cask google-chrome
        else
            echo "❌ Please install Chrome manually from https://www.google.com/chrome/"
        fi
    fi
fi

# Download ChromeDriver
echo "🚗 Setting up ChromeDriver..."
if [ ! -f "chromedriver" ]; then
    echo "📥 Downloading ChromeDriver..."
    
    # Get latest ChromeDriver version
    CHROMEDRIVER_VERSION=$(curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE)
    
    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        CHROMEDRIVER_OS="linux64"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        CHROMEDRIVER_OS="mac64"
    else
        CHROMEDRIVER_OS="win32"
    fi
    
    # Download and extract
    curl -sS -o chromedriver.zip "http://chromedriver.storage.googleapis.com/$CHROMEDRIVER_VERSION/chromedriver_$CHROMEDRIVER_OS.zip"
    unzip chromedriver.zip
    chmod +x chromedriver
    rm chromedriver.zip
    
    echo "✅ ChromeDriver installed"
else
    echo "⚠️  ChromeDriver already exists"
fi

# Create sample configuration
echo "⚙️  Creating sample configurations..."
if [ ! -f "config/settings.json" ]; then
    mkdir -p config
    cat > config/settings.json << EOF
{
  "app_name": "Aviator Predictor",
  "version": "1.0.0",
  "author": "MiniMax Agent",
  "settings": {
    "data_collection": {
      "interval_seconds": 60,
      "max_retries": 3,
      "timeout_seconds": 10
    },
    "prediction": {
      "model_type": "ensemble",
      "confidence_threshold": 0.7,
      "update_frequency": 300
    },
    "ui": {
      "theme": "dark",
      "refresh_rate": 1000,
      "show_notifications": true
    }
  }
}
EOF
    echo "✅ Sample configuration created"
fi

# Create startup script
echo "🎬 Creating startup script..."
cat > start.sh << EOF
#!/bin/bash
# Aviator Predictor Startup Script

echo "🚀 Starting Aviator Predictor..."

# Activate virtual environment
source venv/bin/activate

# Load environment variables
if [ -f .env ]; then
    export \$(cat .env | grep -v '^#' | xargs)
fi

# Start the application
echo "🌐 Starting web server on http://localhost:5000"
python app.py
EOF

chmod +x start.sh
echo "✅ Startup script created (start.sh)"

# Create Docker setup
echo "🐳 Docker setup available"
echo "   To build: docker build -t aviator-predictor ."
echo "   To run: docker-compose up -d"

# Final instructions
echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Review the .env file and add your API keys if needed"
echo "   2. Start the application: ./start.sh"
echo "   3. Open your browser to: http://localhost:5000"
echo ""
echo "📖 For more information, see README.md"
echo ""
echo "⚠️  Remember: This is for educational purposes only!"
echo "   Do not use for actual gambling decisions."
echo ""