# generate_readme.py
readme_content = """# AI Companion - Child Bot

An AI-powered chatbot companion for children with emotion detection and text-to-speech capabilities.

## Features

- 🤖 Interactive chatbot with natural language processing
- 😊 Real-time emotion detection using camera
- 🔊 Text-to-speech responses
- 💬 Multiple response modes (Text, Voice, Both)
- 📊 Disease prediction models integration
- 👤 User authentication system

## Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd "MJ project"

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
# Run the Flask application
python app.py
```

Open browser: `http://localhost:8000`

## Technologies Used

- Flask
- Flask-SQLAlchemy
- gTTS
- Bootstrap
- JavaScript

## Author

Your Name
"""

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme_content)

print("✅ README.md generated successfully!")