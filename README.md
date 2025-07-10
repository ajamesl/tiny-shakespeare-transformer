# 🎭 Tiny Shakespeare Generator

A modern FastAPI web application that generates Shakespearean text using a trained transformer model with real-time streaming capabilities.

## 🌐 Live Demo

**Try it now:** [http://65.109.84.92:8000/](http://65.109.84.92:8000/)

## 🚀 Features

- **Modern Web Interface**: Clean, responsive UI with real-time text streaming
- **Token-by-Token Streaming**: Watch text generate character by character
- **Configurable Generation**: Control the number of tokens (50-1000)
- **Pre-trained Model**: Uses a transformer model trained on Shakespeare's works
- **Docker Support**: Easy deployment with Docker and Docker Compose
- **FastAPI Backend**: RESTful API with automatic documentation

## 📁 Project Structure

```
tiny-shakespeare-generator/
├── src/                    # Source code package
│   ├── __init__.py
│   ├── config.py          # Model configuration
│   ├── model.py           # Transformer model architecture
│   ├── train.py           # Training script
│   └── utils.py           # Utility functions
├── app/                   # FastAPI application
│   ├── __init__.py
│   ├── main.py           # FastAPI app and routes
│   └── inference.py      # Text generation functions
├── data/                  # Training data
│   └── input.txt         # Shakespeare text corpus
├── checkpoints/           # Model checkpoints
│   └── tiny_shakespeare.pt # Trained model
├── templates/             # HTML templates
│   └── index.html        # Modern web interface
├── static/               # Static files (CSS, JS, images)
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose setup
├── .dockerignore         # Docker ignore rules
├── pyproject.toml        # Python project configuration
├── uv.lock              # Dependency lock file
├── requirements.txt      # Python dependencies
└── run_app.py           # Application launcher
```

## 🛠️ Installation

### Option 1: Using uv (Recommended)

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd tiny-shakespeare-generator
   ```

2. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install dependencies**:
   ```bash
   uv sync
   ```

### Option 2: Using pip

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd tiny-shakespeare-generator
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Option 3: Using Docker

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd tiny-shakespeare-generator
   ```

2. **Build and run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

## 🎯 Usage

### Web Application

1. **Start the server**:
   ```bash
   # Using uv
   uv run python run_app.py
   
   # Using pip
   python run_app.py
   
   # Using Docker
   docker-compose up
   ```

2. **Open your browser** and navigate to `http://localhost:8000`

3. **Generate text**:
   - Set the number of tokens to generate (50-1000, default: 500)
   - Click "Generate Text" to see the results stream in real-time

### API Usage

#### Generate Text with Streaming (POST)
```bash
curl -X POST "http://localhost:8000/generate/" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "max_tokens=300"
```

#### Generate Text Batch (POST)
```bash
curl -X POST "http://localhost:8000/generate-batch/" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "max_tokens=300"
```

#### Health Check
```bash
curl "http://localhost:8000/health"
```

## 🧠 Model Architecture

The model uses a transformer architecture with:
- **Embedding Dimension**: 384
- **Attention Heads**: 6
- **Layers**: 6
- **Context Length**: 256 tokens
- **Dropout**: 0.2

## 🔧 Configuration

Modify `src/config.py` to adjust:
- Model hyperparameters
- Training settings
- Device selection (CPU/GPU/MPS)

## 📊 Training

To train your own model:

1. **Prepare your data**: Place text data in `data/input.txt`
2. **Run training**: 
   ```bash
   uv run python src/train.py
   ```
3. **Monitor progress**: Training logs will show loss progression
4. **Model saved**: Final model saved to `checkpoints/tiny_shakespeare.pt`

## 🚀 Deployment

### Local Development
```bash
uv run python run_app.py
```

### Production
```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment
```bash
docker-compose up --build
```

The application will be available at `http://localhost:8000`

## 📝 API Documentation

Once running, visit `http://localhost:8000/docs` for interactive API documentation powered by FastAPI.

## 🎨 UI Features

- **Modern Design**: Clean, responsive interface with gradient backgrounds
- **Real-time Streaming**: Watch text generate character by character
- **Error Handling**: Graceful error messages and loading states
- **Responsive Layout**: Works on desktop and mobile devices

## 🐳 Docker Support

The project includes full Docker support:
- `Dockerfile`: Multi-stage build with Python 3.11
- `docker-compose.yml`: Easy development setup
- `.dockerignore`: Optimized build context

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Based on the transformer architecture from "Attention Is All You Need"
- Trained on Shakespeare's works from Project Gutenberg
- Inspired by Andrej Karpathy's educational content
- Built with FastAPI and modern web technologies
