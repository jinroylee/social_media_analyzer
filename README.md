# Social Media Engagement Prediction API

A FastAPI-based microservice that predicts social media engagement scores using CLIP (Contrastive Language-Image Pre-training) and sentiment analysis. The API processes text, images, and comments to generate engagement predictions for social media content.

## Features

- **Multi-modal Analysis**: Combines text, image, and sentiment analysis for engagement prediction
- **CLIP Integration**: Uses OpenAI's CLIP model for vision-language understanding
- **Sentiment Analysis**: Processes comments to extract sentiment features
- **S3 Model Storage**: Automatically downloads trained models from AWS S3
- **LoRA Support**: Efficient fine-tuning with Low-Rank Adaptation
- **Dockerized**: Production-ready containerized deployment
- **Health Monitoring**: Built-in health checks and model information endpoints

## Architecture

```
├── src/
│   ├── api/
│   │   └── routes.py           # API endpoint definitions
│   ├── schemas/
│   │   ├── request.py          # Request data models
│   │   └── response.py         # Response data models
│   ├── services/
│   │   ├── preprocessing.py    # Text, image, and sentiment processing
│   │   └── prediction.py       # Model loading and inference
│   ├── app.py                  # FastAPI application setup
│   └── config.py               # Configuration management
├── model/
│   ├── clip_regressor.py       # CLIP-based engagement prediction model
│   └── weights/                # Model weights storage
├── docker/
│   ├── Dockerfile              # Production container
│   └── Dockerfile.mlflow       # MLflow container
├── main.py                     # Application entry point
└── pyproject.toml              # Dependencies and project metadata
```

## API Endpoints

### Core Endpoints

- **POST** `/api/v1/predict` - Predict engagement score for social media content
- **GET** `/api/v1/health` - Service health check
- **GET** `/api/v1/model-info` - Detailed model information
- **GET** `/` - API documentation redirect

### Prediction Request Format

```json
{
    "raw_text": "Check out this amazing sunset! #beautiful #nature",
    "image_base64": "iVBORw0KGgo...", 
    "comments": [
        "Wow, so beautiful!",
        "Amazing shot!",
        "Love this!"
    ]
}
```

### Prediction Response Format

```json
{
    "engagement_score": 0.75,
    "sentiment_score": 0.8,
    "cleaned_text": "check out this amazing sunset beautiful nature"
}
```

## Setup & Installation

### Prerequisites

- Python 3.10+
- UV package manager (recommended) or pip
- AWS credentials (if using S3 model storage)

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd social_media_analyzer
   ```

2. **Install dependencies**
   ```bash
   # Using UV (recommended)
   uv sync
   
   # Or using pip
   pip install -e .
   ```

3. **Configuration**
   Create a `.env` file in the project root:
   ```env
   # Server Configuration
   HOST=0.0.0.0
   PORT=8000
   LOG_LEVEL=info
   RELOAD=true
   
   # Model Configuration
   USE_LORA=true
   LORA_RANK=8
   FORCE_CPU=false
   
   # S3 Configuration (if using S3 model storage)
   USE_S3_MODEL=true
   S3_BUCKET_NAME=your-bucket-name
   S3_MODEL_KEY=models/best_model_lora.pth
   AWS_REGION=ap-northeast-2
   
   # API Limits
   MAX_TEXT_LENGTH_REQUEST=10000
   MAX_COMMENTS=1000
   ```

4. **Run the server**
   ```bash
   python main.py
   ```
   
   The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

### Docker Deployment

1. **Build the container**
   ```bash
   docker build -f docker/Dockerfile -t social-media-analyzer .
   ```

2. **Run the container**
   ```bash
   docker run -p 8000:8000 \
     -e AWS_ACCESS_KEY_ID=your-key \
     -e AWS_SECRET_ACCESS_KEY=your-secret \
     -e S3_BUCKET_NAME=your-bucket \
     social-media-analyzer
   ```

## Model Details

### CLIP Engagement Regressor

The core model combines:
- **CLIP Vision Encoder**: Processes images to extract visual features
- **CLIP Text Encoder**: Processes text to extract semantic features  
- **Sentiment Features**: Aggregated sentiment from comments
- **Engagement Head**: Multi-layer perceptron for final score prediction

### LoRA Fine-tuning

The model supports Low-Rank Adaptation (LoRA) for efficient fine-tuning:
- Rank: 8 (configurable)
- Alpha: 16
- Target modules: Attention and MLP layers
- Dropout: 0.1

### S3 Model Storage

Models are automatically downloaded from S3 on startup:
- Primary path: `models/best_model_lora.pth`
- Fallback paths for different model versions
- Local caching to avoid repeated downloads

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `LOG_LEVEL` | `info` | Logging level |
| `USE_LORA` | `true` | Enable LoRA fine-tuning |
| `LORA_RANK` | `8` | LoRA rank parameter |
| `USE_S3_MODEL` | `true` | Download model from S3 |
| `S3_BUCKET_NAME` | `socialmediaanalyzer` | S3 bucket name |
| `FORCE_CPU` | `false` | Force CPU usage |
| `MAX_TEXT_LENGTH_REQUEST` | `10000` | Max text length |
| `MAX_COMMENTS` | `1000` | Max comments per request |

### Model Configuration

- **CLIP Model**: `openai/clip-vit-large-patch14`
- **Sentiment Model**: `tabularisai/multilingual-sentiment-analysis`
- **Image Size**: 256x256 pixels
- **Text Length**: 77 tokens (CLIP standard)

## Development

### Project Structure

- **`src/api/`**: API route definitions and request handling
- **`src/services/`**: Core business logic (preprocessing, prediction)
- **`src/schemas/`**: Pydantic models for request/response validation
- **`model/`**: Custom PyTorch model definitions
- **`docker/`**: Containerization files

### Key Components

1. **PreprocessingService**: Handles text cleaning, image processing, and sentiment analysis
2. **PredictionService**: Manages model loading, S3 integration, and inference
3. **CLIPEngagementRegressor**: Custom PyTorch model combining CLIP with engagement prediction

## Health Monitoring

### Health Check Endpoint

The `/api/v1/health` endpoint provides:
- Service status (healthy/unhealthy)
- Model loading status
- Device information (CPU/GPU)

### Model Information

The `/api/v1/model-info` endpoint returns:
- Model architecture details
- Parameter counts
- LoRA configuration
- S3 integration status

## Production Considerations

### Performance

- Models cached in memory after first load
- Automatic GPU utilization when available
- Batch processing for sentiment analysis
- Efficient image preprocessing pipeline

### Scalability

- Stateless design for horizontal scaling
- Docker-ready for container orchestration
- Health checks for load balancer integration
- Environment-based configuration

### Security

- Input validation with Pydantic schemas
- Request size limits
- Non-root container execution
- Minimal container image (production build)

## License

[Add your license here]

## Contributing

[Add contribution guidelines here] 