# PDF Q&A Analysis Tool

A powerful web application that allows you to upload PDF documents and ask questions about their content using advanced AI models with Retrieval-Augmented Generation (RAG) technology.

## Features

- **PDF Upload & Processing**: Securely upload and process multiple PDF documents
- **Multi-Model Support**: Choose from different AI models (Hugging Face, OpenAI, Azure OpenAI)
- **Intelligent Q&A**: Ask natural language questions and get instant answers
- **RAG-Enabled**: Answers are grounded in your PDF content for higher accuracy
- **Vector Search**: Uses Pinecone vector database for semantic search
- **Customizable Behavior**: Define system messages to control AI behavior
- **Multi-PDF Querying**: Query single PDFs or all uploaded documents simultaneously
- **Persistent Storage**: Your PDFs and settings are saved between sessions

## Quick Start

### Prerequisites

- Python 3.11+
- Pinecone account (for vector storage)
- AI model API keys (at least one required)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/dikshant-dave007/pdf-chat-bot-analysis.git
cd pdf-chat-bot-analysis
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set up environment variables**

Create a `.env` file in the project root with your configuration:

```env
# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment

# AI Model APIs (configure at least one)
HUGGINGFACE_API_KEY=your_huggingface_token
OPENAI_API_KEY=your_openai_key
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
```

4. **Create necessary directories**

```bash
mkdir -p uploads pdfs_to_process logs templates
```

5. **Run the application**

```bash
python app.py
```

6. **Access the application**

Open your browser and navigate to `http://localhost:8000`

## Configuration

### AI Models

Configure at least one AI model in your `.env` file:

- Hugging Face (recommended for free tier)

```env
HUGGINGFACE_API_KEY=your_huggingface_token_here
```

- OpenAI

```env
OPENAI_API_KEY=your_openai_api_key_here
```

- Azure OpenAI

```env
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
```

### Pinecone Setup

1. Create a free account at https://pinecone.io
2. Create a new index with the following settings:
   - Dimension: 3072
   - Metric: cosine
3. Copy your API key and environment and add them to the `.env` file.

## Usage

### Uploading PDFs

- Single PDF Upload:
  - Use the upload area in the sidebar
  - Drag & drop or click to select PDF files
  - Click "Upload & Process PDFs"

- Bulk Processing:
  - Place PDFs in the `pdfs_to_process/` directory
  - Use the API endpoint: `POST /api/process-directory`

### Asking Questions

- Select PDF: Choose a specific PDF or "All PDFs" from the sidebar
- Choose Model: Select your preferred AI model
- Configure RAG: Toggle RAG mode on/off
- Set System Message: Customize how the AI should respond
- Ask Questions: Type your question in the chat input

### RAG Mode

- Enabled: Answers are based solely on your uploaded PDF content
- Disabled: Uses the AI model's general knowledge (may be less accurate for PDF-specific questions)

## API Endpoints

### Core Endpoints

- `POST /api/upload-pdf` - Upload and process a PDF
- `POST /api/ask-question` - Ask questions about PDFs
- `GET /api/list-pdfs` - Get list of uploaded PDFs
- `DELETE /api/delete-pdf/{id}` - Delete a specific PDF

### Settings Endpoints

- `GET /api/settings` - Get current settings
- `POST /api/system-message` - Update system message
- `POST /api/model-setting` - Change AI model
- `POST /api/rag-status` - Toggle RAG mode

### Debug Endpoints

- `GET /api/health` - Application health check
- `GET /api/debug/config` - Check configuration
- `GET /api/debug/pinecone` - Pinecone status
- `POST /api/debug/test-rag` - Test RAG pipeline

## Architecture

```
Frontend (HTML/CSS/JS)
        ↓
Backend (FastAPI)
        ↓
   AI Models
   ↓       ↓
OpenAI   HuggingFace
   ↓       ↓
Pinecone Vector DB
        ↓
   PDF Storage
```

## Key Components

- PDF Processor: Extracts text and chunks content
- Embedding Generator: Creates vector embeddings using OpenAI
- Vector Store: Pinecone for semantic search
- AI Clients: Multiple model providers for answers
- Persistent Storage: JSON-based document database

## Troubleshooting

### Common Issues

- PDF Text Extraction Fails
  - Ensure PDFs are not scanned/image-based
  - Check if PDFs contain extractable text

- Pinecone Connection Issues
  - Verify API key and environment
  - Check index dimension matches (3072)

- Model API Errors
  - Verify API keys are valid
  - Check rate limits and quotas
  - Ensure proper configuration for Azure OpenAI

- Upload Failures
  - Check file size (max 10MB)
  - Verify PDF format
  - Ensure `uploads` directory has write permissions

### Debug Mode

Use the debug endpoints to diagnose issues:

```bash
# Check application health
curl http://localhost:8000/api/health

# Test model availability
curl http://localhost:8000/api/debug/test-model?model_type=huggingface

# Check Pinecone status
curl http://localhost:8000/api/debug/pinecone
```

## Project Structure

```
pdf-chat-bot-analysis/
├── app.py                 # Main FastAPI application
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (ignored in VCS)
├── uploads/               # Uploaded PDF storage
├── pdfs_to_process/       # Bulk PDF processing
├── logs/                  # Application logs
├── templates/             # HTML templates
└── pdf_database.json      # Persistent PDF metadata
```

## Security Notes

- API keys are stored in environment variables
- Uploaded files are stored locally
- No user authentication implemented (add for production)
- Consider adding rate limiting for production use

## Deployment

### Local Development

```bash
python app.py
```

### Production with Uvicorn

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

## License

This project is licensed under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Support

For issues and questions:

- Check the troubleshooting section
- Review application logs in `logs/app.log`
- Use the debug endpoints for diagnosis
- Create an issue in the repository

## Acknowledgments

- Built with FastAPI for high-performance backend
- Uses Pinecone for vector similarity search
- Supports multiple AI providers for flexibility
- RAG implementation for accurate PDF-based answers

**Note:** This tool is designed for educational and research purposes. Always ensure you have the right to process and analyze the PDF documents you upload.

