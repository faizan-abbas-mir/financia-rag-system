# FinanceRAG - Streamlit UI Guide

## Overview

A modern, interactive Streamlit interface for the FinanceRAG system. This UI provides a user-friendly way to upload financial documents, query them with natural language, and analyze metrics.

## Features

✨ **Core Features:**
- 🔍 **Query Documents** - Natural language search with AI-powered answers
- 📤 **Upload Documents** - Support for PDF, DOCX, and TXT files
- 📊 **Metrics Dashboard** - Real-time performance and system analytics
- 📚 **History Tracking** - View all queries and uploads in the current session

## Installation

### 1. Install Dependencies

```bash
# Install all requirements (including Streamlit)
pip install -r requirements.txt
```

### 2. Configure Environment

Ensure your `.env` file is properly configured with:
```
ANTHROPIC_API_KEY=your_api_key_here
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=financial-documents
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

## Running the Application

### Option 1: Run Everything Together (Recommended for first-time users)

```bash
# Make script executable
chmod +x run.sh

# Run both backend and frontend
./run.sh
```

This will:
1. Start the FastAPI backend on `http://localhost:8000`
2. Start the Streamlit frontend on `http://localhost:8501`

### Option 2: Run Backend and Frontend Separately

**Terminal 1 - Start the Backend:**
```bash
chmod +x run_backend.sh
./run_backend.sh
```
The backend will run on `http://localhost:8000`

**Terminal 2 - Start the Frontend:**
```bash
chmod +x run_frontend.sh
./run_frontend.sh
```
The frontend will run on `http://localhost:8501`

### Option 3: Manual Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Terminal 1: Start backend
cd src
python -m uvicorn main:app --reload

# Terminal 2: Start Streamlit
streamlit run streamlit_app.py
```

## Usage

### 1. Accessing the UI

Once running, open your browser to: **http://localhost:8501**

### 2. Configure API Connection

In the left sidebar:
- Enter the FastAPI URL (default: `http://localhost:8000`)
- Click "Test Connection" to verify connectivity

### 3. Upload Documents

1. Go to the **📤 Upload Documents** tab
2. Select one or multiple files (PDF, DOCX, or TXT)
3. Click **⬆️ Upload & Process**
4. Wait for processing to complete
5. Files will be split into chunks and added to the vector database

**Supported Formats:**
- PDF files (.pdf)
- Word documents (.docx)
- Text files (.txt)

**Tips:**
- Start with smaller documents for faster processing
- Ensure filenames are descriptive
- Financial reports, earnings calls, and SEC filings work best

### 4. Query Documents

1. Go to the **🔍 Query** tab
2. Enter your financial question in natural language
3. Set the number of source documents to retrieve (1-10)
4. Click **🚀 Submit Query**

**Example Questions:**
- "What was the total revenue in Q3 2023?"
- "What are the main risk factors mentioned?"
- "List all debt obligations"
- "What is the company's cash position?"

**Output includes:**
- 💡 AI-generated answer from Claude
- 📖 Source documents with relevance scores
- ⏱️ Performance metrics

### 5. View Metrics

Go to the **📊 Metrics** tab to see:
- 📄 Total documents in the system
- 🔍 Total queries processed
- ⏱️ Average query latency
- ⭐ Average relevance scores
- 🏥 System health status

### 6. Check History

View your session activity in the **📚 History** tab:
- All queries you've submitted
- All files you've uploaded
- Timestamps for each action

## Troubleshooting

### Connection Error: "Connection error. Please check if the FastAPI server is running."

**Solution:**
1. Verify the FastAPI backend is running on port 8000
2. Check the API URL in the sidebar matches your backend URL
3. Run the backend: `./run_backend.sh`

### File Upload Error: "File type ... not supported"

**Solution:**
- Only PDF, DOCX, and TXT files are supported
- Check file extension (case-sensitive)
- Ensure file is not corrupted

### Query Timeout: "Request timeout. The API took too long to respond."

**Solution:**
- Reduce the number of sources (top_k parameter)
- Check if backend is processing other requests
- Ensure vector database has documents loaded
- Try a simpler query

### Module Not Found: "No module named 'streamlit'"

**Solution:**
```bash
pip install streamlit pandas requests
```

### Port Already in Use

If port 8000 or 8501 is already in use:

```bash
# Find process using port 8000 (backend)
lsof -i :8000
kill -9 <PID>

# Find process using port 8501 (frontend)
lsof -i :8501
kill -9 <PID>
```

## Performance Tips

1. **Document Upload:** Break large documents into smaller chunks for faster processing
2. **Queries:** Be specific with questions for better results
3. **Vector Store:** Regularly check metrics to ensure index health
4. **Memory:** For large document collections, ensure adequate system RAM

## Advanced Configuration

### Change API Port

Edit `src/main.py`:
```python
python -m uvicorn main:app --port 9000
```

Then update Streamlit config with new URL.

### Adjust Chunk Size

Edit `.env`:
```
CHUNK_SIZE=1024
CHUNK_OVERLAP=100
```

### Change Embedding Model

Edit `.env`:
```
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L12-v2
```

## Development

### Run with Auto-reload

The `run.sh` script uses `--reload` flag, automatically restarting when code changes.

### View Logs

Streamlit logs appear in the console. Backend logs show in the backend terminal.

### Debug Mode

Add to `streamlit_app.py`:
```python
st.write(st.session_state)  # View all session variables
```

## Architecture

```
┌─────────────────────────────────────┐
│     Streamlit Frontend (8501)       │
├─────────────────────────────────────┤
│  - File Upload Interface            │
│  - Query Input                      │
│  - Results Display                  │
│  - Metrics Dashboard                │
└──────────────┬──────────────────────┘
               │ HTTP Requests
               ▼
┌──────────────────────────────────────────┐
│    FastAPI Backend (8000)                │
├──────────────────────────────────────────┤
│  - Document Processing                   │
│  - Vector Store Management               │
│  - LLM Integration (Claude)              │
│  - Query Processing & RAG                │
└──────────────┬───────────────────────────┘
               │
      ┌────────┴────────────────┐
      ▼                         ▼
  ┌────────────┐         ┌─────────────────┐
  │  Pinecone  │         │  Anthropic API  │
  │ Vector DB  │         │  (Claude Model) │
  └────────────┘         └─────────────────┘
```

## API Endpoints

The Streamlit UI calls these backend endpoints:

- `POST /api/upload` - Upload and process documents
- `POST /api/query` - Query documents
- `GET /api/metrics` - Get system metrics
- `GET /health` - Health check

See `docs/API.md` for detailed endpoint documentation.

## Security Considerations

⚠️ **Important:**
- Never commit `.env` file with real API keys
- Use environment variables for sensitive data
- Validate file uploads in production
- Implement user authentication for shared deployments

## Contributing

To improve the Streamlit UI:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

See `CONTRIBUTING.md` for detailed guidelines.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review backend logs in the other terminal
3. Ensure `.env` is properly configured
4. Check that Pinecone and OpenAI APIs are accessible

## License

MIT License - See LICENSE file for details

---

**Happy Analyzing!** 📊✨
