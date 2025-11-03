import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Try to import Pinecone with the new API
try:
    from pinecone import Pinecone, ServerlessSpec

    PINECONE_NEW_API = True
except ImportError:
    try:
        import pinecone

        PINECONE_NEW_API = False
    except ImportError:
        pinecone = None
        PINECONE_NEW_API = False

# Try to import pypdf (newer) or fall back to PyPDF2
try:
    import pypdf

    PyPDF2 = pypdf  # Alias for compatibility
    logger.info("Using pypdf library")
except ImportError:
    try:
        import PyPDF2

        logger.info("Using PyPDF2 library")
    except ImportError:
        PyPDF2 = None
        logger.error("No PDF library found. Please install pypdf or PyPDF2")


# Model types
class ModelType(str, Enum):
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"


# Initialize FastAPI app
app = FastAPI(
    title="PDF Q&A API with Pinecone VectorDB",
    description="API for uploading multiple PDFs and asking questions using different AI models with Vector Database",
    version="4.5.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Pinecone with new API
pinecone_initialized = False
index = None
pc = None

try:
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    if not pinecone_api_key:
        logger.warning("Pinecone credentials not found. Please set PINECONE_API_KEY")
        pinecone_initialized = False
    else:
        if PINECONE_NEW_API:
            pc = Pinecone(api_key=pinecone_api_key)
            logger.info("Pinecone initialized successfully with new API")
        else:
            pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
            if not pinecone_env:
                logger.warning("PINECONE_ENVIRONMENT not set for old API")
                pinecone_initialized = False
            else:
                pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
                logger.info("Pinecone initialized successfully with old API")

        pinecone_initialized = True

except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {str(e)}")
    pinecone_initialized = False

# Pinecone index configuration - USING text-embedding-ada-002
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") or "pdf-vector-index"
EMBEDDING_DIMENSION = 3072  # text-embedding-ada-002 dimension

# FIXED: Pinecone index initialization - only create if doesn't exist
if pinecone_initialized:
    try:
        if PINECONE_NEW_API:
            existing_indexes = pc.list_indexes()
            index_names = (
                [idx.name for idx in existing_indexes.indexes]
                if hasattr(existing_indexes, "indexes")
                else []
            )

            if INDEX_NAME in index_names:
                # JUST CONNECT TO EXISTING INDEX - DON'T DELETE IT!
                logger.info(f"Connecting to existing Pinecone index: {INDEX_NAME}")
                index = pc.Index(INDEX_NAME)

                # Optional: Verify the dimension matches
                try:
                    stats = index.describe_index_stats()
                    current_dimension = (
                        stats.dimension
                        if hasattr(stats, "dimension")
                        else EMBEDDING_DIMENSION
                    )
                    if current_dimension != EMBEDDING_DIMENSION:
                        logger.warning(
                            f"Index dimension mismatch. Expected {EMBEDDING_DIMENSION}, got {current_dimension}"
                        )
                except Exception as e:
                    logger.warning(f"Could not verify index dimension: {e}")

            else:
                # Only create new index if it doesn't exist
                logger.info(
                    f"Creating new Pinecone index: {INDEX_NAME} with dimension {EMBEDDING_DIMENSION}"
                )
                pc.create_index(
                    name=INDEX_NAME,
                    dimension=EMBEDDING_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                time.sleep(5)
                index = pc.Index(INDEX_NAME)

        else:
            # Old API - same logic
            if INDEX_NAME in pinecone.list_indexes():
                logger.info(f"Connecting to existing Pinecone index: {INDEX_NAME}")
                index = pinecone.Index(INDEX_NAME)
            else:
                logger.info(
                    f"Creating new Pinecone index: {INDEX_NAME} with dimension {EMBEDDING_DIMENSION}"
                )
                pinecone.create_index(
                    name=INDEX_NAME, dimension=EMBEDDING_DIMENSION, metric="cosine"
                )
                time.sleep(5)
                index = pinecone.Index(INDEX_NAME)

        logger.info(f"Connected to Pinecone index: {INDEX_NAME}")

    except Exception as e:
        logger.error(f"Error setting up Pinecone index: {str(e)}")
        pinecone_initialized = False
else:
    index = None

# Persistent storage configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_FILE = os.path.join(BASE_DIR, "pdf_database.json")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
PDFS_TO_PROCESS_DIR = os.path.join(BASE_DIR, "pdfs_to_process")


class PersistentPDFDatabase:
    def __init__(self, storage_file: str):
        self.storage_file = storage_file
        self.data = self._load_data()

    def _load_data(self) -> Dict[str, dict]:
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    logger.info(f"✅ Loaded {len(data)} PDFs from persistent storage")
                    return data
            else:
                logger.info("📝 No existing storage file found, starting fresh")
                return {}
        except Exception as e:
            logger.error(f"❌ Error loading PDF database: {str(e)}")
            return {}

    def save(self):
        try:
            with open(self.storage_file, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Saved {len(self.data)} PDFs to persistent storage")
        except Exception as e:
            logger.error(f"❌ Error saving PDF database: {str(e)}")

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __getitem__(self, key: str) -> dict:
        return self.data[key]

    def __setitem__(self, key: str, value: dict):
        self.data[key] = value
        self.save()

    def __delitem__(self, key: str):
        del self.data[key]
        self.save()

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def __len__(self):
        return len(self.data)


# Initialize persistent storage
pdf_database = PersistentPDFDatabase(STORAGE_FILE)

user_settings: Dict[str, dict] = {
    "default": {
        "system_message": "You are a helpful assistant that answers questions based on provided PDF content. Always use the context provided to answer questions. If the context doesn't contain the answer, say so.",
        "model_type": ModelType.HUGGINGFACE,
        "use_rag": True,
    }
}


# Pydantic Models
class QuestionRequest(BaseModel):
    question: str
    document_id: Optional[str] = None
    model_type: ModelType = ModelType.HUGGINGFACE
    system_message: Optional[str] = None
    use_rag: bool = True


class QuestionResponse(BaseModel):
    question: str
    answer: str
    document_used: str
    source_documents: Optional[List[str]] = None
    model_used: str
    confidence: Optional[float] = None
    system_message_used: str
    rag_enabled: bool


class UploadResponse(BaseModel):
    message: str
    filename: str
    document_id: str
    document_count: int
    model_used: str


class PDFInfo(BaseModel):
    document_id: str
    filename: str
    upload_time: str
    document_count: int
    file_size: int
    file_hash: str
    model_used: str


class MultipleUploadResponse(BaseModel):
    message: str
    uploaded_files: List[dict]
    skipped_files: List[dict]


class DuplicateCheckResponse(BaseModel):
    is_duplicate: bool
    existing_document: Optional[dict] = None
    message: str


class ModelInfo(BaseModel):
    name: str
    type: str
    provider: str
    description: str
    is_available: bool


class SystemMessageRequest(BaseModel):
    system_message: str


class ModelSettingRequest(BaseModel):
    model_type: ModelType


class RAGStatusRequest(BaseModel):
    use_rag: bool


class SettingsResponse(BaseModel):
    system_message: str
    model_type: str
    use_rag: bool


class ProcessDirectoryRequest(BaseModel):
    directory_path: str = "pdfs_to_process"
    model_type: str = "huggingface"


class ProcessDirectoryResponse(BaseModel):
    processed_files: List[dict]
    successful_uploads: int
    failed_uploads: int
    total_vectors: int
    message: str


# Utility Functions
def calculate_file_hash(file_content: bytes) -> str:
    return hashlib.md5(file_content).hexdigest()


def is_pdf_duplicate(file_hash: str, filename: str) -> Optional[str]:
    for doc_id, doc_info in pdf_database.items():
        if doc_info.get("file_hash") == file_hash:
            return doc_id
        if doc_info.get("filename") == filename:
            return doc_id
    return None


def extract_text_from_pdf(file_path: str) -> List[str]:
    start_time = time.time()
    text_chunks = []

    if PyPDF2 is None:
        logger.error("No PDF library available. Please install pypdf or PyPDF2")
        return text_chunks

    try:
        logger.info(f"📄 Extracting text from PDF: {file_path}")

        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)
            logger.info(f"📑 PDF has {total_pages} pages")

            extracted_text = ""
            for page_num in range(total_pages):
                page = reader.pages[page_num]
                text = page.extract_text()

                if text and text.strip():
                    extracted_text += text + " "
                else:
                    logger.warning(
                        f"⚠️ Page {page_num + 1} has no extractable text (might be scanned)"
                    )

            # Check if we got any text
            if not extracted_text.strip():
                logger.error(
                    "❌ No text could be extracted from PDF - might be scanned or image-based"
                )
                return text_chunks

            # Clean and chunk the text
            clean_text = " ".join(extracted_text.split())

            # Split into sentences and create chunks
            sentences = [s.strip() + "." for s in clean_text.split(".") if s.strip()]

            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < 1000:  # Increased chunk size
                    current_chunk += " " + sentence
                else:
                    if current_chunk.strip():
                        text_chunks.append(current_chunk.strip())
                    current_chunk = sentence

            # Add the last chunk
            if current_chunk.strip():
                text_chunks.append(current_chunk.strip())

            extraction_time = time.time() - start_time
            logger.info(
                f"✅ Extracted {len(text_chunks)} chunks ({len(clean_text)} chars) in {extraction_time:.2f}s"
            )

            # Log sample of first chunk for debugging
            if text_chunks:
                sample = (
                    text_chunks[0][:200] + "..."
                    if len(text_chunks[0]) > 200
                    else text_chunks[0]
                )
                logger.info(f"📝 First chunk sample: {sample}")

    except Exception as e:
        logger.error(f"❌ Error extracting text from PDF {file_path}: {e}")

    return text_chunks


# FIXED: Embedding function with fallback
def generate_embeddings(text_chunks: List[str]) -> List[List[float]]:
    """Generate embeddings for text chunks using text-embedding-3-large"""
    # First try: OpenAI embeddings
    try:
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            logger.error("❌ OpenAI API key not configured for embeddings")
            raise ValueError("OpenAI API key not configured")

        embeddings = []
        batch_size = 10  # Smaller batches for 3072-dimension vectors

        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i : i + batch_size]
            logger.info(
                f"🔢 Generating embeddings for batch {i//batch_size + 1}/{(len(text_chunks)-1)//batch_size + 1}"
            )

            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "text-embedding-3-large",  # CHANGED TO 3-LARGE
                    "input": batch,
                    "dimensions": 3072,  # EXPLICITLY SET DIMENSIONS
                },
                timeout=60,  # Increased timeout for larger vectors
            )

            if response.status_code != 200:
                error_msg = f"OpenAI embedding API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                continue

            batch_result = response.json()
            batch_embeddings = [item["embedding"] for item in batch_result["data"]]
            embeddings.extend(batch_embeddings)

            logger.info(f"✅ Generated {len(batch_embeddings)} embeddings for batch")

        if embeddings:
            logger.info(
                f"🎉 Successfully generated {len(embeddings)} text-embedding-3-large embeddings"
            )
            # Verify dimension
            if embeddings and len(embeddings[0]) != EMBEDDING_DIMENSION:
                logger.error(
                    f"❌ Embedding dimension mismatch: expected {EMBEDDING_DIMENSION}, got {len(embeddings[0])}"
                )
            return embeddings
        else:
            raise Exception("No embeddings generated from OpenAI")

    except Exception as e:
        logger.error(f"❌ OpenAI embeddings failed: {str(e)}")

    # Fallback: Simple deterministic embeddings (3072 dimensions)
    logger.warning("🔄 Using simple fallback embeddings")
    try:
        embeddings = []
        for i, chunk in enumerate(text_chunks):
            # Create deterministic embeddings with 3072 dimensions
            vector = [0.0] * EMBEDDING_DIMENSION
            hash_val = hash(chunk) % 10000

            # Fill vector with deterministic values
            for j in range(
                min(200, EMBEDDING_DIMENSION)
            ):  # Fill more dimensions for 3072
                vector[j] = ((hash_val + i + j) % 1000) / 1000.0

            embeddings.append(vector)

        logger.info(f"✅ Generated {len(embeddings)} fallback embeddings (3072-dim)")
        return embeddings

    except Exception as e:
        logger.error(f"❌ Fallback embeddings also failed: {str(e)}")
        return []


def vector_similarity_search(
    question: str, document_id: Optional[str] = None, top_k: int = 5
) -> List[dict]:
    if not pinecone_initialized or index is None:
        logger.error("Pinecone not initialized for vector search")
        return []

    try:
        logger.info(f"🔍 Starting vector search for: '{question}'")

        # Generate question embedding
        question_embeddings = generate_embeddings([question])
        if not question_embeddings:
            logger.error("❌ Failed to generate question embedding")
            return []

        question_embedding = question_embeddings[0]
        logger.info(
            f"✅ Question embedding generated, dimension: {len(question_embedding)}"
        )

        # Build filter
        filter_dict = {}
        if document_id:
            filter_dict = {"document_id": document_id}
            logger.info(f"🔍 Filtering search to document: {document_id}")

        # Perform search
        query_response = index.query(
            vector=question_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict if filter_dict else None,
        )

        results = []
        if hasattr(query_response, "matches"):
            for match in query_response.matches:
                results.append(
                    {
                        "text": match.metadata.get("text", ""),
                        "score": match.score,
                        "document_id": match.metadata.get("document_id", ""),
                        "chunk_index": match.metadata.get("chunk_index", 0),
                        "filename": match.metadata.get("filename", ""),
                    }
                )
        else:
            for match in query_response.get("matches", []):
                results.append(
                    {
                        "text": match.get("metadata", {}).get("text", ""),
                        "score": match.get("score", 0),
                        "document_id": match.get("metadata", {}).get("document_id", ""),
                        "chunk_index": match.get("metadata", {}).get("chunk_index", 0),
                        "filename": match.get("metadata", {}).get("filename", ""),
                    }
                )

        # Filter out low-score results
        filtered_results = [r for r in results if r["score"] > 0.5]
        logger.info(
            f"📊 Vector search found {len(results)} total, {len(filtered_results)} after filtering (score > 0.5)"
        )

        return filtered_results

    except Exception as e:
        logger.error(f"❌ Error in vector similarity search: {str(e)}")
        return []


def simple_keyword_search(
    question: str, text_chunks: List[str], top_k: int = 3
) -> List[dict]:
    logger.info("Using fallback keyword search")
    start_time = time.time()

    question_lower = question.lower()
    question_words = set(question_lower.split())

    scored_chunks = []
    for chunk in text_chunks:
        chunk_lower = chunk.lower()
        chunk_words = set(chunk_lower.split())

        common_words = question_words.intersection(chunk_words)
        score = len(common_words)

        if question_lower in chunk_lower:
            score += 5

        scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    results = [
        {"text": chunk, "score": score / 10.0}
        for score, chunk in scored_chunks[:top_k]
        if score > 0
    ]

    search_time = time.time() - start_time
    logger.info(
        f"Keyword search found {len(results)} results in {search_time:.4f} seconds"
    )

    return results


def get_all_text_chunks() -> List[str]:
    all_chunks = []
    for doc_id, doc_info in pdf_database.items():
        file_path = doc_info.get("file_path")
        if file_path and os.path.exists(file_path):
            try:
                chunks = extract_text_from_pdf(file_path)
                all_chunks.extend(chunks)
                logger.info(
                    f"Extracted {len(chunks)} chunks from {doc_info['filename']}"
                )
            except Exception as e:
                logger.error(f"Error extracting text from {file_path}: {str(e)}")
        else:
            logger.warning(f"File not found for {doc_info['filename']}: {file_path}")

    logger.info(f"Total text chunks available for search: {len(all_chunks)}")
    return all_chunks


def store_vectors_in_pinecone(
    document_id: str, text_chunks: List[str], embeddings: List[List[float]]
) -> bool:
    if not pinecone_initialized or index is None:
        logger.error("❌ Pinecone not initialized, cannot store vectors")
        return False

    try:
        logger.info(
            f"🔄 Starting to store {len(embeddings)} vectors for document {document_id}"
        )

        # Validate inputs
        if len(text_chunks) != len(embeddings):
            logger.error(
                f"❌ Mismatch: {len(text_chunks)} chunks vs {len(embeddings)} embeddings"
            )
            return False

        if not embeddings or not text_chunks:
            logger.error("❌ No embeddings or text chunks to store")
            return False

        # Check embedding dimensions
        embedding_dim = len(embeddings[0])
        if embedding_dim != EMBEDDING_DIMENSION:
            logger.error(
                f"❌ Embedding dimension mismatch: expected {EMBEDDING_DIMENSION}, got {embedding_dim}"
            )
            return False

        vectors = []
        successful_vectors = 0

        for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
            vector_id = f"{document_id}_{i}"

            # Validate embedding
            if len(embedding) != EMBEDDING_DIMENSION:
                logger.warning(
                    f"⚠️ Skipping vector {i} - wrong dimension: {len(embedding)}"
                )
                continue

            if not all(isinstance(x, (int, float)) for x in embedding):
                logger.warning(f"⚠️ Skipping vector {i} - invalid values")
                continue

            vectors.append(
                {
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "text": chunk[:1000],  # Limit text length
                        "document_id": document_id,
                        "chunk_index": i,
                        "timestamp": datetime.now().isoformat(),
                        "filename": pdf_database[document_id]["filename"],
                    },
                }
            )

        if not vectors:
            logger.error("❌ No valid vectors to store")
            return False

        logger.info(f"📦 Prepared {len(vectors)} vectors for upsert")

        # Upload in smaller batches for better reliability
        batch_size = 50
        total_uploaded = 0

        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            try:
                logger.info(
                    f"📤 Uploading batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}"
                )

                # Upsert with timeout
                upsert_response = index.upsert(
                    vectors=batch, namespace=""  # Explicitly use default namespace
                )

                total_uploaded += len(batch)
                logger.info(f"✅ Successfully uploaded batch of {len(batch)} vectors")

                # Small delay between batches
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"❌ Failed to upload batch {i//batch_size + 1}: {str(e)}")
                # Continue with next batch instead of failing completely

        logger.info(
            f"🎉 Successfully stored {total_uploaded}/{len(vectors)} vectors for document {document_id}"
        )

        # Verify upload by checking index stats
        time.sleep(2)  # Wait for index to update
        try:
            stats = index.describe_index_stats()
            total_vectors = (
                stats.total_vector_count if hasattr(stats, "total_vector_count") else 0
            )
            logger.info(f"📊 Pinecone index now has {total_vectors} total vectors")
        except Exception as e:
            logger.warning(f"⚠️ Could not verify index stats: {str(e)}")

        return total_uploaded > 0

    except Exception as e:
        logger.error(f"❌ Error storing vectors in Pinecone: {str(e)}")
        return False


def delete_vectors_from_pinecone(document_id: str) -> bool:
    if not pinecone_initialized or index is None:
        return False

    try:
        index.delete(filter={"document_id": document_id})
        logger.info(f"Deleted vectors for document {document_id}")
        return True
    except Exception as e:
        logger.error(f"Error deleting vectors from Pinecone: {str(e)}")
        return False


# PDF Processing and Storage Functions
def process_pdf_directory(
    directory_path: str = "pdfs_to_process", model_type: str = "huggingface"
) -> Dict:
    """
    Process all PDFs in a directory and populate Pinecone index
    """
    if not os.path.exists(directory_path):
        return {"error": f"Directory {directory_path} does not exist"}

    if not pinecone_initialized:
        return {"error": "Pinecone is not initialized"}

    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith(".pdf")]

    if not pdf_files:
        return {"error": f"No PDF files found in {directory_path}"}

    results = {
        "processed_files": [],
        "successful_uploads": 0,
        "failed_uploads": 0,
        "total_vectors": 0,
    }

    logger.info(f"📁 Found {len(pdf_files)} PDF files in {directory_path}")

    for pdf_file in pdf_files:
        file_path = os.path.join(directory_path, pdf_file)
        logger.info(f"🔄 Processing: {pdf_file}")

        try:
            # Check if file already exists in database
            with open(file_path, "rb") as f:
                file_content = f.read()
                file_hash = calculate_file_hash(file_content)

            # Check for duplicates
            existing_doc_id = None
            for doc_id, doc_info in pdf_database.items():
                if doc_info.get("file_hash") == file_hash:
                    existing_doc_id = doc_id
                    break

            if existing_doc_id:
                logger.info(f"⏭️  Skipping duplicate: {pdf_file}")
                results["processed_files"].append(
                    {
                        "filename": pdf_file,
                        "status": "duplicate",
                        "document_id": existing_doc_id,
                    }
                )
                continue

            # Generate unique document ID
            document_id = str(uuid.uuid4())
            unique_filename = f"{document_id}.pdf"
            destination_path = os.path.join(UPLOADS_DIR, unique_filename)

            # Copy file to uploads directory
            with open(destination_path, "wb") as f:
                f.write(file_content)

            # Extract text from PDF
            text_chunks = extract_text_from_pdf(destination_path)

            if not text_chunks:
                logger.error(f"❌ No text extracted from: {pdf_file}")
                results["processed_files"].append(
                    {
                        "filename": pdf_file,
                        "status": "failed",
                        "error": "No text extracted",
                    }
                )
                results["failed_uploads"] += 1

                # Clean up the copied file
                if os.path.exists(destination_path):
                    os.remove(destination_path)
                continue

            logger.info(f"📝 Extracted {len(text_chunks)} text chunks")

            # Generate embeddings
            embeddings = generate_embeddings(text_chunks)

            if not embeddings:
                logger.error(f"❌ Failed to generate embeddings for: {pdf_file}")
                results["processed_files"].append(
                    {
                        "filename": pdf_file,
                        "status": "failed",
                        "error": "Embedding generation failed",
                    }
                )
                results["failed_uploads"] += 1
                continue

            logger.info(f"🔢 Generated {len(embeddings)} embeddings")

            # Store vectors in Pinecone
            success = store_vectors_in_pinecone(document_id, text_chunks, embeddings)

            if not success:
                logger.error(f"❌ Failed to store vectors for: {pdf_file}")
                results["processed_files"].append(
                    {
                        "filename": pdf_file,
                        "status": "failed",
                        "error": "Vector storage failed",
                    }
                )
                results["failed_uploads"] += 1
                continue

            # Store metadata in database
            pdf_database[document_id] = {
                "filename": pdf_file,
                "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "document_count": len(text_chunks),
                "file_path": destination_path,
                "file_hash": file_hash,
                "file_size": len(file_content),
                "model_used": model_type,
                "has_vectors": True,
            }

            logger.info(f"✅ Successfully processed: {pdf_file}")
            results["processed_files"].append(
                {
                    "filename": pdf_file,
                    "status": "success",
                    "document_id": document_id,
                    "chunks": len(text_chunks),
                    "vectors": len(embeddings),
                }
            )
            results["successful_uploads"] += 1
            results["total_vectors"] += len(embeddings)

        except Exception as e:
            logger.error(f"❌ Error processing {pdf_file}: {str(e)}")
            results["processed_files"].append(
                {"filename": pdf_file, "status": "failed", "error": str(e)}
            )
            results["failed_uploads"] += 1

    return results


def check_pinecone_status():
    """Check Pinecone index status"""
    if not pinecone_initialized:
        return {"error": "Pinecone not initialized"}

    try:
        stats = index.describe_index_stats()
        total_vectors = (
            stats.total_vector_count if hasattr(stats, "total_vector_count") else 0
        )

        return {
            "status": "connected",
            "index_name": INDEX_NAME,
            "dimension": EMBEDDING_DIMENSION,
            "total_vectors": total_vectors,
        }
    except Exception as e:
        return {"error": f"Failed to get Pinecone status: {str(e)}"}


# FIXED: Model Calling Functions
def get_fallback_answer(question: str, context: str = "") -> str:
    """Simple fallback when no APIs work"""
    if context:
        return f"I found some relevant information in your PDFs, but the AI service is currently unavailable. Here's what I found:\n\n{context[:500]}..."
    else:
        simple_answers = {
            "hello": "Hello! I'm here to help with your PDF questions.",
            "hi": "Hi there! How can I assist you today?",
            "thank you": "You're welcome! Let me know if you have more questions.",
            "what can you do": "I can help you analyze PDF documents. Upload a PDF and ask questions about its content!",
        }
        return simple_answers.get(
            question.lower(),
            "I'm currently experiencing issues with AI services. Please check your API configuration and try again.",
        )


def call_azure_openai(question: str, context: str, system_message: str) -> str:
    """Call Azure OpenAI API - IMPROVED CONTEXT HANDLING"""
    logger.info("🔄 Starting Azure OpenAI API call")

    try:
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = "2023-12-01-preview"
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

        if not all([api_key, api_base, deployment_name]):
            missing = []
            if not api_key:
                missing.append("API_KEY")
            if not api_base:
                missing.append("ENDPOINT")
            if not deployment_name:
                missing.append("DEPLOYMENT_NAME")
            logger.error(f"❌ Azure OpenAI missing: {', '.join(missing)}")
            return f"Azure OpenAI not configured. Missing: {', '.join(missing)}"

        url = f"{api_base.rstrip('/')}/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}"

        headers = {"Content-Type": "application/json", "api-key": api_key}

        # FIXED: Better context formatting
        messages = [{"role": "system", "content": system_message}]

        if context and context.strip():
            user_content = f"""Please answer the following question based EXCLUSIVELY on the provided context from PDF documents. If the context doesn't contain the answer, say you cannot find the answer in the provided documents.

CONTEXT FROM PDFS:
{context}

QUESTION: {question}

ANSWER BASED ON CONTEXT:"""
        else:
            user_content = f"QUESTION: {question}"

        messages.append({"role": "user", "content": user_content})

        data = {
            "messages": messages,
            "max_tokens": 800,
            "temperature": 0.1,
        }

        logger.info(f"📤 Calling Azure OpenAI: {deployment_name}")
        response = requests.post(url, headers=headers, json=data, timeout=30)

        logger.info(f"📥 Response status: {response.status_code}")

        if response.status_code != 200:
            error_text = response.text[:500]  # Limit error text length
            return f"Azure OpenAI API error: {response.status_code} - {error_text}"

        result = response.json()
        answer = result["choices"][0]["message"]["content"].strip()

        logger.info("✅ Azure OpenAI call successful")
        return answer

    except Exception as e:
        logger.error(f"❌ Azure OpenAI error: {str(e)}")
        return f"Error calling Azure OpenAI: {str(e)}"


def call_openai(question: str, context: str, system_message: str) -> str:
    """Call OpenAI API - FIXED VERSION"""
    logger.info("🔄 Starting OpenAI API call")

    try:
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            logger.error("❌ OpenAI API key not set")
            return "OpenAI API key not configured."

        messages = [{"role": "system", "content": system_message}]

        if context and context.strip():
            user_content = f"""Please answer the following question based EXCLUSIVELY on the provided context from PDF documents. If the context doesn't contain the answer, say you cannot find the answer in the provided documents.

CONTEXT FROM PDFS:
{context}

QUESTION: {question}

ANSWER BASED ON CONTEXT:"""
        else:
            user_content = f"QUESTION: {question}"

        messages.append({"role": "user", "content": user_content})

        data = {
            "model": "gpt-3.5-turbo",
            "messages": messages,
            "max_tokens": 800,
            "temperature": 0.1,
        }

        logger.info("📤 Calling OpenAI API")
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json=data,
            timeout=30,
        )

        logger.info(f"📥 Response status: {response.status_code}")

        if response.status_code != 200:
            return f"OpenAI API error: {response.status_code} - {response.text}"

        result = response.json()
        answer = result["choices"][0]["message"]["content"].strip()

        logger.info("✅ OpenAI call successful")
        return answer

    except Exception as e:
        logger.error(f"❌ OpenAI error: {str(e)}")
        return f"Error calling OpenAI: {str(e)}"


def call_huggingface(question: str, context: str, system_message: str) -> str:
    """Call Hugging Face API - SIMPLIFIED VERSION"""
    logger.info("🔄 Starting Hugging Face API call")

    try:
        api_key = os.getenv("HUGGINGFACE_API_KEY")

        if not api_key:
            logger.warning("❌ Hugging Face API key not set")
            return "Hugging Face API key not configured."

        # Use a free model that doesn't require payment
        model_url = "https://api-inference.huggingface.co/models/google/flan-t5-large"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # FIXED: Better prompt formatting for context
        if context and context.strip():
            prompt = f"""Answer this question based ONLY on the provided context. If the context doesn't contain the answer, say "I cannot find the answer in the provided documents."

Context: {context}

Question: {question}

Answer:"""
        else:
            prompt = f"""Question: {question}

Answer:"""

        data = {
            "inputs": prompt,
            "parameters": {
                "max_length": 300,
                "temperature": 0.7,
                "do_sample": True,
            },
            "options": {"wait_for_model": True, "use_cache": True},
        }

        logger.info("📤 Calling Hugging Face API")
        response = requests.post(model_url, headers=headers, json=data, timeout=60)

        logger.info(f"📥 Response status: {response.status_code}")

        if response.status_code == 503:
            return "Hugging Face model is loading, please try again in 30 seconds."
        elif response.status_code != 200:
            return f"Hugging Face API error: {response.status_code}"

        result = response.json()

        # Extract answer from different response formats
        if isinstance(result, list) and len(result) > 0:
            if "generated_text" in result[0]:
                answer = result[0]["generated_text"]
            else:
                answer = str(result[0])
        elif isinstance(result, dict) and "generated_text" in result:
            answer = result["generated_text"]
        else:
            answer = str(result)

        # Clean up the answer
        if answer.startswith(prompt):
            answer = answer[len(prompt) :].strip()

        logger.info("✅ Hugging Face call successful")
        return answer.strip() if answer else "No response generated."

    except Exception as e:
        logger.error(f"❌ Hugging Face error: {str(e)}")
        return f"Error calling Hugging Face: {str(e)}"


def call_model(
    question: str, context: str, model_type: ModelType, system_message: str
) -> str:
    """Call the appropriate model with enhanced logging"""
    logger.info(f"🎯 Calling model: {model_type.value}")

    try:
        if model_type == ModelType.AZURE_OPENAI:
            result = call_azure_openai(question, context, system_message)
        elif model_type == ModelType.OPENAI:
            result = call_openai(question, context, system_message)
        else:
            result = call_huggingface(question, context, system_message)

        # Check if the result is an error message
        if any(
            error_indicator in result.lower()
            for error_indicator in ["error", "not configured", "unavailable", "failed"]
        ):
            logger.warning(f"⚠️ Model returned error, using fallback: {result}")
            return get_fallback_answer(question, context)

        return result

    except Exception as e:
        logger.error(f"❌ All models failed: {str(e)}")
        return get_fallback_answer(question, context)


def calculate_confidence(search_results: List[dict]) -> float:
    if not search_results:
        return 0.0

    avg_score = sum(result["score"] for result in search_results) / len(search_results)
    confidence = min(max(avg_score, 0.0), 1.0)
    return confidence


def get_system_message(user_id: str = "default") -> str:
    return user_settings.get(user_id, user_settings["default"])["system_message"]


def generate_answer(
    question: str,
    search_results: List[dict],
    model_type: ModelType,
    system_message: str,
    use_rag: bool,
) -> tuple[str, float]:
    start_time = time.time()

    # FIXED: Better context handling
    context = ""
    source_texts = []

    if use_rag and search_results:
        # Build context from search results
        context_parts = []
        for i, result in enumerate(search_results[:3]):  # Use top 3 results
            context_parts.append(f"[Document {i+1}]: {result['text']}")
            source_texts.append(result["text"])

        context = "\n\n".join(context_parts)
        logger.info(f"📚 Using {len(search_results)} context chunks for RAG")

    elif use_rag and not search_results:
        # No results found in RAG mode
        return (
            "I couldn't find relevant information in the uploaded PDFs to answer your question. This could be because:\n\n"
            "1. The PDF doesn't contain information related to your question\n"
            "2. The PDF text might not be extractable (scanned PDF)\n"
            "3. Try rephrasing your question or asking about different topics\n\n"
            "You can also try disabling RAG mode to use the AI's general knowledge.",
            0.1,
        )

    if use_rag:
        confidence = calculate_confidence(search_results) if search_results else 0.1
    else:
        confidence = 0.9  # Higher confidence for direct model queries
        context = ""  # Ensure no context for non-RAG mode

    try:
        # FIXED: Pass context properly
        answer = call_model(question, context, model_type, system_message)

        if answer and not any(
            error in answer.lower()
            for error in ["error", "not configured", "unavailable"]
        ):
            generation_time = time.time() - start_time
            mode = "RAG" if use_rag else "Direct"
            logger.info(
                f"✅ {model_type.value} {mode} answer completed in {generation_time:.4f}s"
            )
            return answer, confidence
        else:
            # Model returned error, use fallback
            raise Exception(f"Model returned error: {answer}")

    except Exception as e:
        logger.warning(f"⚠️ {model_type.value} API failed: {str(e)}")

        # FIXED: Better fallback responses
        if use_rag and source_texts:
            # Create answer from source texts
            answer = f"Based on the PDF content, here's what I found relevant to your question '{question}':\n\n"
            for i, text in enumerate(source_texts[:3], 1):
                answer += f"{i}. {text}\n\n"
            answer += "This information is extracted directly from your uploaded PDF document(s)."
            return answer, confidence * 0.7
        else:
            return (
                "I'm currently having trouble accessing the AI services. Please check your API configuration or try again later.",
                0.0,
            )


def check_model_availability() -> Dict[str, bool]:
    """Check which models are available with detailed logging"""
    logger.info("🔍 Checking model availability...")

    azure_openai_available = bool(
        os.getenv("AZURE_OPENAI_API_KEY")
        and os.getenv("AZURE_OPENAI_ENDPOINT")
        and os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    )
    openai_available = bool(os.getenv("OPENAI_API_KEY"))
    huggingface_available = bool(os.getenv("HUGGINGFACE_API_KEY"))

    # Log detailed status for each model
    logger.info(
        f"🤖 Azure OpenAI: {'✅ Available' if azure_openai_available else '❌ Not configured'}"
    )
    logger.info(
        f"🤖 OpenAI: {'✅ Available' if openai_available else '❌ Not configured'}"
    )
    logger.info(
        f"🤖 Hugging Face: {'✅ Available' if huggingface_available else '❌ Not configured'}"
    )
    logger.info(
        f"📊 Pinecone: {'✅ Connected' if pinecone_initialized else '❌ Disconnected'}"
    )

    return {
        "huggingface": huggingface_available,
        "openai": openai_available,
        "azure_openai": azure_openai_available,
        "pinecone": pinecone_initialized,
    }


# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(
            content="<html><body><h1>PDF Q&A Application</h1><p>Please create templates/index.html</p></body></html>"
        )


@app.get("/api/health")
async def health_check():
    model_availability = check_model_availability()

    pinecone_stats = None
    if pinecone_initialized and index is not None:
        try:
            pinecone_stats = index.describe_index_stats()
        except Exception as e:
            logger.error(f"Error getting Pinecone stats: {str(e)}")

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "pdf_count": len(pdf_database),
        "model_availability": model_availability,
        "pinecone_status": "connected" if pinecone_initialized else "disconnected",
        "pinecone_api": "new" if PINECONE_NEW_API else "old",
        "pinecone_stats": pinecone_stats,
        "storage_file": STORAGE_FILE,
        "storage_file_exists": os.path.exists(STORAGE_FILE),
    }


# Debug endpoints
@app.get("/api/debug/config")
async def debug_config():
    """Debug environment configuration"""
    config = {
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "azure_configured": bool(
            os.getenv("AZURE_OPENAI_API_KEY")
            and os.getenv("AZURE_OPENAI_ENDPOINT")
            and os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        ),
        "huggingface_configured": bool(os.getenv("HUGGINGFACE_API_KEY")),
        "pinecone_configured": bool(os.getenv("PINECONE_API_KEY")),
        "embedding_dimension": EMBEDDING_DIMENSION,
    }
    return config


@app.post("/api/debug/test-embedding")
async def debug_test_embedding():
    """Test embedding generation"""
    test_text = ["This is a test sentence for embedding."]

    try:
        embeddings = generate_embeddings(test_text)
        return {
            "success": len(embeddings) > 0,
            "embedding_count": len(embeddings),
            "dimension": len(embeddings[0]) if embeddings else 0,
            "sample_embedding": embeddings[0][:5] if embeddings else None,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/debug/test-model")
async def debug_test_model(model_type: str = "huggingface"):
    """Test a specific model"""
    test_question = "What is 2+2?"

    try:
        if model_type == "azure_openai":
            answer = call_azure_openai(
                test_question, "", "You are a helpful assistant."
            )
        elif model_type == "openai":
            answer = call_openai(test_question, "", "You are a helpful assistant.")
        else:
            answer = call_huggingface(test_question, "", "You are a helpful assistant.")

        return {
            "model": model_type,
            "question": test_question,
            "answer": answer,
            "success": not answer.startswith("Error")
            and not answer.startswith("Azure")
            and not answer.startswith("Hugging")
            and not answer.startswith("not configured"),
        }
    except Exception as e:
        return {"model": model_type, "success": False, "error": str(e)}


@app.get("/api/debug/vector-count")
async def debug_vector_count():
    """Check how many vectors are stored in Pinecone"""
    if not pinecone_initialized or index is None:
        return {"error": "Pinecone not initialized"}

    try:
        stats = index.describe_index_stats()
        total_vectors = (
            stats.total_vector_count if hasattr(stats, "total_vector_count") else 0
        )

        return {
            "total_vectors": total_vectors,
            "index_name": INDEX_NAME,
            "dimension": EMBEDDING_DIMENSION,
        }
    except Exception as e:
        return {"error": f"Failed to get vector count: {str(e)}"}


@app.post("/api/debug/test-rag")
async def debug_test_rag(
    question: str = "What is the main topic?", document_id: Optional[str] = None
):
    """Test the RAG pipeline end-to-end"""
    try:
        logger.info("🧪 Testing RAG pipeline...")

        # Step 1: Vector search
        search_results = vector_similarity_search(question, document_id, top_k=3)

        # Step 2: Build context
        context = ""
        if search_results:
            context_parts = []
            for i, result in enumerate(search_results):
                context_parts.append(
                    f"[Result {i+1}, Score: {result['score']:.3f}]: {result['text']}"
                )
            context = "\n\n".join(context_parts)

        # Step 3: Test model with context
        test_answer = call_model(
            question,
            context,
            ModelType.HUGGINGFACE,
            "Answer based on the provided context.",
        )

        return {
            "question": question,
            "search_results_count": len(search_results),
            "search_results": [
                {
                    "score": r["score"],
                    "text_preview": (
                        r["text"][:100] + "..." if len(r["text"]) > 100 else r["text"]
                    ),
                    "document_id": r["document_id"],
                    "filename": r.get("filename", "unknown"),
                }
                for r in search_results
            ],
            "context_length": len(context),
            "context_preview": context[:200] + "..." if len(context) > 200 else context,
            "model_response": test_answer,
            "rag_working": len(search_results) > 0
            and "cannot find" not in test_answer.lower(),
        }

    except Exception as e:
        logger.error(f"❌ RAG test failed: {str(e)}")
        return {"error": str(e)}


# PDF Processing and Storage Endpoints
@app.post("/api/process-directory", response_model=ProcessDirectoryResponse)
async def process_directory(request: ProcessDirectoryRequest):
    """Process all PDFs in a directory and populate Pinecone"""
    try:
        logger.info(f"🚀 Processing directory: {request.directory_path}")

        # Ensure the directory exists
        if not os.path.exists(request.directory_path):
            # Try relative path
            request.directory_path = os.path.join(BASE_DIR, request.directory_path)
            if not os.path.exists(request.directory_path):
                raise HTTPException(
                    status_code=400,
                    detail=f"Directory {request.directory_path} does not exist",
                )

        results = process_pdf_directory(request.directory_path, request.model_type)

        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])

        message = f"Processed {results['successful_uploads']} PDFs successfully, {results['failed_uploads']} failed. Added {results['total_vectors']} vectors to Pinecone."

        return ProcessDirectoryResponse(
            processed_files=results["processed_files"],
            successful_uploads=results["successful_uploads"],
            failed_uploads=results["failed_uploads"],
            total_vectors=results["total_vectors"],
            message=message,
        )

    except Exception as e:
        logger.error(f"Error processing directory: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing directory: {str(e)}"
        )


@app.get("/api/bulk-status")
async def get_bulk_processing_status():
    """Get status of bulk processing directory"""
    pdf_files = []

    if os.path.exists(PDFS_TO_PROCESS_DIR):
        pdf_files = [
            f for f in os.listdir(PDFS_TO_PROCESS_DIR) if f.lower().endswith(".pdf")
        ]

    pinecone_status = check_pinecone_status()

    return {
        "pdfs_to_process_directory": PDFS_TO_PROCESS_DIR,
        "directory_exists": os.path.exists(PDFS_TO_PROCESS_DIR),
        "pdf_files_count": len(pdf_files),
        "pdf_files": pdf_files,
        "pinecone_status": pinecone_status,
        "database_pdfs_count": len(pdf_database),
        "uploads_directory": UPLOADS_DIR,
        "uploads_exists": os.path.exists(UPLOADS_DIR),
    }


@app.post("/api/setup-directories")
async def setup_directories():
    """Create necessary directories for PDF processing"""
    try:
        directories = [UPLOADS_DIR, PDFS_TO_PROCESS_DIR, "logs", "templates"]

        created = []
        existing = []

        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                created.append(directory)
            else:
                existing.append(directory)

        return {
            "message": "Directories setup completed",
            "created": created,
            "existing": existing,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error setting up directories: {str(e)}"
        )


@app.get("/api/debug/storage")
async def debug_storage():
    """Debug endpoint for storage system"""
    storage_info = {
        "storage_file": STORAGE_FILE,
        "storage_file_exists": os.path.exists(STORAGE_FILE),
        "uploads_dir": UPLOADS_DIR,
        "uploads_dir_exists": os.path.exists(UPLOADS_DIR),
        "pdfs_to_process_dir": PDFS_TO_PROCESS_DIR,
        "pdfs_to_process_exists": os.path.exists(PDFS_TO_PROCESS_DIR),
        "pdf_count": len(pdf_database),
        "stored_pdfs": list(pdf_database.keys()),
        "storage_file_size": (
            os.path.getsize(STORAGE_FILE) if os.path.exists(STORAGE_FILE) else 0
        ),
    }

    # Check file existence for each PDF
    file_status = {}
    for doc_id, doc_info in pdf_database.items():
        file_path = doc_info.get("file_path")
        file_status[doc_id] = {
            "filename": doc_info.get("filename"),
            "file_path": file_path,
            "file_exists": os.path.exists(file_path) if file_path else False,
            "file_size": (
                os.path.getsize(file_path)
                if file_path and os.path.exists(file_path)
                else 0
            ),
        }

    storage_info["file_status"] = file_status
    return storage_info


@app.get("/api/debug/pinecone")
async def debug_pinecone():
    """Debug endpoint to check Pinecone status"""
    if not pinecone_initialized:
        return {"status": "pinecone_not_initialized"}

    try:
        stats = index.describe_index_stats()
        sample_results = vector_similarity_search("test", top_k=2)

        return {
            "status": "connected",
            "pinecone_api": "new" if PINECONE_NEW_API else "old",
            "index_stats": stats,
            "sample_query_results": len(sample_results),
            "total_vectors": (
                stats.total_vector_count
                if hasattr(stats, "total_vector_count")
                else stats.get("total_vector_count", 0)
            ),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/api/debug/upload-test")
async def debug_upload_test(file: UploadFile = File(...)):
    """Debug endpoint to test file upload without processing"""
    try:
        content = await file.read()

        uploads_info = {
            "exists": os.path.exists(UPLOADS_DIR),
            "writable": (
                os.access(UPLOADS_DIR, os.W_OK)
                if os.path.exists(UPLOADS_DIR)
                else False
            ),
            "path": UPLOADS_DIR,
        }

        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size": len(content),
            "uploads_info": uploads_info,
            "file_hash": calculate_file_hash(content),
            "is_pdf": (
                file.filename.lower().endswith(".pdf") if file.filename else False
            ),
        }
    except Exception as e:
        logger.error(f"Debug upload test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload test failed: {str(e)}")


@app.get("/api/models")
async def get_available_models():
    """Get available AI models with their status"""
    model_availability = check_model_availability()

    models = [
        {
            "name": "Hugging Face",
            "type": ModelType.HUGGINGFACE,
            "provider": "Hugging Face",
            "description": "Open-source models running via API",
            "is_available": model_availability["huggingface"],
        },
        {
            "name": "OpenAI GPT",
            "type": ModelType.OPENAI,
            "provider": "OpenAI",
            "description": "GPT models via OpenAI API",
            "is_available": model_availability["openai"],
        },
        {
            "name": "Azure OpenAI",
            "type": ModelType.AZURE_OPENAI,
            "provider": "Microsoft Azure",
            "description": "GPT models via Azure OpenAI service",
            "is_available": model_availability["azure_openai"],
        },
    ]

    return {"models": models}


@app.get("/api/settings")
async def get_current_settings(user_id: str = "default"):
    """Get current user settings"""
    settings = user_settings.get(user_id, user_settings["default"])
    return SettingsResponse(
        system_message=settings["system_message"],
        model_type=settings["model_type"],
        use_rag=settings["use_rag"],
    )


@app.post("/api/system-message")
async def update_system_message(
    request: SystemMessageRequest, user_id: str = "default"
):
    """Update the system message for the user"""
    if user_id not in user_settings:
        user_settings[user_id] = user_settings["default"].copy()

    user_settings[user_id]["system_message"] = request.system_message
    logger.info(f"Updated system message for user {user_id}")
    return {
        "message": "System message updated successfully",
        "system_message": request.system_message,
    }


@app.post("/api/model-setting")
async def update_model_setting(request: ModelSettingRequest, user_id: str = "default"):
    """Update the model type for the user"""
    if user_id not in user_settings:
        user_settings[user_id] = user_settings["default"].copy()

    user_settings[user_id]["model_type"] = request.model_type
    logger.info(f"Updated model type for user {user_id}: {request.model_type.value}")
    return {
        "message": f"Model updated to {request.model_type.value}",
        "model_type": request.model_type.value,
    }


@app.post("/api/rag-status")
async def update_rag_status(request: RAGStatusRequest, user_id: str = "default"):
    """Update the RAG status for the user"""
    if user_id not in user_settings:
        user_settings[user_id] = user_settings["default"].copy()

    user_settings[user_id]["use_rag"] = request.use_rag
    status = "enabled" if request.use_rag else "disabled"
    logger.info(f"Updated RAG status for user {user_id}: {status}")
    return {"message": f"RAG {status} successfully", "use_rag": request.use_rag}


@app.post("/api/reset-system-message")
async def reset_system_message(user_id: str = "default"):
    """Reset system message to default"""
    default_message = "You are a helpful assistant that answers questions based on provided PDF content. Always use the context provided to answer questions. If the context doesn't contain the answer, say so."

    if user_id not in user_settings:
        user_settings[user_id] = user_settings["default"].copy()

    user_settings[user_id]["system_message"] = default_message
    logger.info(f"Reset system message for user {user_id}")
    return {
        "message": "System message reset to default",
        "system_message": default_message,
    }


@app.post("/api/check-duplicate")
async def check_duplicate(file: UploadFile = File(...)):
    """Check if a PDF is already uploaded"""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        content = await file.read()
        file_hash = calculate_file_hash(content)
        existing_doc_id = is_pdf_duplicate(file_hash, file.filename)

        if existing_doc_id:
            existing_doc = pdf_database[existing_doc_id]
            return DuplicateCheckResponse(
                is_duplicate=True,
                existing_document={
                    "document_id": existing_doc_id,
                    "filename": existing_doc["filename"],
                    "upload_time": existing_doc["upload_time"],
                    "document_count": existing_doc["document_count"],
                    "file_size": existing_doc.get("file_size", 0),
                    "model_used": existing_doc.get("model_used", "huggingface"),
                },
                message=f"PDF '{file.filename}' is already uploaded",
            )
        else:
            return DuplicateCheckResponse(
                is_duplicate=False, message="PDF is not uploaded yet"
            )

    except Exception as e:
        logger.error(f"Error checking duplicate: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error checking duplicate: {str(e)}"
        )


# FIXED: Single file upload endpoint
@app.post("/api/upload-pdf", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...), model_type: str = Form("huggingface")
):
    """Upload a single PDF file"""
    logger.info(f"Upload request received: {file.filename}")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        # Create uploads directory
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        logger.info(f"Uploads directory: {UPLOADS_DIR}")

        # Read file content
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        logger.info(f"Read {len(content)} bytes from {file.filename}")

        # Check for duplicates
        file_hash = calculate_file_hash(content)
        existing_doc_id = is_pdf_duplicate(file_hash, file.filename)
        if existing_doc_id:
            logger.info(f"Duplicate PDF detected: {file.filename}")
            existing_doc = pdf_database[existing_doc_id]
            return UploadResponse(
                message="PDF already uploaded (duplicate detected)",
                filename=file.filename,
                document_id=existing_doc_id,
                document_count=existing_doc["document_count"],
                model_used=existing_doc.get("model_used", "huggingface"),
            )

        # Generate unique document ID and save file
        document_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{document_id}{file_extension}"
        file_path = os.path.join(UPLOADS_DIR, unique_filename)

        try:
            with open(file_path, "wb") as buffer:
                buffer.write(content)
            logger.info(
                f"Successfully saved PDF file: {file_path} ({len(content)} bytes)"
            )
        except Exception as e:
            logger.error(f"Failed to save file {file_path}: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to save file: {str(e)}"
            )

        # Extract text from PDF
        text_chunks = extract_text_from_pdf(file_path)
        logger.info(f"Extracted {len(text_chunks)} text chunks from {file.filename}")

        if not text_chunks:
            # Clean up the saved file
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from PDF. The PDF might be scanned, contain only images, or be corrupted.",
            )

        # Store basic metadata
        pdf_database[document_id] = {
            "filename": file.filename,
            "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "document_count": len(text_chunks),
            "file_path": file_path,
            "file_hash": file_hash,
            "file_size": len(content),
            "model_used": model_type,
            "has_vectors": False,
        }

        # Generate embeddings and store in Pinecone
        if pinecone_initialized:
            try:
                logger.info("Generating embeddings...")
                embeddings = generate_embeddings(text_chunks)
                logger.info(f"Generated {len(embeddings)} embeddings")

                if embeddings and len(embeddings) == len(text_chunks):
                    success = store_vectors_in_pinecone(
                        document_id, text_chunks, embeddings
                    )
                    pdf_database[document_id]["has_vectors"] = success
                    logger.info(
                        f"Vector storage {'succeeded' if success else 'failed'}"
                    )
                else:
                    logger.error("Embeddings generation failed - mismatched count")
                    pdf_database[document_id]["has_vectors"] = False

            except Exception as e:
                logger.error(f"Failed to store vectors in Pinecone: {str(e)}")
                pdf_database[document_id]["has_vectors"] = False
        else:
            logger.warning("Pinecone not initialized")
            pdf_database[document_id]["has_vectors"] = False

        logger.info(f"Successfully processed PDF: {file.filename}")

        return UploadResponse(
            message="PDF uploaded and processed successfully"
            + (
                " with vector storage"
                if pdf_database[document_id]["has_vectors"]
                else " (vector storage not available)"
            ),
            filename=file.filename,
            document_id=document_id,
            document_count=len(text_chunks),
            model_used=model_type,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF {file.filename}: {str(e)}")
        # Clean up any partially saved files
        if "file_path" in locals() and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up partial file: {file_path}")
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/api/ask-question", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest, user_id: str = "default"):
    """Ask a question about PDFs"""
    try:
        system_message = (
            request.system_message
            if request.system_message
            else get_system_message(user_id)
        )

        # If RAG is disabled, ensure the system message doesn't force reliance on PDF context.
        # Some system messages instruct the model to always use provided PDF context; when
        # no context is supplied (RAG disabled) that can cause the model to ask for the PDF
        # instead of answering. Replace with a general assistant prompt for Direct mode.
        if not request.use_rag:
            lower_sys = (system_message or "").lower()
            if "pdf" in lower_sys or "context" in lower_sys or "provided" in lower_sys:
                system_message = (
                    "You are a helpful assistant that answers questions using general knowledge. "
                    "Use the user's question to provide a clear, concise answer. If you don't know, say so."
                )

        logger.info(
            f"Processing question: '{request.question}' using model: {request.model_type.value}, RAG: {request.use_rag}"
        )

        search_results = []
        document_name = ""
        search_method = "none"

        if request.use_rag:
            if not pdf_database:
                raise HTTPException(
                    status_code=400,
                    detail="No PDFs uploaded yet. Please upload a PDF first or disable RAG mode.",
                )

            logger.info(f"Available PDFs: {list(pdf_database.keys())}")

            if request.document_id:
                # Query specific PDF
                if request.document_id not in pdf_database:
                    raise HTTPException(status_code=404, detail="Document ID not found")

                doc_info = pdf_database[request.document_id]
                document_name = doc_info["filename"]
                logger.info(f"Searching in specific PDF: {document_name}")

                # Try vector search first
                if pinecone_initialized and doc_info.get("has_vectors", False):
                    search_results = vector_similarity_search(
                        request.question, request.document_id, top_k=5
                    )
                    if search_results:
                        search_method = "vector"
                        logger.info(
                            f"Vector search found {len(search_results)} results"
                        )

                # Fallback to keyword search if vector search failed
                if (
                    not search_results
                    and "file_path" in doc_info
                    and os.path.exists(doc_info["file_path"])
                ):
                    text_chunks = extract_text_from_pdf(doc_info["file_path"])
                    search_results = simple_keyword_search(
                        request.question, text_chunks, top_k=5
                    )
                    if search_results:
                        search_method = "keyword"
                        logger.info(
                            f"Keyword search found {len(search_results)} results"
                        )

            else:
                # Query all PDFs
                document_name = f"All PDFs ({len(pdf_database)} documents)"
                logger.info(f"Searching across all {len(pdf_database)} PDFs")

                # Try vector search first
                if pinecone_initialized:
                    search_results = vector_similarity_search(request.question, top_k=5)
                    if search_results:
                        search_method = "vector"
                        logger.info(
                            f"Vector search found {len(search_results)} results"
                        )

                # Fallback to keyword search
                if not search_results:
                    all_text_chunks = get_all_text_chunks()
                    search_results = simple_keyword_search(
                        request.question, all_text_chunks, top_k=5
                    )
                    if search_results:
                        search_method = "keyword"
                        logger.info(
                            f"Keyword search found {len(search_results)} results"
                        )

        else:
            # RAG disabled
            document_name = "Direct model query (RAG disabled)"
            search_results = []

        # Generate answer
        answer, confidence = generate_answer(
            request.question,
            search_results,
            request.model_type,
            system_message,
            request.use_rag,
        )

        # Prepare source documents
        source_docs = []
        if request.use_rag and search_results:
            for result in search_results:
                source_text = (
                    result["text"][:300] + "..."
                    if len(result["text"]) > 300
                    else result["text"]
                )
                source_docs.append(f"({search_method} search) {source_text}")

        mode = "RAG" if request.use_rag else "Direct"
        logger.info(
            f"Question answered using {mode} mode with {request.model_type.value}"
        )

        return QuestionResponse(
            question=request.question,
            answer=answer,
            document_used=document_name,
            source_documents=source_docs if source_docs else None,
            model_used=request.model_type.value,
            confidence=confidence,
            system_message_used=system_message,
            rag_enabled=request.use_rag,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error answering question: {str(e)}"
        )


@app.get("/api/list-pdfs")
async def list_pdfs():
    """Get list of all uploaded PDFs"""
    logger.info(f"Listing {len(pdf_database)} PDFs")

    pdf_list = []
    for doc_id, doc_info in pdf_database.items():
        pdf_list.append(
            PDFInfo(
                document_id=doc_id,
                filename=doc_info["filename"],
                upload_time=doc_info["upload_time"],
                document_count=doc_info["document_count"],
                file_size=doc_info.get("file_size", 0),
                file_hash=doc_info.get("file_hash", ""),
                model_used=doc_info.get("model_used", "huggingface"),
            )
        )

    total_size = sum(doc.get("file_size", 0) for doc in pdf_database.values())

    return {
        "pdfs": pdf_list,
        "total_count": len(pdf_list),
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "pinecone_connected": pinecone_initialized,
        "pinecone_api": "new" if PINECONE_NEW_API else "old",
        "storage_file": STORAGE_FILE,
    }


@app.get("/api/pdf-info/{document_id}")
async def get_pdf_info(document_id: str):
    """Get information about a specific PDF"""
    if document_id not in pdf_database:
        raise HTTPException(status_code=404, detail="Document ID not found")

    doc_info = pdf_database[document_id]
    return PDFInfo(
        document_id=document_id,
        filename=doc_info["filename"],
        upload_time=doc_info["upload_time"],
        document_count=doc_info["document_count"],
        file_size=doc_info.get("file_size", 0),
        file_hash=doc_info.get("file_hash", ""),
        model_used=doc_info.get("model_used", "huggingface"),
    )


@app.delete("/api/delete-pdf/{document_id}")
async def delete_pdf(document_id: str):
    """Delete a specific PDF"""
    if document_id not in pdf_database:
        raise HTTPException(status_code=404, detail="Document ID not found")

    try:
        # Remove file from filesystem
        file_path = pdf_database[document_id].get("file_path")
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

        # Remove vectors from Pinecone
        if pinecone_initialized:
            delete_vectors_from_pinecone(document_id)

        # Remove from database
        filename = pdf_database[document_id]["filename"]
        del pdf_database[document_id]

        logger.info(f"Successfully deleted PDF: {filename}")
        return {"message": "PDF deleted successfully", "document_id": document_id}

    except Exception as e:
        logger.error(f"Error deleting PDF {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting PDF: {str(e)}")


@app.delete("/api/reset-all")
async def reset_all():
    """Reset all PDFs"""
    global pdf_database

    logger.info(f"Resetting all PDFs. Current count: {len(pdf_database)}")

    # Remove all files
    for doc_info in pdf_database.values():
        file_path = doc_info.get("file_path")
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {str(e)}")

    # Clear Pinecone
    if pinecone_initialized:
        try:
            if PINECONE_NEW_API:
                pc.delete_index(INDEX_NAME)
                # Recreate index
                pc.create_index(
                    name=INDEX_NAME,
                    dimension=EMBEDDING_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
            else:
                pinecone.delete_index(INDEX_NAME)
                # Recreate index
                pinecone.create_index(
                    name=INDEX_NAME, dimension=EMBEDDING_DIMENSION, metric="cosine"
                )
            logger.info("Pinecone index reset successfully")
        except Exception as e:
            logger.error(f"Error resetting Pinecone index: {str(e)}")

    # Clear database
    pdf_count_before = len(pdf_database)
    pdf_database.clear()

    logger.info(f"Reset completed. Cleared {pdf_count_before} PDFs")
    return {"message": "All PDFs reset successfully"}


@app.post("/api/debug/test-pinecone-upload")
async def debug_test_pinecone_upload():
    """Test if we can upload vectors to Pinecone"""
    if not pinecone_initialized or index is None:
        return {"error": "Pinecone not initialized"}

    try:
        # Test with a simple vector
        test_vectors = [
            {
                "id": "test_vector_1",
                "values": [0.1] * EMBEDDING_DIMENSION,  # Simple test vector
                "metadata": {
                    "text": "This is a test document chunk",
                    "document_id": "test_doc",
                    "chunk_index": 0,
                    "filename": "test.pdf",
                    "timestamp": datetime.now().isoformat(),
                },
            }
        ]

        # Try to upsert
        upsert_response = index.upsert(vectors=test_vectors)
        logger.info(f"Pinecone upsert response: {upsert_response}")

        # Check if vector exists
        time.sleep(2)  # Wait for indexing
        stats = index.describe_index_stats()

        return {
            "upsert_response": str(upsert_response),
            "index_stats": stats.to_dict() if hasattr(stats, "to_dict") else str(stats),
            "test_vector_id": "test_vector_1",
        }

    except Exception as e:
        logger.error(f"Pinecone test upload failed: {str(e)}")
        return {"error": str(e)}


@app.post("/api/debug/test-pdf-processing")
async def debug_test_pdf_processing(file: UploadFile = File(...)):
    """Test PDF processing step by step"""
    try:
        results = {"steps": {}, "success": False}

        # Step 1: Save file
        content = await file.read()
        file_hash = calculate_file_hash(content)
        results["steps"]["file_read"] = {
            "success": True,
            "file_size": len(content),
            "file_hash": file_hash,
        }

        # Save to temp location
        temp_path = f"/tmp/test_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(content)

        # Step 2: Extract text
        text_chunks = extract_text_from_pdf(temp_path)
        results["steps"]["text_extraction"] = {
            "success": len(text_chunks) > 0,
            "chunks_found": len(text_chunks),
            "sample_chunk": text_chunks[0][:100] + "..." if text_chunks else None,
        }

        if not text_chunks:
            os.remove(temp_path)
            return results

        # Step 3: Generate embeddings
        embeddings = generate_embeddings(text_chunks[:2])  # Just test 2 chunks
        results["steps"]["embedding_generation"] = {
            "success": len(embeddings) > 0,
            "embeddings_generated": len(embeddings),
            "embedding_dimension": len(embeddings[0]) if embeddings else 0,
            "sample_embedding": embeddings[0][:5] if embeddings else None,
        }

        if not embeddings:
            os.remove(temp_path)
            return results

        # Step 4: Test Pinecone storage
        test_doc_id = f"test_doc_{int(time.time())}"
        success = store_vectors_in_pinecone(test_doc_id, text_chunks[:2], embeddings)

        results["steps"]["pinecone_storage"] = {
            "success": success,
            "document_id": test_doc_id,
            "vectors_stored": len(embeddings) if success else 0,
        }

        # Clean up
        os.remove(temp_path)

        # Final check
        if success:
            time.sleep(2)
            stats = index.describe_index_stats()
            results["final_stats"] = (
                stats.to_dict() if hasattr(stats, "to_dict") else str(stats)
            )
            results["success"] = True

        return results

    except Exception as e:
        logger.error(f"PDF processing test failed: {str(e)}")
        return {"error": str(e), "steps": results.get("steps", {})}


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("templates", exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    os.makedirs(PDFS_TO_PROCESS_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    logger.info("🚀 Starting PDF Q&A Application with Enhanced Persistent Storage...")
    logger.info(f"📁 Storage file: {STORAGE_FILE}")
    logger.info(f"📁 Uploads directory: {UPLOADS_DIR}")
    logger.info(f"📁 PDFs to process directory: {PDFS_TO_PROCESS_DIR}")
    logger.info(f"📚 Loaded {len(pdf_database)} PDFs from persistent storage")
    logger.info(f"🌐 Pinecone API: {'New' if PINECONE_NEW_API else 'Old'}")
    logger.info("🌐 Access the application at: http://localhost:8000")

    # Check model configurations with detailed logging
    model_availability = check_model_availability()
    for model, available in model_availability.items():
        status = "✅ Available" if available else "❌ Not configured"
        logger.info(f"{model}: {status}")

    # Instructions for populating Pinecone
    if pinecone_initialized:
        status = check_pinecone_status()
        if "total_vectors" in status:
            logger.info(f"📊 Current Pinecone vectors: {status['total_vectors']}")

        if os.path.exists(PDFS_TO_PROCESS_DIR):
            pdf_files = [
                f for f in os.listdir(PDFS_TO_PROCESS_DIR) if f.lower().endswith(".pdf")
            ]
            if pdf_files:
                logger.info(
                    f"📄 Found {len(pdf_files)} PDFs in 'pdfs_to_process' directory ready for processing"
                )
                logger.info(
                    "💡 Use POST /api/process-directory to populate Pinecone with these PDFs"
                )

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
