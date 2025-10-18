from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import shutil
from pathlib import Path
import uuid
from dotenv import load_dotenv
from gemini_extractor import extract_tables_from_pdf, get_extraction_summary

# Load environment variables
load_dotenv()

app = FastAPI(title="PDF Intelligence API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "PDF Intelligence API is running"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file and store it temporarily.
    Returns a unique file ID for later processing.
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed"
            )

        # Generate unique file ID
        file_id = str(uuid.uuid4())
        file_extension = ".pdf"
        file_path = UPLOAD_DIR / f"{file_id}{file_extension}"

        # Save uploaded file
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "success": True,
            "message": "File uploaded successfully",
            "file_id": file_id,
            "filename": file.filename,
            "file_path": str(file_path)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading file: {str(e)}"
        )
    finally:
        await file.close()


@app.post("/extract")
async def extract_pdf_tables(file_id: str):
    """
    Extract tables from an uploaded PDF file.

    Args:
        file_id: The unique file ID returned from the upload endpoint

    Returns:
        JSON with extracted tables and metadata
    """
    try:
        # Find the file in uploads directory
        file_path = UPLOAD_DIR / f"{file_id}.pdf"

        if not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"File not found. Please upload a PDF first."
            )

        # Extract tables from PDF (returns dict with document_context and tables)
        extraction_result = extract_tables_from_pdf(str(file_path))

        # Generate summary from the tables array
        summary = get_extraction_summary(extraction_result["tables"])

        return {
            "success": True,
            "file_id": file_id,
            "document_context": extraction_result["document_context"],
            "summary": summary,
            "tables": extraction_result["tables"]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error extracting tables: {str(e)}"
        )


from analysis_engine import detect_contradictions, find_gaps, get_clarification


@app.post("/analyze/contradictions")
async def analyze_contradictions(tables_data: dict):
    try:
        contradictions = detect_contradictions(tables_data['tables'])
        return {"success": True, "contradictions": contradictions}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/analyze/gaps")
async def analyze_gaps_endpoint(tables_data: dict):
    try:
        gaps = find_gaps(tables_data['tables'])
        return {"success": True, "gaps": gaps}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/clarify")
async def get_clarification_endpoint(request: dict):
    try:
        finding = request['finding']
        pdf_context = request['pdf_context']
        
        clarification = get_clarification(finding, pdf_context)
        
        return {
            "success": True,
            "clarification": clarification
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unexpected errors"""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "An unexpected error occurred",
            "detail": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
