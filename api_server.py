from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import os
import tempfile
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional, Dict, List, Any

# Internal imports
from complete_pipeline_demo import CompletePipelineDemo
from hybrid_ocr_monopoly import process_image, get_easyocr_reader

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("MathMendAPI")

# --- Global State ---
class GlobalState:
    demo_pipeline: Optional[CompletePipelineDemo] = None
    system_loaded: bool = False

state = GlobalState()

# --- Lifespan Manager (Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🔧 Starting up MathMend API...")
    
    # 1. Load Neuro-Symbolic Pipeline
    try:
        logger.info("Loading neuro-symbolic pipeline...")
        # Run potentially blocking load in a thread
        state.demo_pipeline = await asyncio.to_thread(CompletePipelineDemo)
        
        # We need to call load_complete_system(). This is also blocking.
        success = await asyncio.to_thread(state.demo_pipeline.load_complete_system)
        
        if success:
            state.system_loaded = True
            logger.info("✅ Pipeline loaded successfully!")
        else:
            logger.error("❌ Failed to load pipeline.")
    except Exception as e:
        logger.error(f"❌ Exception during pipeline loading: {e}")

    # 2. Warmup OCR (Lazy load trigger)
    try:
        logger.info("Warming up OCR engine...")
        # Trigger the global reader initialization in a thread
        await asyncio.to_thread(get_easyocr_reader)
        logger.info("✅ OCR engine ready.")
    except Exception as e:
        logger.error(f"⚠️ OCR warmup failed (will retry on first request): {e}")

    yield
    
    # Shutdown
    logger.info("🛑 Shutting down MathMend API...")
    # Clean up resources if needed
    state.demo_pipeline = None

# --- FastAPI App ---
app = FastAPI(
    title="MathMend API",
    description="Neuro-Symbolic Math Solver & OCR API",
    version="2.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"], # Allow all for dev convenience if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Math question to solve")

class ChatResponse(BaseModel):
    answer: str
    solution: Dict[str, Any] = {}
    reasoning: Optional[str] = None
    equations: List[str] = []
    latency_ms: float
    result_type: str = "unknown"

class OCRResponse(BaseModel):
    extracted_text: str
    latency_ms: float

# --- Helper Functions ---

def _process_pipeline_request(question: str):
    """
    Blocking wrapper for pipeline processing.
    """
    if not state.system_loaded or not state.demo_pipeline:
        raise RuntimeError("System not fully loaded")
    
    # Run the pipeline
    return state.demo_pipeline.pipeline(question, top_k=3, explain=True)

# --- Routes ---

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if state.system_loaded else "degraded", 
        "pipeline_loaded": state.system_loaded
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    Answer a math question using the neuro-symbolic pipeline.
    Runs asynchronously to avoid blocking the server.
    """
    if not state.system_loaded:
        raise HTTPException(status_code=503, detail="System is still loading, please try again in a moment.")

    start_time = time.time()
    question = req.question.strip()
    
    logger.info(f"Processing chat request: {question[:50]}...")

    try:
        # Offload blocking work to threadpool
        result = await asyncio.to_thread(_process_pipeline_request, question)
        
        # Extract data
        result_type = getattr(result, "result_type", "unknown")
        raw_solution = getattr(result, "solution", None) or {}
        reasoning = getattr(result, "reasoning", "")
        note = getattr(result, "note", None)
        equations = getattr(result, "equations", []) or []

        # Normalize solution
        solution_dict = {}
        if isinstance(raw_solution, dict):
            for k, v in raw_solution.items():
                try:
                    solution_dict[str(k)] = float(v)
                except:
                    solution_dict[str(k)] = str(v)

        # Construct Answer Text
        answer_parts = []
        
        # 1. Explicit Solution
        if solution_dict:
            sol_str = ", ".join(f"{k} = {v}" for k, v in solution_dict.items())
            answer_parts.append(f"Solution: {sol_str}")
        
        # 2. Reasoning (Truncated if too long)
        if isinstance(reasoning, str) and reasoning.strip():
            clean_reasoning = reasoning.strip()
            if len(clean_reasoning) > 1000:
                clean_reasoning = clean_reasoning[:1000] + "... (truncated)"
            answer_parts.insert(0, clean_reasoning)
            
        # 3. Fallback message
        if not solution_dict:
            msg = "I processed your question but couldn't find a precise mathematical solution."
            if result_type == "unresolved":
                msg = "I couldn't reliably deduce a solution from the extracted equations."
            
            if note:
                msg += f" Note: {note}"
            
            answer_parts.append(msg)

        final_answer = "\n\n".join(answer_parts)
        
        latency = (time.time() - start_time) * 1000
        logger.info(f"Chat request completed in {latency:.2f}ms. Result: {result_type}")

        return ChatResponse(
            answer=final_answer,
            solution=solution_dict,
            reasoning=reasoning if reasoning else None,
            equations=[str(e) for e in equations],
            latency_ms=latency,
            result_type=result_type
        )

    except Exception as e:
        logger.error(f"Error in /chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/ocr", response_model=OCRResponse)
async def ocr_endpoint(image: UploadFile = File(...)):
    """
    Extract text from an uploaded image.
    Uses cached OCR engine for speed.
    """
    if not image:
        raise HTTPException(status_code=400, detail="No image provided")

    start_time = time.time()
    logger.info(f"Processing OCR for image: {image.filename}")

    # Create temp file
    suffix = os.path.splitext(image.filename)[1] or ".png"
    tmp_path = ""
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await image.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Run OCR in threadpool to allow concurrency
        extracted_text = await asyncio.to_thread(process_image, tmp_path)
        
        latency = (time.time() - start_time) * 1000
        logger.info(f"OCR completed in {latency:.2f}ms")

        return OCRResponse(
            extracted_text=extracted_text,
            latency_ms=latency
        )

    except Exception as e:
        logger.error(f"Error in /ocr: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"OCR error: {str(e)}")
    
    finally:
        # Cleanup temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)









# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import uvicorn
# import os
# import tempfile

# from complete_pipeline_demo import CompletePipelineDemo
# from hybrid_ocr_monopoly import process_image

# app = FastAPI(title="Math Chatbot API")

# # Allow your React dev server to call this API
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # --- Initialize the pipeline once at startup ---

# demo = CompletePipelineDemo()
# system_loaded = False

# @app.on_event("startup")
# async def startup_event():
#     global system_loaded
#     print("🔧 Loading neuro-symbolic pipeline for API...")
#     system_loaded = demo.load_complete_system()
#     if not system_loaded:
#         print("❌ Failed to load complete system at startup.")


# # --- Request models ---

# class ChatRequest(BaseModel):
#     question: str


# # --- Routes ---

# @app.post("/chat")
# async def chat_endpoint(req: ChatRequest):
#     """
#     Answer a single math question via the neuro-symbolic pipeline.
#     """
#     if not system_loaded:
#         raise HTTPException(status_code=500, detail="System not loaded")

#     question = req.question.strip()
#     if not question:
#         raise HTTPException(status_code=400, detail="Question is empty")

#     try:
#         # Use the main pipeline directly for a single query
#         result = demo.pipeline(question, top_k=3, explain=True)

#         # Extract structured fields from the pipeline result
#         result_type = getattr(result, "result_type", "unknown")
#         raw_solution = getattr(result, "solution", None) or {}
#         reasoning = getattr(result, "reasoning", None)
#         note = getattr(result, "note", None)
#         equations = getattr(result, "equations", None)

#         # Normalize solution to a plain dict of Python types
#         def _normalize_solution(sol):
#             normalized = {}
#             try:
#                 for k, v in sol.items():
#                     key = str(k)
#                     try:
#                         # Try numeric conversion first
#                         normalized[key] = float(v)
#                     except Exception:
#                         # Fallback to string representation
#                         normalized[key] = str(v)
#                 return normalized
#             except Exception:
#                 return {}

#         solution = _normalize_solution(raw_solution) if isinstance(raw_solution, dict) else {}

#         # Build user-facing answer text with clear priorities
#         if solution:
#             # For resolved cases, always show the solution clearly
#             solution_str = ", ".join(f"{k} = {v}" for k, v in solution.items())
#             answer_parts = [f"Solution: {solution_str}"]

#             # Append a short reasoning/explanation if available (but avoid huge code dumps)
#             if isinstance(reasoning, str) and reasoning.strip():
#                 trimmed_reasoning = reasoning.strip()
#                 # Limit extremely long reasoning blocks
#                 if len(trimmed_reasoning) > 800:
#                     trimmed_reasoning = trimmed_reasoning[:800] + "... (truncated)"
#                 answer_parts.insert(0, trimmed_reasoning)

#             answer_text = "\n\n".join(answer_parts)

#         else:
#             # No usable solution from the pipeline
#             # Try to give a friendlier and more specific message
#             if result_type == "unresolved":
#                 base_msg = "I couldn't reliably solve this problem from the extracted equations."
#             else:
#                 base_msg = "I processed your question but could not find a clear solution."

#             # If we have a note from the pipeline, append it for context
#             extra_note = note if isinstance(note, str) and note.strip() else ""

#             # Show any equations that were extracted, to help the user debug OCR/text
#             eq_info = ""
#             if equations:
#                 try:
#                     eq_preview = ", ".join(str(e) for e in equations)
#                     if len(eq_preview) > 300:
#                         eq_preview = eq_preview[:300] + "... (truncated)"
#                     eq_info = f" Extracted equations were: {eq_preview}"
#                 except Exception:
#                     eq_info = ""

#             answer_text = base_msg + (f" {extra_note}" if extra_note else "") + eq_info

#         return {
#             "answer": answer_text,
#             "result_type": result_type,
#             "solution": solution,
#         }

#     except Exception as e:
#         print(f"Error in /chat endpoint: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/ocr")
# async def ocr_endpoint(image: UploadFile = File(...)):
#     """
#     Extract text from an uploaded math question image.
#     """
#     if not image:
#         raise HTTPException(status_code=400, detail="No image uploaded")

#     try:
#         # Save to a temporary file for OpenCV/EasyOCR
#         suffix = os.path.splitext(image.filename)[1]
#         with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#             tmp.write(await image.read())
#             tmp_path = tmp.name

#         try:
#             fused_text = process_image(tmp_path)
#         finally:
#             # Clean up temp file
#             if os.path.exists(tmp_path):
#                 os.remove(tmp_path)

#         return {"extractedText": fused_text}

#     except Exception as e:
#         print(f"Error in /ocr endpoint: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


# if __name__ == "__main__":
#     # Run the API server
#     uvicorn.run(app, host="0.0.0.0", port=8000)