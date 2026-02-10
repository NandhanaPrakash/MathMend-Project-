from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import tempfile

from complete_pipeline_demo import CompletePipelineDemo
from hybrid_ocr_monopoly import process_image

app = FastAPI(title="Math Chatbot API")

# Allow your React dev server to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialize the pipeline once at startup ---

demo = CompletePipelineDemo()
system_loaded = False

@app.on_event("startup")
async def startup_event():
    global system_loaded
    print("🔧 Loading neuro-symbolic pipeline for API...")
    system_loaded = demo.load_complete_system()
    if not system_loaded:
        print("❌ Failed to load complete system at startup.")


# --- Request models ---

class ChatRequest(BaseModel):
    question: str


# --- Routes ---

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Answer a single math question via the neuro-symbolic pipeline.
    """
    if not system_loaded:
        raise HTTPException(status_code=500, detail="System not loaded")

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty")

    try:
        # Use the main pipeline directly for a single query
        result = demo.pipeline(question, top_k=3, explain=True)

        # Pick a human-readable answer:
        solution = getattr(result, "solution", None)
        reasoning = getattr(result, "reasoning", None)
        note = getattr(result, "note", None)

        if solution:
            # Convert dict solution to text
            solution_str = ", ".join(f"{k} = {v}" for k, v in solution.items())
            answer_text = f"Solution: {solution_str}"
            if reasoning:
                answer_text = reasoning + "\n\n" + answer_text
        elif reasoning:
            answer_text = reasoning
        elif note:
            answer_text = note
        else:
            answer_text = "I processed your question but could not find a clear solution."

        return {"answer": answer_text}

    except Exception as e:
        print(f"Error in /chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ocr")
async def ocr_endpoint(image: UploadFile = File(...)):
    """
    Extract text from an uploaded math question image.
    """
    if not image:
        raise HTTPException(status_code=400, detail="No image uploaded")

    try:
        # Save to a temporary file for OpenCV/EasyOCR
        suffix = os.path.splitext(image.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await image.read())
            tmp_path = tmp.name

        try:
            fused_text = process_image(tmp_path)
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return {"extractedText": fused_text}

    except Exception as e:
        print(f"Error in /ocr endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run the API server
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