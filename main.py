import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Optional Google Generative AI (Gemini) SDK
GEMINI_AVAILABLE = True
try:
    import google.generativeai as genai
except Exception:
    GEMINI_AVAILABLE = False

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str

class FileObject(BaseModel):
    path: str
    content: str

class GenerateResponse(BaseModel):
    files: List[FileObject]
    model: Optional[str] = None

class ChatRequest(BaseModel):
    instructions: str
    files: List[FileObject]

class ChatResponse(BaseModel):
    files: List[FileObject]
    model: Optional[str] = None

class ConnectRequest(BaseModel):
    api_key: str

class ConnectResponse(BaseModel):
    connected: bool
    model: Optional[str] = None
    message: Optional[str] = None


@app.get("/")
def read_root():
    return {"message": "LLM App Generator Backend Running"}


SYSTEM_SPEC = (
    "You are an expert full‑stack code generator. Return ONLY valid JSON matching this schema: "
    "{\n  files: [ { path: string, content: string } ]\n}. "
    "Rules: 1) No markdown, no backticks. 2) Create a runnable front-end app using Vite + React by default. "
    "3) Include package.json, index.html, src/main.jsx, src/App.jsx (and any components), and instructions in code (README.md optional). "
    "4) Keep file paths relative starting at project root. 5) Do not include binary assets; use placeholders or URLs. "
)

# Prefer a model variant that exists on AI Studio v1beta and supports standard calls
# Users can override via GEMINI_MODEL env var
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-002")

# In-memory runtime key set via /llm/connect
_RUNTIME_GEMINI_API_KEY: Optional[str] = None


def _ensure_gemini_config():
    if not GEMINI_AVAILABLE:
        raise HTTPException(status_code=500, detail="google-generativeai package not installed on backend.")
    api_key = _RUNTIME_GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set. Provide it via /llm/connect or environment.")
    genai.configure(api_key=api_key)


def _extract_text(resp: Any) -> str:
    # Handle SDK differences safely
    if resp is None:
        return ""
    if hasattr(resp, "text") and callable(getattr(resp, "text")):
        try:
            return resp.text()
        except Exception:
            pass
    if hasattr(resp, "candidates") and resp.candidates:
        parts = getattr(resp.candidates[0], "content", None)
        if parts and hasattr(parts, "parts") and parts.parts:
            # Join text parts
            return "".join(getattr(p, "text", "") for p in parts.parts)
    # Fallback string conversion
    return str(resp)


@app.post("/llm/connect", response_model=ConnectResponse)
def connect_gemini(req: ConnectRequest):
    global _RUNTIME_GEMINI_API_KEY
    if not GEMINI_AVAILABLE:
        raise HTTPException(status_code=500, detail="google-generativeai package not installed on backend.")
    key = (req.api_key or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="api_key is required")
    try:
        # Configure with the provided key and validate with a lightweight call
        genai.configure(api_key=key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        # Prefer count_tokens for inexpensive validation, but gracefully fallback
        try:
            _ = model.count_tokens("ping")
        except Exception:
            # Some model+API combinations may not expose countTokens. Fallback to a minimal generate.
            _ = model.generate_content("ping")
        # If successful, store for this process
        _RUNTIME_GEMINI_API_KEY = key
        return ConnectResponse(connected=True, model=GEMINI_MODEL, message="Connected to Gemini.")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid API key or connection error: {e}")


@app.get("/llm/status")
def llm_status():
    return {
        "sdk": "available" if GEMINI_AVAILABLE else "missing",
        "model": GEMINI_MODEL,
        "connected": bool(_RUNTIME_GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")),
    }


@app.post("/llm/generate", response_model=GenerateResponse)
def generate_app(req: GenerateRequest):
    _ensure_gemini_config()
    model = genai.GenerativeModel(GEMINI_MODEL)

    user_prompt = (
        f"Build a minimal but complete project that satisfies this description: '{req.prompt}'. "
        "Target: JavaScript, Vite + React front-end app startable with `npm run dev`.\n"
        "Include all required files."
    )

    try:
        resp = model.generate_content([
            {"role": "user", "parts": [SYSTEM_SPEC]},
            {"role": "user", "parts": [user_prompt]},
        ])
        text = _extract_text(resp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {e}")

    # Attempt to parse JSON; if it fails, try to salvage by locating first and last braces
    import json, re
    payload: Dict[str, Any]
    try:
        payload = json.loads(text)
    except Exception:
        # Strip code fences if present
        cleaned = re.sub(r"^```[a-zA-Z]*|```$", "", text.strip())
        # Find JSON object
        match = re.search(r"\{[\s\S]*\}\s$", cleaned)
        if not match:
            raise HTTPException(status_code=500, detail="Model did not return valid JSON.")
        try:
            payload = json.loads(match.group(0))
        except Exception:
            raise HTTPException(status_code=500, detail="Failed to parse JSON from model response.")

    files_in = payload.get("files", [])
    files: List[FileObject] = []
    for f in files_in:
        path = f.get("path")
        content = f.get("content", "")
        if path and isinstance(content, str):
            files.append(FileObject(path=path, content=content))

    if not files:
        raise HTTPException(status_code=500, detail="No files returned by model.")

    return GenerateResponse(files=files, model=GEMINI_MODEL)


@app.post("/llm/chat", response_model=ChatResponse)
def chat_modify(req: ChatRequest):
    _ensure_gemini_config()
    model = genai.GenerativeModel(GEMINI_MODEL)

    file_manifest = "\n".join([f"- {f.path} ({len(f.content)} chars)" for f in req.files])
    files_concat = "\n\n".join([f"FILE: {f.path}\n<content>\n{f.content}\n</content>" for f in req.files])

    instruction = (
        "You are updating an existing project. Return ONLY JSON { files: [ { path, content } ] } with the FULL, UPDATED contents for every file that changed. "
        "Do not return diffs. Include new files if needed."
    )

    try:
        resp = model.generate_content([
            {"role": "user", "parts": [instruction]},
            {"role": "user", "parts": [f"User instructions: {req.instructions}"]},
            {"role": "user", "parts": [f"Current project manifest:\n{file_manifest}"]},
            {"role": "user", "parts": [f"Current files with content:\n{files_concat}"]},
        ])
        text = _extract_text(resp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {e}")

    import json, re
    try:
        payload = json.loads(text)
    except Exception:
        cleaned = re.sub(r"^```[a-zA-Z]*|```$", "", text.strip())
        match = re.search(r"\{[\s\S]*\}\s$", cleaned)
        if not match:
            raise HTTPException(status_code=500, detail="Model did not return valid JSON.")
        payload = json.loads(match.group(0))

    files_in = payload.get("files", [])
    files: List[FileObject] = []
    for f in files_in:
        path = f.get("path")
        content = f.get("content", "")
        if path and isinstance(content, str):
            files.append(FileObject(path=path, content=content))

    if not files:
        raise HTTPException(status_code=500, detail="No files returned by model.")

    return ChatResponse(files=files, model=GEMINI_MODEL)


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "gemini_sdk": "✅ Available" if GEMINI_AVAILABLE else "❌ Not Installed",
        "gemini_key": "✅ Set" if (_RUNTIME_GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")) else "❌ Not Set",
    }
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
