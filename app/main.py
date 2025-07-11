import json
import os
import sys

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Add parent directory to path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .inference import generate_text_stream

app = FastAPI(
    title="Shakespeare Text Generator",
    description="Generate Shakespeare-style text using a trained transformer model",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Static & template serving
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE_DIR, "..", "static")),
    name="static",
)
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "..", "templates"))

# Serve the frontend
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """Serve the main web interface."""
    return templates.TemplateResponse("index.html", {"request": request})


# API endpoint to generate text with streaming
@app.post("/generate/")
async def generate(
    max_tokens: int = Form(300, description="Number of tokens to generate"),
    seed_text: str = Form("", description="Optional seed text to start generation"),
):
    """Generate Shakespeare-style text with streaming."""
    try:
        # Ensure reasonable limits
        max_tokens = min(max(max_tokens, 50), 1000)  # Between 50 and 1000 tokens

        def generate_stream():
            """Inner generator function for streaming text character by character."""
            try:
                for char in generate_text_stream(
                    max_new_tokens=max_tokens, seed_text=seed_text
                ):
                    # Send each character as a JSON chunk
                    yield f"data: {json.dumps({'char': char})}\n\n"
                # Send completion signal
                yield f"data: {json.dumps({'complete': True})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Methods": "*",
            },
        )
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# Health check endpoint
@app.get("/health")
def health_check():
    """Health check endpoint to verify the service is running."""
    return {"status": "healthy", "message": "Shakespeare Text Generator is running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
