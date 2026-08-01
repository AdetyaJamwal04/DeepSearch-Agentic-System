import os
import json
import asyncio
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from graph import app as research_graph

app = FastAPI(
    title="DeepSearch API",
    description="Agentic Deep Research Pipeline powered by LangGraph",
    version="1.0.0"
)

# ── CORS ─────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down to specific domains in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global Exception Handler ────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    print(f"[API] Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again."}
    )


# ── Schemas ──────────────────────────────────────────────────────
class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=1000)

class ResearchResponse(BaseModel):
    query: str
    report: str


# ── Health Check ─────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


# ── POST /research (blocking) ───────────────────────────────────
REQUEST_TIMEOUT = 180.0  # 3 minute hard limit

@app.post("/research", response_model=ResearchResponse)
async def run_research(request: ResearchRequest):
    print(f"\n[API] Received research request: '{request.query}'")

    try:
        final_state = await asyncio.wait_for(
            research_graph.ainvoke({"query": request.query}),
            timeout=REQUEST_TIMEOUT,
        )

        report_content = final_state.get("report_markdown")
        if not report_content:
            raise HTTPException(status_code=500, detail="Research completed but no report was generated.")

        return ResearchResponse(
            query=request.query,
            report=report_content
        )
    except asyncio.TimeoutError:
        print(f"[API] Research timed out after {REQUEST_TIMEOUT}s")
        raise HTTPException(status_code=504, detail="Research timed out. Try a simpler query.")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] Research execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /research/stream (SSE) ──────────────────────────────────
async def research_event_generator(query: str):
    """
    Generator that yields Server-Sent Events (SSE) as LangGraph nodes finish.
    """
    try:
        async for event in research_graph.astream({"query": query}, stream_mode="updates"):
            for node_name, state_update in event.items():
                payload = {"node": node_name, "status": "completed"}
                yield f"data: {json.dumps(payload)}\n\n"

                if isinstance(state_update, dict) and "report_markdown" in state_update:
                    final_payload = {"node": "END", "report": state_update["report_markdown"]}
                    yield f"data: {json.dumps(final_payload)}\n\n"
    except Exception as e:
        print(f"[API] SSE stream error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.get("/research/stream")
async def stream_research(query: str = Query(..., min_length=5, max_length=1000)):
    """
    Server-Sent Events endpoint. Streams progress updates to the client
    in real-time as the LangGraph executes.
    """
    return StreamingResponse(
        research_event_generator(query),
        media_type="text/event-stream"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8080, reload=True)
