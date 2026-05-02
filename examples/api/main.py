import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from examples.api.models import (
    StartSessionRequest, StartSessionResponse,
    ChatRequest, ChatResponse,
    EndSessionRequest, EndSessionResponse,
    MemoryContextResponse, HealthResponse,
)
from examples.api.session_manager import session_manager
from memoryos import MemoryOS, MemoryConfig

logger = logging.getLogger(__name__)

# Single shared MemoryOS instance for the whole API
memory = MemoryOS(config=MemoryConfig.from_env())


@asynccontextmanager
async def lifespan(app: FastAPI):
    await memory.init()
    logger.info("[MemoryOS API] Started")
    yield
    await memory.close()
    logger.info("[MemoryOS API] Shutdown")


app = FastAPI(
    title="MemoryOS API",
    description="Personal AI assistant with Knowledge Graph memory",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/session/start", response_model=StartSessionResponse)
async def start_session(request: StartSessionRequest):
    session = session_manager.create_session(user_id=request.user_id)
    print(f"[API] Session started user={request.user_id} session={session.session_id}")
    return StartSessionResponse(
        session_id=session.session_id,
        user_id=session.user_id,
        created_at=session.created_at,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Call /session/start first.")

    print(f"[API] Chat user={request.user_id} session={request.session_id}")

    try:
        result = await session.agent.ainvoke(
            memory.wrap_input(request.message, request.user_id),
            config=memory.make_config(request.user_id, request.session_id),
        )
    except Exception as e:
        logger.error(f"[chat] Agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    reply = result["messages"][-1].content
    memory.save_message(request.session_id, "user", request.message)
    memory.save_message(request.session_id, "assistant", reply)

    return ChatResponse(
        session_id=request.session_id,
        user_id=request.user_id,
        message=reply,
    )


@app.post("/session/end", response_model=EndSessionResponse)
async def end_session(request: EndSessionRequest):
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    kg_status = memory.end_session(request.user_id, request.session_id)
    session_manager.remove_session(request.session_id)
    print(f"[API] Session ended user={request.user_id} session={request.session_id} kg={kg_status}")

    return EndSessionResponse(
        session_id=request.session_id,
        user_id=request.user_id,
        status="ended",
        kg_extraction=kg_status,
        ended_at=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/memory/{user_id}", response_model=MemoryContextResponse)
async def get_memory(user_id: str):
    ctx = memory.get_context(user_id)
    return MemoryContextResponse(
        user_id=user_id,
        context=ctx,
        has_data=bool(ctx),
    )


@app.delete("/memory/{user_id}")
async def clear_memory(user_id: str):
    deleted = memory.delete_user_memory(user_id)
    # Invalidate middleware cache for any active sessions belonging to this user
    for session in session_manager._sessions.values():
        if session.user_id == user_id:
            memory.invalidate_cache(user_id)
    return {"status": "cleared", "user_id": user_id, "nodes_deleted": deleted}


@app.delete("/memory/{user_id}/fact/{fact_id}")
async def delete_fact(user_id: str, fact_id: str):
    from memoryos.graph.connection import GraphConnection
    conn = memory._connection
    with conn.get_driver().session() as s:
        result = s.run(
            "MATCH (:User {user_id: $uid})-[r:KNOWS]->(f:Fact {fact_id: $fid}) "
            "DELETE r, f RETURN count(f) AS deleted",
            uid=user_id, fid=fact_id,
        ).data()
    deleted = result[0]["deleted"] if result else 0
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Fact {fact_id} not found.")
    return {"status": "deleted", "fact_id": fact_id}


@app.get("/memory/{user_id}/status")
async def memory_status(user_id: str):
    status = memory.get_extraction_status(user_id)
    return {"user_id": user_id, "status": status}


@app.get("/health", response_model=HealthResponse)
async def health():
    checks = memory.verify_connectivity()
    return HealthResponse(
        status="ok" if all(checks.values()) else "degraded",
        redis=checks["redis"],
        neo4j=checks["graph"],
    )
