"""
Framework-agnostic example — uses MemoryOS with the raw OpenAI SDK.

No LangChain. No LangGraph. No deepagents.
Just OpenAI's chat.completions API + MemoryOS for persistent memory.

Run:
    python examples/openai_sdk_example.py

Required env: OPENAI_API_KEY, NEO4J_PASSWORD (and Neo4j + Redis running).
"""

import asyncio
import json
import os
import uuid

from dotenv import load_dotenv
from openai import OpenAI

from memoryos import MemoryOS

load_dotenv()


SYSTEM_PROMPT = (
    "You are a personal assistant with persistent memory. "
    "Use the memory context (if provided above) to personalize responses. "
    "When the user shares a new fact, call remember_fact to store it."
)


async def chat_turn(client, memory, user_id, session_id, user_message):
    """Run one chat turn with memory injection + tool dispatch."""
    # Build the message list
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # 1. INJECT memory context as a system message
    messages = memory.inject_context(messages, user_id=user_id)

    # 2. CALL OpenAI with memory tools available
    response = client.chat.completions.create(
        model=os.getenv("CHAT_MODEL", "gpt-4o"),
        messages=messages,
        tools=memory.openai_tools(),       # ← memoryos provides OpenAI-format tool schemas
    )
    msg = response.choices[0].message

    # 3. DISPATCH any tool calls back through MemoryOS
    if msg.tool_calls:
        messages.append(msg)
        for tc in msg.tool_calls:
            result = memory.handle_tool_call(
                name=tc.function.name,
                args=json.loads(tc.function.arguments),
                user_id=user_id,                # ← explicit user_id, no hallucination
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })
        # Second call so the model can summarize the tool results
        response = client.chat.completions.create(
            model=os.getenv("CHAT_MODEL", "gpt-4o"),
            messages=messages,
        )
        msg = response.choices[0].message

    reply = msg.content or ""

    # 4. SAVE the turn to STM (so it survives session end → LTM extraction)
    memory.save_message(session_id, "user", user_message)
    memory.save_message(session_id, "assistant", reply)

    return reply


async def main():
    user_id = "demo_user_openai_sdk"
    session_id = f"session_{uuid.uuid4().hex[:8]}"

    memory = MemoryOS.from_env()
    await memory.init()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print(f"\n=== Session: {session_id} | User: {user_id} ===\n")

    turns = [
        "Hi, I'm Burhan and I'm 24.",
        "I work as an AI engineer and love building agents.",
        "What do you know about me so far?",
    ]
    for msg in turns:
        print(f"USER: {msg}")
        reply = await chat_turn(client, memory, user_id, session_id, msg)
        print(f"ASSISTANT: {reply}\n")

    # Trigger background extraction → write to graph
    status = memory.end_session(user_id, session_id)
    print(f"[end_session] status={status}")

    # Wait for extraction to finish
    print("Waiting for extraction...")
    while memory.get_extraction_status(user_id) == "processing":
        await asyncio.sleep(1)
    print(f"[extraction] status={memory.get_extraction_status(user_id)}")

    print("\n=== Final graph context for this user ===")
    print(memory.get_context(user_id))

    await memory.close()


if __name__ == "__main__":
    asyncio.run(main())
