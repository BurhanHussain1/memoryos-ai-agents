"""
MemoryOS — Streamlit Chat UI

Run:
    streamlit run streamlit_app.py

Make sure the FastAPI server is running first:
    python run.py
"""

import requests
import streamlit as st

API_BASE = "http://localhost:8000"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MemoryOS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state init ────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "user_id" not in st.session_state:
    st.session_state.user_id = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_ended" not in st.session_state:
    st.session_state.session_ended = False
if "past_sessions" not in st.session_state:
    st.session_state.past_sessions = []   # list of ended session IDs


# ── Helpers ───────────────────────────────────────────────────────────────────
def api_post(path: str, payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the API server. Run `python run.py` first.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_get(path: str) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the API server. Run `python run.py` first.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def start_session(user_id: str) -> bool:
    data = api_post("/session/start", {"user_id": user_id})
    if data:
        st.session_state.session_id = data["session_id"]
        st.session_state.user_id = user_id
        st.session_state.messages = []
        st.session_state.session_ended = False
        return True
    return False


def end_session() -> str | None:
    if not st.session_state.session_id:
        return None
    data = api_post("/session/end", {
        "session_id": st.session_state.session_id,
        "user_id": st.session_state.user_id,
    })
    if data:
        st.session_state.past_sessions.append(st.session_state.session_id)
        st.session_state.session_ended = True
        return data.get("kg_extraction", "unknown")
    return None


def send_message(message: str) -> str | None:
    data = api_post("/chat", {
        "session_id": st.session_state.session_id,
        "user_id": st.session_state.user_id,
        "message": message,
    })
    return data["message"] if data else None


def get_memory(user_id: str) -> dict | None:
    return api_get(f"/memory/{user_id}")


def get_health() -> dict | None:
    return api_get("/health")


def get_memory_status(user_id: str) -> str | None:
    data = api_get(f"/memory/{user_id}/status")
    return data.get("status") if data else None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 MemoryOS")
    st.caption("Personal AI with Knowledge Graph Memory")
    st.divider()

    # ── Health indicator ──
    health = get_health()
    if health:
        redis_icon = "🟢" if health.get("redis") else "🔴"
        neo4j_icon = "🟢" if health.get("neo4j") else "🔴"
        st.markdown(f"{redis_icon} Redis &nbsp;&nbsp; {neo4j_icon} Neo4j", unsafe_allow_html=True)
    else:
        st.markdown("🔴 API offline")
    st.divider()

    # ── User ID ──
    st.subheader("👤 User")
    user_id_input = st.text_input(
        "User ID",
        value=st.session_state.user_id or "",
        placeholder="e.g. user_001",
        label_visibility="collapsed",
    )

    # ── Session controls ──
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🆕 New Session", use_container_width=True, type="primary"):
            uid = user_id_input.strip()
            if not uid:
                st.warning("Enter a User ID first.")
            else:
                with st.spinner("Starting session..."):
                    ok = start_session(uid)
                if ok:
                    st.success(f"Session started!")
                    st.rerun()

    with col2:
        end_disabled = not st.session_state.session_id or st.session_state.session_ended
        if st.button("🔚 End Session", use_container_width=True, disabled=end_disabled):
            with st.spinner("Ending session & saving memory..."):
                kg_status = end_session()
            if kg_status:
                st.success(f"Session ended. Memory extraction: **{kg_status}**")
                st.rerun()

    # Show extraction status indicator when session has ended
    if st.session_state.session_ended and st.session_state.user_id:
        mem_status = get_memory_status(st.session_state.user_id)
        if mem_status == "processing":
            st.info("⏳ Saving conversation to Knowledge Graph...")
        elif mem_status == "ready" and st.session_state.past_sessions:
            st.success("✅ Memory saved to Knowledge Graph")

    # ── Current session info ──
    if st.session_state.session_id:
        st.divider()
        st.subheader("📋 Current Session")
        status = "⛔ Ended" if st.session_state.session_ended else "🟢 Active"
        st.markdown(f"**Status:** {status}")
        st.code(st.session_state.session_id, language=None)
        st.markdown(f"**User:** `{st.session_state.user_id}`")
        st.markdown(f"**Messages:** {len(st.session_state.messages)}")

    # ── Past sessions ──
    if st.session_state.past_sessions:
        st.divider()
        st.subheader("📂 Past Sessions")
        for sid in reversed(st.session_state.past_sessions[-5:]):
            st.caption(f"✓ {sid}")

    # ── Memory viewer ──
    st.divider()
    st.subheader("🗂️ Knowledge Graph")

    mem_uid = user_id_input.strip() or st.session_state.user_id
    col_mem1, col_mem2 = st.columns(2)

    with col_mem1:
        if st.button("🔍 View Memory", use_container_width=True, disabled=not mem_uid):
            with st.spinner("Reading Neo4j..."):
                mem = get_memory(mem_uid)
            if mem:
                if mem.get("has_data"):
                    st.session_state["_mem_display"] = mem["context"]
                else:
                    st.session_state["_mem_display"] = "_No memory stored yet for this user._"
            else:
                st.session_state["_mem_display"] = "_Could not fetch memory._"

    with col_mem2:
        if st.button("🗑️ Clear Memory", use_container_width=True, disabled=not mem_uid, type="secondary"):
            st.session_state["_confirm_clear"] = True

    # Confirmation step — avoids accidental wipe
    if st.session_state.get("_confirm_clear"):
        st.warning(f"Delete ALL memory for **{mem_uid}**? This cannot be undone.")
        cc1, cc2 = st.columns(2)
        with cc1:
            if st.button("✅ Yes, clear it", use_container_width=True, type="primary"):
                try:
                    r = requests.delete(f"{API_BASE}/memory/{mem_uid}", timeout=10)
                    r.raise_for_status()
                    st.success("Memory cleared.")
                    st.session_state.pop("_mem_display", None)
                except Exception as e:
                    st.error(f"Failed: {e}")
                st.session_state["_confirm_clear"] = False
                st.rerun()
        with cc2:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state["_confirm_clear"] = False
                st.rerun()

    if "_mem_display" in st.session_state:
        with st.expander("Memory Context", expanded=True):
            st.markdown(st.session_state["_mem_display"])


# ── Main chat area ────────────────────────────────────────────────────────────
st.title("🧠 MemoryOS Chat")

if not st.session_state.session_id:
    # Landing state — no session yet
    st.info("👈 Enter your User ID and click **New Session** to start chatting.")

    st.markdown("### How it works")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**1️⃣ Start a session**")
        st.caption("Enter any User ID and click New Session.")
    with col_b:
        st.markdown("**2️⃣ Chat & teach it**")
        st.caption("Tell it your name, projects, preferences. It stores everything.")
    with col_c:
        st.markdown("**3️⃣ End & restart**")
        st.caption("End the session. Start a new one. Ask 'What do you know about me?' — it remembers.")

    st.divider()
    st.markdown("### Try these messages")
    st.code(
        "I prefer Python over JavaScript\n"
        "I'm working on a project called AlphaSort\n"
        "My manager is Sarah\n"
        "Remind me to submit the report by Friday\n"
        "What do you know about me?",
        language=None,
    )

else:
    # ── Chat header ──
    session_label = "⛔ Session ended — start a new session to keep chatting" \
        if st.session_state.session_ended else \
        f"Session: `{st.session_state.session_id}`"
    st.caption(session_label)

    # ── Memory banner — shown after first ended session for same user ──
    if st.session_state.past_sessions and st.session_state.session_id not in st.session_state.past_sessions:
        mem = get_memory(st.session_state.user_id)
        if mem and mem.get("has_data"):
            with st.expander("🗂️ What I remember about you (from previous sessions)", expanded=False):
                st.markdown(mem["context"])

    # ── Chat history ──
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.markdown(
                "<div style='text-align:center; color:#888; padding: 60px 0;'>"
                "Send a message to start the conversation."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🧠"):
                    st.markdown(msg["content"])

    # ── Input ──
    if st.session_state.session_ended:
        st.warning("This session has ended. Click **New Session** in the sidebar to continue.")
    else:
        if prompt := st.chat_input("Message MemoryOS..."):
            # Show user message immediately
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)

            # Get agent response
            with st.chat_message("assistant", avatar="🧠"):
                with st.spinner("Thinking..."):
                    reply = send_message(prompt)
                if reply:
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                else:
                    st.error("No response from agent.")
