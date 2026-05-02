import uuid
from datetime import datetime, timezone
from typing import Optional


class SessionData:
    def __init__(self, session_id: str, user_id: str, agent):
        self.session_id = session_id
        self.user_id = user_id
        self.agent = agent
        self.created_at = datetime.now(timezone.utc).isoformat()


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, SessionData] = {}

    def create_session(self, user_id: str) -> SessionData:
        from examples.api.main import memory
        from examples.agent import get_agent

        session_id = f"session_{uuid.uuid4().hex[:12]}"
        agent = get_agent(memory)
        session = SessionData(session_id=session_id, user_id=user_id, agent=agent)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[SessionData]:
        return self._sessions.get(session_id)

    def remove_session(self, session_id: str) -> Optional[SessionData]:
        return self._sessions.pop(session_id, None)


session_manager = SessionManager()
