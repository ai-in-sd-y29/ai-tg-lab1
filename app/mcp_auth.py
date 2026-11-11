"""Capability для одного подтверждённого действия, вне аргументов модели."""

import hashlib
import hmac
import uuid

from app.lab2_models import AgentError


def capability(secret: str, action_id: str) -> dict:
    action_id = str(uuid.UUID(action_id))
    signature = hmac.new(secret.encode(), action_id.encode(), hashlib.sha256).hexdigest()
    return {"action_id": action_id, "signature": signature}


def verify_capability(secret: str, meta: dict) -> str:
    try:
        identity = str(uuid.UUID(meta["action_id"]))
        expected = capability(secret, identity)["signature"]
        if len(secret) < 32 or not hmac.compare_digest(expected, meta["signature"]):
            raise ValueError
        return identity
    except (KeyError, TypeError, ValueError):
        raise AgentError("trusted_context_required") from None
