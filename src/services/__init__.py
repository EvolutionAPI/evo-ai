# google-adk is an optional dependency — guard so unit tests run without the full stack
try:
    from .adk.agent_runner import run_agent
except ImportError:
    pass
