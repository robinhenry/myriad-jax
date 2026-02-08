from .agent import Agent, AgentParams, AgentState
from .classical import bangbang, pid, random
from .registration import AgentInfo, get_agent_info, list_agents, make_agent, register_agent
from .rl import dqn, pqn

# Register built-in agents
register_agent("random", random.make_agent)
register_agent("bangbang", bangbang.make_agent)
register_agent("pid", pid.make_agent)
register_agent("dqn", dqn.make_agent, is_off_policy=True)
register_agent("pqn", pqn.make_agent, is_on_policy=True)

__all__ = [
    "make_agent",
    "Agent",
    "AgentParams",
    "AgentState",
    "AgentInfo",
    "register_agent",
    "get_agent_info",
    "list_agents",
]
