# ruff: noqa: F403
import argparse
import sys
from typing import Type

from code_agents.agents import *
from code_agents.agents_core import AgentCommand, CodeAgent

# Dynamic discovery of subclasses
agent_subclasses = CodeAgent.__subclasses__()
agent_subclass_dict = {cls.__name__.lower(): cls for cls in agent_subclasses}
agent_subclass_names = list(agent_subclass_dict.keys())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="code-agents",
        description="Run code agents inside a secure Docker sandbox.",
        usage="%(prog)s <agent> [command...]",
    )
    parser.add_argument(
        "agent",
        help=f"Agent class to run (one of: {', '.join(sorted(agent_subclass_names))})",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Optional shell command to run inside the agent sandbox",
    )

    args = parser.parse_args(argv)

    agent_name = args.agent.lower()
    Agent: Type[CodeAgent] = agent_subclass_dict.get(agent_name)

    if Agent is None:
        parser.error(f"Unknown agent '{args.agent}'. Available: {', '.join(sorted(agent_subclass_names))}")

    agent: CodeAgent = Agent()
    Command: Type[AgentCommand] = agent.get_command_class()
    command = Command()

    if not args.command:
        # `<cli-app> <code-agent>` -> use Pydantic defaults of the command class
        agent.run(command=command)
    else:
        # `<cli-app> <code-agent> <custom args>` -> use raw command string
        raw_cmd = " ".join([command.executable, *args.command, *DEFAULT_ARGS_AIDER]) # noqa
        agent.run(raw_cmd=raw_cmd)


if __name__ == "__main__":
    main(sys.argv[1:])
