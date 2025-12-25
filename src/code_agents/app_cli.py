# ruff: noqa: F403
from code_agents.agents import *
from code_agents.agents_core import AgentCommand, CodeAgent
import argparse

# --------- Dynamic discovery of subclasses --------- #
agent_subclasses = CodeAgent.__subclasses__()
agent_subclass_dict = {cls.__name__: cls for cls in agent_subclasses}
agent_subclass_names = list(agent_subclass_dict.keys())

if __name__ == "__main__":
    
# required features
# 1. argparse: 
#       parse first arg as agent subclass (map names to lowercase), return error if not found
#       -h --help shows available agents
#       case 1: <module> <agent_subclass> (without args) -> launch CLI with defaults: CodeAgent.run(command)
#       case 2: <module> <agent_subclass> [args...] -> launch CLI with: CodeAgent.run_cli(args)
# 2. always initialize CodeAgent with empty args to avoid repo cloning
