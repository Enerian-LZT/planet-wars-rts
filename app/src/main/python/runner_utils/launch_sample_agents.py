from pathlib import Path
from runner_utils.agent_entry import sample_entries
from runner_utils.launch_agent import launch_agent

if __name__ == "__main__":

    base_dir = Path("/tmp/simonl-planetwars-run")

    for agent in sample_entries:
        print(f"Launching agent: {agent.id}")
        launch_agent(agent, base_dir)
        print(f"Agent {agent.id} launched successfully.")

