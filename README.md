# gym-bridge
The gym-bridge is multi agent environment based on OpenAI gym.
Currently two types of environment are supported:
## BridgeEnv
At each timestep only one agent ("active_player") takes action.

User can get "active_player" by env.get_active_player().
## BridgeSimultaneousActionsEnv
At each timestep all agents take actions. Only "active_player" should play a card at the time,
other agents should take "empty" actions.
## Examples of usage
All examples are stored in gym_tests directory. Files description:
* agents.py - three types of agents are implemented (AgentRandom, AgentDummy and DQNAgent).
To use DQNAgent *tensorflow* library is necessary.
* random_agent_play.py - three players are instances of AgentRandom and one (dummy) is an AgentDummy instance.
You can use this script to check if environment works as intended.
* train_agent.py - train DQNAgent to play only available cards. Other players are instances of AgentRandom.
* evaluate_agent.py - evaluate trained DQNAgent.
# Installation
```bash
cd gym-bridge
pip install -e .
```