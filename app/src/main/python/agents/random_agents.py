import random
from typing import Optional

from agents.planet_wars_agent import PlanetWarsPlayer
from core.game_state import GameState, Action, Player, GameParams
from core.game_state_factory import GameStateFactory


class CarefulRandomAgent(PlanetWarsPlayer):
    def get_action(self, game_state: GameState) -> Action:
        # Filter the planets owned by the player and without a transporter
        my_planets = [p for p in game_state.planets if p.owner == self.player and p.transporter is None]
        if not my_planets:
            return Action.do_nothing()

        # Filter opponent planets
        opponent_planets = [p for p in game_state.planets if p.owner == self.player.opponent()]
        if not opponent_planets:
            return Action.do_nothing()

        source = random.choice(my_planets)
        target = random.choice(opponent_planets)

        return Action(
            player_id=self.player,
            source_planet_id=source.id,
            destination_planet_id=target.id,
            num_ships=source.n_ships / 2
        )

    def get_agent_type(self) -> str:
        return "Careful Random Agent"


# Example usage
if __name__ == "__main__":
    agent = CarefulRandomAgent()
    agent.prepare_to_play_as(Player.Player1, GameParams())
    game_state = GameStateFactory(GameParams()).create_game()
    action = agent.get_action(game_state)
    print(action)
