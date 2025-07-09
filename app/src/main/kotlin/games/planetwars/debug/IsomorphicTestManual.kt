package games.planetwars.debug


import games.planetwars.agents.Action
import games.planetwars.agents.random.BetterRandomAgent
import games.planetwars.agents.random.PureRandomAgent
import games.planetwars.core.*
import games.planetwars.runners.GameRunner

class IsomorphicTest {
    fun runTest(): Boolean {
        val gameParams = GameParams(numPlanets = 20)
        val agent1 = PureRandomAgent()
        val agent2 = BetterRandomAgent()

        val gameRunner = GameRunner(agent1, agent2, gameParams)

        // Run the first game and record actions
        val initialGameState = gameRunner.gameState.deepCopy()
        val actionHistory = mutableListOf<Map<Player, Action>>()

        while (!gameRunner.forwardModel.isTerminal()) {
            val actions = mapOf(
                Player.Player1 to agent1.getAction(gameRunner.forwardModel.state.deepCopy()),
                Player.Player2 to agent2.getAction(gameRunner.forwardModel.state.deepCopy())
            )
            actionHistory.add(actions)
            gameRunner.forwardModel.step(actions)
        }
        val finalState1 = gameRunner.forwardModel.state.deepCopy()

        // Run the second game with the same initial state and replay actions
        val gameRunner2 = GameRunner(agent1, agent2, gameParams)
        gameRunner2.gameState = initialGameState.deepCopy()
        gameRunner2.forwardModel = ForwardModel(gameRunner2.gameState.deepCopy(), gameParams)

        for (actions in actionHistory) {
            gameRunner2.forwardModel.step(actions)
        }
        val finalState2 = gameRunner2.forwardModel.state

        // Compare the two final states using DeepEquals
        return finalState1.deepEquals(finalState2)
    }
}

fun main() {
    val test = IsomorphicTest()
    val result = test.runTest()
    println("Isomorphic test passed: $result")
}
