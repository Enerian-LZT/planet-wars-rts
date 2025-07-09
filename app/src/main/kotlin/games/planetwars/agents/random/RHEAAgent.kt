package games.planetwars.agents.random

import games.planetwars.agents.Action
import games.planetwars.agents.PlanetWarsPlayer
import games.planetwars.agents.evo.GameStateWrapper
import games.planetwars.core.*
import kotlin.math.roundToInt
import kotlin.random.Random

class RHEAAgent(
    private val sequenceLength: Int = 200,
    private val populationSize: Int = 80,
    private val generations: Int = 50,
    private val initialMutationRate: Double = 0.25,
    private val eliteCount: Int = 8,
    private val useShiftBuffer: Boolean = true,
    private val useParallelEval: Boolean = true
) : PlanetWarsPlayer() {

    private var bestSolution: FloatArray? = null
    private var mutationRate: Double = initialMutationRate
    private val epsilon = 1e-7f

    override fun getAction(gameState: GameState): Action {
        val myPlanets = gameState.planets.filter { it.owner == player && it.transporter == null }
        val targetPlanets = gameState.planets.filter { it.owner != player }

        if (myPlanets.isEmpty() || targetPlanets.isEmpty()) return Action.doNothing()

        val floatLength = sequenceLength * 2

        // Seed/shift the solution
        bestSolution = when {
            bestSolution == null || !useShiftBuffer -> smartSeedSequence(myPlanets, targetPlanets, sequenceLength)
            else -> shiftAndMutate(bestSolution!!, mutationRate)
        }

        var topScore = evaluateFloatSequence(bestSolution!!, gameState.deepCopy())
        var topSolution = bestSolution!!

        // Main evolutionary loop
        repeat(generations) {
            val population = Array(populationSize) { mutateFloatSequence(topSolution, mutationRate) }
            // Elitism: include previous best & a greedy seed
            population[0] = topSolution
            population[1] = smartSeedSequence(myPlanets, targetPlanets, sequenceLength)
            // Optionally: parallel evaluation
            val evaluated = if (useParallelEval) {
                population.asList().parallelStream().map { it to evaluateFloatSequence(it, gameState.deepCopy()) }.toList()
            } else {
                population.map { it to evaluateFloatSequence(it, gameState.deepCopy()) }
            }
            val sorted = evaluated.sortedByDescending { it.second }
            val elites = sorted.take(eliteCount).map { it.first.copyOf() }
            topSolution = elites.first()
            topScore = sorted.first().second
            // Adaptive mutation: reduce as we converge
            mutationRate = (mutationRate * 0.98).coerceAtLeast(0.05)
        }

        // Decode the best solution into an action
        val fromIndex = ((topSolution[0] * myPlanets.size).roundToInt()).coerceIn(0, myPlanets.lastIndex)
        val toIndex = ((topSolution[1] * targetPlanets.size).roundToInt()).coerceIn(0, targetPlanets.lastIndex)
        val source = myPlanets.getOrNull(fromIndex) ?: myPlanets.first()
        val target = targetPlanets.getOrNull(toIndex) ?: targetPlanets.first()
        val ships = (source.nShips * 0.5).coerceAtLeast(1.0)
        return Action(player, source.id, target.id, ships)
    }

    private fun randomFloatSequence(length: Int): FloatArray = FloatArray(length) { Random.nextFloat() }

    private fun mutateFloatSequence(seq: FloatArray, mutRate: Double): FloatArray {
        val out = FloatArray(seq.size)
        for (i in seq.indices) {
            out[i] = if (Random.nextDouble() < mutRate) Random.nextFloat() else seq[i]
        }
        return out
    }

    private fun shiftAndMutate(seq: FloatArray, mutRate: Double): FloatArray {
        val shifted = FloatArray(seq.size)
        // Shift the sequence, but keep the last two random
        for (i in 0 until seq.size - 2) {
            shifted[i] = seq[i + 2]
        }
        shifted[seq.size - 2] = Random.nextFloat()
        shifted[seq.size - 1] = Random.nextFloat()
        // Mutate
        for (i in shifted.indices) {
            if (Random.nextDouble() < mutRate) shifted[i] = Random.nextFloat()
        }
        return shifted
    }

    private fun smartSeedSequence(myPlanets: List<Planet>, targetPlanets: List<Planet>, length: Int): FloatArray {
        val fromIndex = myPlanets.indexOf(myPlanets.maxByOrNull { it.nShips } ?: myPlanets.random())
        val toIndex = targetPlanets.indexOf(targetPlanets.minByOrNull { it.nShips } ?: targetPlanets.random())
        val arr = FloatArray(length * 2)
        for (i in arr.indices step 2) {
            arr[i] = fromIndex / myPlanets.size.toFloat() + epsilon // avoid exact 0/1
            arr[i + 1] = toIndex / targetPlanets.size.toFloat() + epsilon
        }
        // Some diversity: sprinkle in some randomness in the last third
        for (i in arr.size * 2 / 3 until arr.size) {
            arr[i] = Random.nextFloat()
        }
        return arr
    }

    private fun evaluateFloatSequence(seq: FloatArray, state: GameState): Double {
        val wrapper = GameStateWrapper(state, GameParams(), player)
        // Encourage planet ownership and penalize inaction
        return wrapper.runForwardModel(seq) + 0.01 * seq.count { it > 0.5 }
    }

    override fun getAgentType(): String = "RHEA Agent (Optimized)"
}

fun main() {
    val agent = RHEAAgent()
    agent.prepareToPlayAs(Player.Player1, GameParams())
    val gameState = GameStateFactory(GameParams()).createGame()
    val action = agent.getAction(gameState)
    println(action)
}

