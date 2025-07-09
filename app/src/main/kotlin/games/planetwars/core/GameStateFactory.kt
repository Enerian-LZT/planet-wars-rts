package games.planetwars.core

import util.Vec2d

class GameStateFactory (val params: GameParams) {

    fun makeRandomPlanet(params: GameParams, owner: Player) : Planet {
        // we only use x in the left half of the screen
        val x = (Math.random() * params.width / 2).toInt()
        val y = (Math.random() * params.height).toInt()
        val numShips = (Math.random() * (params.maxInitialShipsPerPlanet - params.minInitialShipsPerPlanet)
                + params.minInitialShipsPerPlanet)
        val growthRate = Math.random() * (params.maxGrowthRate - params.minGrowthRate) + params.minGrowthRate
        val radius = growthRate * params.growthToRadiusFactor
        return Planet(owner, numShips, Vec2d(x.toDouble(), y.toDouble()), growthRate, radius)
    }

    fun canAdd(planets: List<Planet>, candidate: Planet, radialSeparation: Double) : Boolean {
        // check they are not too close to the edge
        val edgeSep = params.edgeSeparation
        if (candidate.position.x - edgeSep < candidate.radius ||
            candidate.position.x + edgeSep > params.width / 2 - candidate.radius) {
            return false
        }
        if (candidate.position.y - edgeSep < candidate.radius ||
            candidate.position.y + edgeSep > params.height - candidate.radius) {
            return false
        }
        for (planet in planets) {
            val planetRadius = planet.growthRate * params.growthToRadiusFactor
            // check various constraints
            // first that two planets are not too close
            if (planet.position.distance(candidate.position) < radialSeparation * (planetRadius + candidate.radius)) {
                return false
            }
        }
        return true
    }

    fun createGame() : GameState {
        val planets = mutableListOf<Planet>()
        // make allocation symmetrical around the vertical centerline
        // when reflecting them also reflect in Y to make for a more interesting looking map
        // now to do the allocation, allocate half randomly within the left half of the screen
        // and then their reflections on the right half
        // the neutral ones are duplicated directly, while the player allocated ones are switched in allocation

        val nNeutral = (params.numPlanets * params.initialNeutralRatio).toInt() / 2
        while (planets.size < params.numPlanets/2) {
            val player = if (planets.size < nNeutral) Player.Neutral else Player.Player1
            val candidate = makeRandomPlanet(params, player)
            if (canAdd(planets, candidate, params.radialSeparation)) {
                planets.add(candidate)
            }
        }
        // now create a reflected set of planets
        val reflectedPlanets = mutableListOf<Planet>()
        for (planet in planets) {
            val reflected = planet.copy(
                position = Vec2d(params.width - planet.position.x, params.height - planet.position.y)
            )
            if (planet.owner == Player.Player1) {
                reflected.owner = Player.Player2
            }
            reflectedPlanets.add(reflected)
        }

        // now add the reflected planets to the main list
        planets.addAll(reflectedPlanets)

        // now number the ids
        for ((i, planet) in planets.withIndex()) {
            planet.id = i
        }
        return GameState(planets)
    }
}

fun main() {
    val params = GameParams()
    val factory = GameStateFactory(params)
    val game = factory.createGame()
    println(game)
    for (planet in game.planets) {
        println(planet)
    }
}
