package games.planetwars.debug

import games.planetwars.agents.random.BetterRandomAgent
import games.planetwars.agents.random.CarefulRandomAgent
import games.planetwars.core.*
import games.planetwars.runners.GameRunner


fun main() {
    val initialGameState = GameStateFactory(GameParams(numPlanets = 20)).createGame()
    val copiedState = initialGameState.deepCopy()

    // now run some game steps
    val gameRunner = GameRunner(
        CarefulRandomAgent(),
        BetterRandomAgent(),
        GameParams(numPlanets = 20),
    )


    val finalModel = gameRunner.runGame()
    gameRunner.runGames(10)
    println(finalModel.statusString())


    if (!initialGameState.deepEquals(copiedState)) {
        println("Initial game state and copied state do not match!")
    } else {
        println("Initial game state and copied state match perfectly.")
    }
}


fun GameState.deepEquals(other: GameState): Boolean {
    if (this === other) return true
    if (this.gameTick != other.gameTick) {
        println("Mismatch in gameTick: ${this.gameTick} vs ${other.gameTick}")
        return false
    }
    if (this.planets.size != other.planets.size) {
        println("Mismatch in number of planets: ${this.planets.size} vs ${other.planets.size}")
        return false
    }
    for (i in planets.indices) {
        val p1 = planets[i]
        val p2 = other.planets[i]
        if (!p1.deepEquals(p2)) {
            println("Mismatch in planet at index $i: $p1 vs $p2")
            return false
        }
    }
    return true
}

fun Planet.deepEquals(other: Planet): Boolean {
    if (this === other) return true
    if (this.owner != other.owner) {
        println("Mismatch in owner: ${this.owner} vs ${other.owner}")
        return false
    }
    if (this.nShips != other.nShips) {
        println("Mismatch in nShips: ${this.nShips} vs ${other.nShips}")
        return false
    }
    if (this.position != other.position) {
        println("Mismatch in position: ${this.position} vs ${other.position}")
        return false
    }
    if (this.growthRate != other.growthRate) {
        println("Mismatch in growthRate: ${this.growthRate} vs ${other.growthRate}")
        return false
    }
    if (this.radius != other.radius) {
        println("Mismatch in radius: ${this.radius} vs ${other.radius}")
        return false
    }
    if (this.id != other.id) {
        println("Mismatch in planet ID: ${this.id} vs ${other.id}")
        return false
    }
    if (!this.transporter.deepEquals(other.transporter)) {
        println("Mismatch in transporter: ${this.transporter} vs ${other.transporter}")
        return false
    }
    return true
}

fun Transporter?.deepEquals(other: Transporter?): Boolean {
    if (this === other) return true
    if (this == null || other == null) return this == other
    if (this.s != other.s) {
        println("Mismatch in transporter position: ${this.s} vs ${other.s}")
        return false
    }
    if (this.v != other.v) {
        println("Mismatch in transporter velocity: ${this.v} vs ${other.v}")
        return false
    }
    if (this.owner != other.owner) {
        println("Mismatch in transporter owner: ${this.owner} vs ${other.owner}")
        return false
    }
    if (this.destinationIndex != other.destinationIndex) {
        println("Mismatch in transporter destinationIndex: ${this.destinationIndex} vs ${other.destinationIndex}")
        return false
    }
    if (this.nShips != other.nShips) {
        println("Mismatch in transporter nShips: ${this.nShips} vs ${other.nShips}")
        return false
    }
    return true
}
