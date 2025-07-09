package games.planetwars.debug


import games.planetwars.core.GameState
import games.planetwars.core.Planet
import games.planetwars.core.Player
import kotlinx.serialization.encodeToString
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import util.Vec2d

fun main() {
    val gameState = GameState(planets = listOf(
        Planet(
            owner = Player.Player1,
            nShips = 42.0,
            position = Vec2d(1.0, 2.0),
            growthRate = 1.0,
            radius = 5.0,
            id = 0
        )
    ))

    val json = Json.encodeToString(gameState)
    println("Serialized JSON:\n$json")

    val parsed = Json.decodeFromString<GameState>(json)
    println("\nDeserialized object:\n$parsed")

    // make a deep equals check
    println("\nDeep equals check: ${gameState == parsed}")

    val modified = parsed.copy(gameTick = 99)
    println("Equal after change? ${gameState == modified}") // Should be false
}
