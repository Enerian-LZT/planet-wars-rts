package games.planetwars.agents

import games.planetwars.agents.*
import games.planetwars.agents.random.RHEAAgent
import games.planetwars.core.*
import kotlin.math.*
import kotlin.random.Random

class HybridSmartAgent : PlanetWarsPlayer() {
    // critic参数
    private var criticWeights = doubleArrayOf(0.0, 0.0, 0.0)
    private val criticAlpha = 0.1
    private val criticGamma = 0.9

    // 阶段参数
    private var enemyPriorityCoef = 1.0
    private var neutralPriorityCoef = 1.0
    private var distancePower = 1.2
    private var candidateLimit = 40
    private var stage = "early"
    private var ticksSinceNeutralAttack = 0
    private var lastPlanetOwners: MutableMap<Int, Player>? = null
    private val random = Random

    // 兜底：RHEA或其它
    private lateinit var fallbackAgent: PlanetWarsPlayer

    override fun prepareToPlayAs(player: Player, params: GameParams, opponent: String?): String {
        super.prepareToPlayAs(player, params, opponent)
        fallbackAgent = RHEAAgent(sequenceLength = 20, useShiftBuffer = true)
        fallbackAgent.prepareToPlayAs(player, params, opponent)
        return getAgentType()
    }

    override fun getAgentType() = "Hybrid Smart Agent v1"

    override fun getAction(gameState: GameState): Action {
        // 初始化归属映射
        if (lastPlanetOwners == null) {
            lastPlanetOwners = gameState.planets.associate { it.id to it.owner }.toMutableMap()
        }

        // 阶段判断与参数调整
        updateStageAndParams(gameState)

        // 统计本回合敌方新占领星球
        val capturedByEnemy = gameState.planets.filter {
            lastPlanetOwners!![it.id] != player.opponent() && it.owner == player.opponent()
        }
        gameState.planets.forEach { lastPlanetOwners!![it.id] = it.owner }

        val myPlanets = gameState.planets.filter { it.owner == player && it.transporter == null && it.nShips > 3 }
        if (myPlanets.isEmpty()) return Action.DO_NOTHING

        val enemyPlanets = gameState.planets.filter { it.owner == player.opponent() }
        val neutralPlanets = gameState.planets.filter { it.owner == Player.Neutral }

        // 是否考虑进攻中立星球
        var considerNeutral = false
        if (neutralPlanets.isNotEmpty()) {
            ticksSinceNeutralAttack++
            if (ticksSinceNeutralAttack >= 5) considerNeutral = true
        }

        // 候选行动
        val candidatePairs = mutableListOf<Pair<Action, Double>>()

        // ---- 敌方目标候选 ----
        if (enemyPlanets.isNotEmpty()) {
            val priorityEnemies = if (capturedByEnemy.isNotEmpty()) capturedByEnemy else enemyPlanets
            val avgEnemyRadius = priorityEnemies.map { it.radius }.average()
            val capturedBoost = 1.5
            for (src in myPlanets) {
                for (tgt in priorityEnemies) {
                    val dist = src.position.distance(tgt.position)
                    if (dist <= 0.0) continue
                    var radiusFactor = tgt.radius
                    if (capturedByEnemy.contains(tgt) && abs(tgt.radius - avgEnemyRadius) / avgEnemyRadius < 0.1)
                        radiusFactor *= capturedBoost
                    if (tgt.transporter != null) radiusFactor *= 1.3
                    val threatBonus = predictOpponentMoves(tgt, gameState)
                    val weight = (radiusFactor / dist.pow(distancePower)) * enemyPriorityCoef + threatBonus
                    val ships = (src.nShips * 0.6).coerceAtLeast(1.0)
                    candidatePairs.add(Action(player, src.id, tgt.id, ships) to weight)
                }
            }
        }

        // ---- 中立目标候选 ----
        if (considerNeutral && neutralPlanets.isNotEmpty()) {
            for (src in myPlanets) {
                for (tgt in neutralPlanets) {
                    val dist = src.position.distance(tgt.position)
                    if (dist <= 0.0) continue
                    val weight = (tgt.radius / dist.pow(distancePower)) * neutralPriorityCoef
                    val required = tgt.nShips * 1.1
                    val ships = min(src.nShips * 0.2, required).coerceAtLeast(1.0)
                    candidatePairs.add(Action(player, src.id, tgt.id, ships) to weight)
                }
            }
        }

        // ---- 防守弱点（资源管理，Greedy思想）----
        val weakPlanets = myPlanets.filter { it.nShips < 10 }
        if (weakPlanets.isNotEmpty()) {
            val weakest = weakPlanets.minByOrNull { it.nShips }!!
            val strongest = myPlanets.maxByOrNull { it.nShips }!!
            val numToSend = strongest.nShips / 4
            candidatePairs.add(Action(player, strongest.id, weakest.id, numToSend) to 50.0)
        }

        if (candidatePairs.isEmpty()) return fallbackAgent.getAction(gameState)

        // 启发式排序top-N
        val topCandidates = candidatePairs.sortedByDescending { it.second }.take(candidateLimit)

        // RL评估（actor-critic）
        var bestScore = Double.NEGATIVE_INFINITY
        var bestAction = topCandidates.first().first
        for ((act, _) in topCandidates) {
            val score = evaluateActionActorCritic(gameState, act)
            if (score > bestScore) {
                bestScore = score
                bestAction = act
            }
        }

        // 若为中立攻击则重置计数
        val bestTarget = gameState.planets[bestAction.destinationPlanetId]
        if (bestTarget.owner == Player.Neutral) ticksSinceNeutralAttack = 0

        return bestAction
    }

    // ===========================
    // 阶段与参数自适应调整
    private fun updateStageAndParams(gameState: GameState) {
        val totalShips = gameState.planets.sumOf {
            when (it.owner) {
                player -> it.nShips
                player.opponent() -> -it.nShips
                else -> 0.0
            }
        }
        stage = when {
            totalShips < 200 -> "early"
            totalShips < 600 -> "mid"
            else -> "late"
        }
        when (stage) {
            "early" -> {
                enemyPriorityCoef = 2.0; neutralPriorityCoef = 2.0; distancePower = 1.1; candidateLimit = 30
            }
            "mid" -> {
                enemyPriorityCoef = 1.5; neutralPriorityCoef = 1.0; distancePower = 1.3; candidateLimit = 40
            }
            "late" -> {
                enemyPriorityCoef = 1.2; neutralPriorityCoef = 0.5; distancePower = 1.5; candidateLimit = 50
            }
        }
    }

    // ===========================
    // 对手行为威胁预测（简化自GreedyLookahead）
    private fun predictOpponentMoves(target: Planet, gameState: GameState): Double {
        val opponentPlanets = gameState.planets.filter { it.owner == player.opponent() }
        var predictedThreatLevel = 0.0
        if (target.owner == Player.Neutral) predictedThreatLevel += 8.0
        for (opp in opponentPlanets) {
            val dist = target.position.distance(opp.position)
            if (dist < 10.0) predictedThreatLevel += 12.0
        }
        return predictedThreatLevel
    }

    // ===========================
    // Actor-Critic评估核心
    private fun evaluateActionActorCritic(state: GameState, action: Action): Double {
        val featuresBefore = featureVector(state)
        val valueBefore = criticValue(featuresBefore)

        val fm = ForwardModel(state.deepCopy(), params)
        val oppA0 = fallbackAgent.getAction(state)
        fm.step(mapOf(player to action, player.opponent() to oppA0))

        var steps = 0
        while (steps < 30 && !fm.isTerminal()) {
            val myAct = randomGreedyAction(fm.state)
            val oppAct = fallbackAgent.getAction(fm.state)
            fm.step(mapOf(player to myAct, player.opponent() to oppAct))
            steps++
        }

        val reward = (fm.getShips(player) - fm.getShips(player.opponent())) -
                (state.planets.filter { it.owner == player }.sumOf { it.nShips } -
                        state.planets.filter { it.owner == player.opponent() }.sumOf { it.nShips })

        val featuresAfter = featureVector(fm.state)
        val valueAfter = criticValue(featuresAfter)
        val target = reward + criticGamma * valueAfter
        val tdError = target - valueBefore

        for (i in criticWeights.indices) {
            criticWeights[i] += criticAlpha * tdError * featuresBefore[i]
        }

        return target
    }

    // 特征：兵力差、星球差、常数
    private fun featureVector(state: GameState): DoubleArray {
        val shipDiff = state.planets.filter { it.owner == player }.sumOf { it.nShips } -
                state.planets.filter { it.owner == player.opponent() }.sumOf { it.nShips }
        val planetDiff = state.planets.count { it.owner == player } -
                state.planets.count { it.owner == player.opponent() }
        return doubleArrayOf(shipDiff, planetDiff.toDouble(), 1.0)
    }

    private fun criticValue(features: DoubleArray): Double {
        var v = 0.0
        for (i in criticWeights.indices) v += criticWeights[i] * features[i]
        return v
    }

    // rollout中的快速策略
    private fun randomGreedyAction(state: GameState): Action {
        val myPlanets = state.planets.filter { it.owner == player && it.transporter == null && it.nShips > 3 }
        if (myPlanets.isEmpty()) return Action.DO_NOTHING
        val source = myPlanets.maxByOrNull { it.nShips } ?: return Action.DO_NOTHING
        val neutralTargets = state.planets.filter { it.owner == Player.Neutral }
        val enemyTargets = state.planets.filter { it.owner == player.opponent() }
        val target = when {
            neutralTargets.isNotEmpty() ->
                neutralTargets.maxByOrNull { it.growthRate / source.position.distance(it.position) }!!
            enemyTargets.isNotEmpty() ->
                enemyTargets.minByOrNull {
                    val dist = source.position.distance(it.position)
                    val futureDef = it.nShips + it.growthRate * (dist / params.transporterSpeed)
                    futureDef
                }!!
            else -> return Action.DO_NOTHING
        }
        val dist = source.position.distance(target.position)
        val travelTicks = dist / params.transporterSpeed
        val shipsToSend = if (target.owner == Player.Neutral) {
            (target.nShips * 1.1).coerceAtLeast(1.0)
        } else {
            val def = target.nShips + target.growthRate * travelTicks
            (def * 1.3).coerceAtMost(source.nShips * 0.8).coerceAtLeast(1.0)
        }
        return Action(player, source.id, target.id, shipsToSend)
    }
}
