package games.planetwars.agents.random

import games.planetwars.agents.Action
import games.planetwars.agents.DoNothingAgent
import games.planetwars.agents.PlanetWarsAgent
import games.planetwars.agents.PlanetWarsPlayer
import games.planetwars.core.*
import kotlin.math.min
import kotlin.math.max
import kotlin.math.pow
import kotlin.random.Random

/**
 * 混合策略智能体 - 结合贪心前瞻和进化算法的优点
 * 核心思想：用贪心策略快速筛选候选动作，用进化算法优化长期规划
 */
class hybrid : PlanetWarsPlayer() {

    override fun getAgentType(): String = "Hybrid Strategy Agent"

    // ========== 全局策略参数 ==========
    private var gamePhase = GamePhase.EARLY_EXPANSION
    private var lastPlanetOwners: MutableMap<Int, Player>? = null
    private var ticksSinceNeutralAttack = 0

    // ========== 进化算法参数 ==========
    private val sequenceLength = 100
    private val nEvals = 15
    private val mutationProb = 0.4
    private val rolloutDepth = 30
    private val nRollouts = 6

    // ========== 策略权重 ==========
    private var expansionWeight = 2.0
    private var attackWeight = 1.5
    private var defenseWeight = 1.2
    private var distancePenalty = 0.8
    private var growthBonus = 1.8
    private var safetyBuffer = 1.3

    // ========== 评估阈值 ==========
    private val criticalActionThreshold = 0.7
    private val neutralAttackInterval = 5

    enum class GamePhase {
        EARLY_EXPANSION,    // 早期扩张：优先占领中立星球
        MID_COMPETITION,    // 中期竞争：平衡攻防
        LATE_DOMINATION     // 后期统治：集中优势兵力
    }

    override fun getAction(gameState: GameState): Action {
        // 初始化行星归属追踪
        if (lastPlanetOwners == null) {
            lastPlanetOwners = gameState.planets.associate { it.id to it.owner }.toMutableMap()
        }

        // 更新游戏阶段和策略参数
        updateGamePhase(gameState)
        updatePlanetOwners(gameState)

        // 获取可用的源星球和目标星球
        val myPlanets = gameState.planets.filter { it.owner == player && it.transporter == null && it.nShips > 10 }
        val neutralPlanets = gameState.planets.filter { it.owner == Player.Neutral }
        val enemyPlanets = gameState.planets.filter { it.owner == player.opponent() }

        if (myPlanets.isEmpty()) {
            return Action.doNothing()
        }

        // 评估局面复杂度，决定使用贪心还是进化策略
        val situationComplexity = evaluateSituationComplexity(gameState)

        if (situationComplexity > criticalActionThreshold) {
            // 复杂局面：使用进化算法深度搜索
            return selectActionWithEvolution(gameState, myPlanets, neutralPlanets, enemyPlanets)
        } else {
            // 简单局面：使用改进的贪心策略
            return selectActionWithGreedy(gameState, myPlanets, neutralPlanets, enemyPlanets)
        }
    }

    private fun updateGamePhase(gameState: GameState) {
        val totalPlanets = gameState.planets.size
        val neutralCount = gameState.planets.count { it.owner == Player.Neutral }
        val myCount = gameState.planets.count { it.owner == player }
        val enemyCount = gameState.planets.count { it.owner == player.opponent() }

        val neutralRatio = neutralCount.toDouble() / totalPlanets
        val myRatio = myCount.toDouble() / totalPlanets

        gamePhase = when {
            neutralRatio > 0.4 -> GamePhase.EARLY_EXPANSION
            myRatio > 0.6 || enemyCount.toDouble() / totalPlanets > 0.6 -> GamePhase.LATE_DOMINATION
            else -> GamePhase.MID_COMPETITION
        }

        // 根据阶段调整策略权重
        when (gamePhase) {
            GamePhase.EARLY_EXPANSION -> {
                expansionWeight = 2.5
                attackWeight = 1.0
                defenseWeight = 1.0
                safetyBuffer = 1.1
            }
            GamePhase.MID_COMPETITION -> {
                expansionWeight = 1.5
                attackWeight = 2.0
                defenseWeight = 1.8
                safetyBuffer = 1.4
            }
            GamePhase.LATE_DOMINATION -> {
                expansionWeight = 1.0
                attackWeight = 2.5
                defenseWeight = 1.5
                safetyBuffer = 1.2
            }
        }
    }

    private fun updatePlanetOwners(gameState: GameState) {
        gameState.planets.forEach { planet ->
            lastPlanetOwners!![planet.id] = planet.owner
        }
    }

    private fun evaluateSituationComplexity(gameState: GameState): Double {
        val totalPlanets = gameState.planets.size
        val activePlanets = gameState.planets.count { it.transporter != null }
        val neutralRatio = gameState.planets.count { it.owner == Player.Neutral }.toDouble() / totalPlanets
        val competitionLevel = 1.0 - neutralRatio

        return (activePlanets.toDouble() / totalPlanets) * 0.4 + competitionLevel * 0.6
    }

    private fun selectActionWithGreedy(
            gameState: GameState,
            myPlanets: List<Planet>,
            neutralPlanets: List<Planet>,
            enemyPlanets: List<Planet>
    ): Action {
        val candidateActions = mutableListOf<Triple<Planet, Planet, Double>>()

        // 处理中立星球攻击
        ticksSinceNeutralAttack++
        val shouldAttackNeutral = neutralPlanets.isNotEmpty() &&
                (ticksSinceNeutralAttack >= neutralAttackInterval || gamePhase == GamePhase.EARLY_EXPANSION)

        if (shouldAttackNeutral) {
            for (source in myPlanets) {
                for (target in neutralPlanets) {
                    val distance = source.position.distance(target.position)
                    val travelTime = distance / params.transporterSpeed
                    val predictedDefense = target.nShips + target.growthRate * travelTime

                    if (source.nShips > predictedDefense * safetyBuffer) {
                        val score = evaluateNeutralTarget(source, target, gameState)
                        candidateActions.add(Triple(source, target, score))
                    }
                }
            }
        }

        // 处理敌方星球攻击
        if (enemyPlanets.isNotEmpty()) {
            for (source in myPlanets) {
                for (target in enemyPlanets) {
                    val distance = source.position.distance(target.position)
                    val travelTime = distance / params.transporterSpeed
                    val predictedDefense = target.nShips + target.growthRate * travelTime

                    if (source.nShips > predictedDefense * safetyBuffer) {
                        val score = evaluateEnemyTarget(source, target, gameState)
                        candidateActions.add(Triple(source, target, score))
                    }
                }
            }
        }

        // 选择最佳候选动作
        val best = candidateActions.maxByOrNull { it.third }
        if (best != null) {
            val (source, target, _) = best
            val distance = source.position.distance(target.position)
            val travelTime = distance / params.transporterSpeed
            val predictedDefense = target.nShips + target.growthRate * travelTime
            val numToSend = min(predictedDefense * safetyBuffer, source.nShips * 0.8)

            if (target.owner == Player.Neutral) {
                ticksSinceNeutralAttack = 0
            }

            return Action(player, source.id, target.id, numToSend)
        }

        // 防御策略：支援脆弱的己方星球
        return reinforceWeakPlanet(gameState, myPlanets)
    }

    private fun selectActionWithEvolution(
            gameState: GameState,
            myPlanets: List<Planet>,
            neutralPlanets: List<Planet>,
            enemyPlanets: List<Planet>
    ): Action {
        val candidateActions = mutableListOf<Action>()

        // 生成候选动作
        for (source in myPlanets) {
            // 中立目标
            for (target in neutralPlanets) {
                val distance = source.position.distance(target.position)
                val travelTime = distance / params.transporterSpeed
                val predictedDefense = target.nShips + target.growthRate * travelTime
                val requiredShips = predictedDefense * safetyBuffer

                if (source.nShips > requiredShips) {
                    val shipsToSend = min(requiredShips, source.nShips * 0.8)
                    candidateActions.add(Action(player, source.id, target.id, shipsToSend))
                }
            }

            // 敌方目标
            for (target in enemyPlanets) {
                val distance = source.position.distance(target.position)
                val travelTime = distance / params.transporterSpeed
                val predictedDefense = target.nShips + target.growthRate * travelTime
                val requiredShips = predictedDefense * safetyBuffer

                if (source.nShips > requiredShips) {
                    val shipsToSend = min(requiredShips, source.nShips * 0.7)
                    candidateActions.add(Action(player, source.id, target.id, shipsToSend))
                }
            }
        }

        if (candidateActions.isEmpty()) {
            return reinforceWeakPlanet(gameState, myPlanets)
        }

        // 使用rollout评估选择最佳动作
        var bestAction = candidateActions.first()
        var bestScore = Double.NEGATIVE_INFINITY

        for (action in candidateActions.take(20)) {
            val score = evaluateActionWithRollout(action, gameState)
            if (score > bestScore) {
                bestScore = score
                bestAction = action
            }
        }

        return bestAction
    }

    private fun evaluateNeutralTarget(source: Planet, target: Planet, gameState: GameState): Double {
        val distance = source.position.distance(target.position)
        val growthPotential = target.growthRate
        val shipCost = target.nShips

        var score = growthPotential * growthBonus * expansionWeight -
                shipCost * 0.5 -
                distance * distancePenalty

        // 早期扩张奖励
        if (gamePhase == GamePhase.EARLY_EXPANSION) {
            score += 15.0
        }

        // 预测对手威胁
        score += predictOpponentThreat(target, gameState)

        return score + Random.nextDouble(0.0, 0.1) // 随机打破平局
    }

    private fun evaluateEnemyTarget(source: Planet, target: Planet, gameState: GameState): Double {
        val distance = source.position.distance(target.position)
        val growthPotential = target.growthRate
        val shipCost = target.nShips

        var score = growthPotential * growthBonus * attackWeight -
                shipCost * 0.8 -
                distance * distancePenalty

        // 优先攻击正在运输的星球（防御薄弱）
        if (target.transporter != null) {
            score += 20.0
        }

        // 高价值目标优先
        score += target.radius * 2.0

        return score + Random.nextDouble(0.0, 0.1)
    }

    private fun predictOpponentThreat(target: Planet, gameState: GameState): Double {
        val opponentPlanets = gameState.planets.filter { it.owner == player.opponent() }
        var threatLevel = 0.0

        for (opponentPlanet in opponentPlanets) {
            val distance = target.position.distance(opponentPlanet.position)
            if (distance < 15.0) {
                threatLevel += (opponentPlanet.nShips / (distance + 1.0)) * 0.3
            }
        }

        return -threatLevel // 威胁越大，目标价值越低
    }

    private fun reinforceWeakPlanet(gameState: GameState, myPlanets: List<Planet>): Action {
        val vulnerablePlanets = myPlanets.filter { it.nShips < 20 }

        if (vulnerablePlanets.isNotEmpty()) {
            val weakest = vulnerablePlanets.minByOrNull { it.nShips }
            val strongest = myPlanets.maxByOrNull { it.nShips }

            if (weakest != null && strongest != null && strongest.nShips > 15) {
                val numToSend = min(strongest.nShips * 0.4, 30.0)
                return Action(player, strongest.id, weakest.id, numToSend)
            }
        }

        return Action.doNothing()
    }

    private fun evaluateActionWithRollout(action: Action, gameState: GameState): Double {
        var totalScore = 0.0

        for (i in 0 until nRollouts) {
            val fm = ForwardModel(gameState.deepCopy(), params)
            val opponentAction = DoNothingAgent().getAction(gameState)

            // 执行当前动作
            fm.step(mapOf(player to action, player.opponent() to opponentAction))

            // 随机rollout
            var steps = 0
            while (steps < rolloutDepth && !fm.isTerminal()) {
                val myAction = generateRandomGreedyAction(fm.state)
                val oppAction = DoNothingAgent().getAction(fm.state)
                fm.step(mapOf(player to myAction, player.opponent() to oppAction))
                steps++
            }

            totalScore += fm.getShips(player) - fm.getShips(player.opponent())
        }

        return totalScore / nRollouts
    }

    private fun generateRandomGreedyAction(state: GameState): Action {
        val sources = state.planets.filter {
            it.owner == player && it.transporter == null && it.nShips > 5
        }
        if (sources.isEmpty()) return Action.doNothing()

        val source = sources.maxByOrNull { it.nShips } ?: return Action.doNothing()
        val neutralTargets = state.planets.filter { it.owner == Player.Neutral }
        val enemyTargets = state.planets.filter { it.owner == player.opponent() }

        val target = if (neutralTargets.isNotEmpty()) {
            neutralTargets.minByOrNull { source.position.distance(it.position) }!!
        } else if (enemyTargets.isNotEmpty()) {
            enemyTargets.minByOrNull { source.position.distance(it.position) }!!
        } else {
            return Action.doNothing()
        }

        val distance = source.position.distance(target.position)
        val travelTime = distance / params.transporterSpeed
        val predictedDefense = target.nShips + target.growthRate * travelTime
        val shipsToSend = min(predictedDefense * 1.2, source.nShips * 0.7)

        return Action(player, source.id, target.id, shipsToSend)
    }
}