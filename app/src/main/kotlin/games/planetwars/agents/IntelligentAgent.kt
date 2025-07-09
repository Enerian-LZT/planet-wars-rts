package games.planetwars.agents

import games.planetwars.agents.evo.GameStateWrapper
import games.planetwars.agents.evo.SimpleEvoAgent
import games.planetwars.core.*
import kotlin.random.Random

data class GameStateWrapper(
        val gameState: GameState,
        val params: GameParams,
        val player: Player,
        val opponentModel: PlanetWarsAgent = DoNothingAgent(),
) {
    var forwardModel = ForwardModel(gameState, params)

    companion object {
        val shiftBy = 2
    }

    fun getAction(gameState: GameState, from: Float, to: Float): Action {
        // filter the planets that are owned by the player AND have a transporter available
        val myPlanets = gameState.planets.filter { it.owner == player && it.transporter == null }
        if (myPlanets.isEmpty()) {
            return Action.doNothing()
        }

        // 优先选择中立行星作为攻击目标（占领中立是重点）
        val neutralTargets = gameState.planets.filter { it.owner == Player.Neutral }
        val enemyTargets = gameState.planets.filter { it.owner == player.opponent() }

        val otherPlanets: List<Planet> = if (neutralTargets.isNotEmpty()) neutralTargets else enemyTargets
        if (otherPlanets.isEmpty()) {
            return Action.doNothing()
        }

        // 以随机向量 from 决定进攻源行星
        val source = myPlanets[(from * myPlanets.size).toInt()]

        // 如果目标为中立，则选择距离 source 最近的一个来加速扩张；
        // 否则（没有中立可攻）按照随机向量 to 选择敌方行星
        val target = if (neutralTargets.isNotEmpty()) {
            otherPlanets.minByOrNull { it.position.distance(source.position) }!!
        } else {
            // 攻击最近的敌方行星
            enemyTargets.minByOrNull { it.position.distance(source.position) }!!
        }

        return Action(player, source.id, target.id, source.nShips / 2)
    }

    fun runForwardModel(seq: FloatArray): Double {
        var ix = 0;
        forwardModel = ForwardModel(gameState.deepCopy(), params)
        while (ix < seq.size && !forwardModel.isTerminal()) {
            val from = seq[ix]
            val to = seq[ix + 1]
            val myAction = getAction(gameState, from, to)
            val opponentAction = opponentModel.getAction(gameState)
            val actions = mapOf(player to myAction, player.opponent() to opponentAction)
            forwardModel.step(actions)
            ix += shiftBy
        }
        return scoreDifference()
    }

    fun scoreDifference(): Double {
        // allow standalone use of this as well
        return forwardModel.getShips(player) - forwardModel.getShips(player.opponent())
    }
}

data class IntelligentAgent(
        var flipAtLeastOneValue: Boolean = true,
        var probMutation: Double = 0.5,
        var sequenceLength: Int = 200,
        var nEvals: Int = 20,
        var useShiftBuffer: Boolean = true,
        var epsilon: Double = 1e-6,
        var timeLimitMillis: Long = 20,
        var opponentModel: PlanetWarsAgent = DoNothingAgent(),

        ) : PlanetWarsPlayer() {
    override fun getAgentType(): String {
        return "IntelligentAgent"
    }

    internal var random = Random

    // these are all the parameters that control the agend
//    internal var buffer: FloatArray? = null // randomPoint(sequenceLength)


    var bestSolution: ScoredSolution? = null

    // 用于记录上一回合各行星的归属，以检测中立行星被敌方刚刚占领的情况
    private var lastPlanetOwners: MutableMap<Int, Player>? = null

    // ===== 新增：行为控制参数 =====
    var enemyPriorityCoef: Double = 1.0      // 半径 / 距离 的权重系数（攻击敌方）
    var neutralPriorityCoef: Double = 1.0    // 半径 / 距离 的权重系数（攻击中立）
    var neutralAttackInterval: Int = 5       // 间隔多少回合去尝试攻击中立
    var rolloutDepth: Int = 60               // RL 评估深度（提高推演深度）
    var nRollouts: Int = 8                   // RL 评估次数（更多样本）
    var distancePower: Double = 1.5          // 距离影响指数，略微降低距离的衰减

    // critic parameters for Actor-Critic
    private var criticWeights: DoubleArray = doubleArrayOf(0.0, 0.0, 0.0)
    var criticAlpha: Double = 0.1            // 学习率
    var criticGamma: Double = 0.9            // 折扣因子

    private var ticksSinceNeutralAttack: Int = 0

    data class ScoredSolution(val score: Double, val solution: FloatArray)

    override fun getAction(gameState: GameState): Action {
        // 初始化行星归属映射
        if (lastPlanetOwners == null) {
            lastPlanetOwners = gameState.planets.associate { it.id to it.owner }.toMutableMap()
        }

        // ---- 统计本回合敌方新占领的行星 ----
        val capturedByEnemy = gameState.planets.filter {
            lastPlanetOwners!![it.id] != player.opponent() && it.owner == player.opponent()
        }
        // 更新归属信息
        gameState.planets.forEach { lastPlanetOwners!![it.id] = it.owner }

        // ===== 新增：根据星球总数动态调整策略参数 =====
        val totalPlanets = gameState.planets.size
        if (totalPlanets < 20) {
            // 增强半径影响，削弱距离影响
            enemyPriorityCoef = 3.0
            neutralPriorityCoef = 2.0
            distancePower = 1.2
        } else {
            // 增强距离影响，并提高评估动作数量
            enemyPriorityCoef = 1.0
            neutralPriorityCoef = 1.0
            distancePower = 1.5
        }
        val candidateLimit = if (totalPlanets < 20) 40 else 80
        // ===== 动态调整结束 =====

        val mySources = gameState.planets.filter { it.owner == player && it.transporter == null && it.nShips > 3 }
        if (mySources.isEmpty()) return Action.DO_NOTHING

        val enemyPlanets = gameState.planets.filter { it.owner == player.opponent() }
        val neutralPlanets = gameState.planets.filter { it.owner == Player.Neutral }

        // 是否考虑进攻中立行星
        var considerNeutral = false
        if (neutralPlanets.isNotEmpty()) {
            ticksSinceNeutralAttack++
            if (ticksSinceNeutralAttack >= neutralAttackInterval) {
                considerNeutral = true
            }
        }

        // Pair of action and heuristic weight，用于快速筛选
        val candidatePairs = mutableListOf<Pair<Action, Double>>()

        // --------- 敌方目标候选 ---------
        if (enemyPlanets.isNotEmpty()) {
            val priorityEnemies = if (capturedByEnemy.isNotEmpty()) capturedByEnemy else enemyPlanets
            // 计算敌方目标的平均半径，用于判断“半径相近”
            val avgEnemyRadius = priorityEnemies.map { it.radius }.average()
            val capturedBoost = 1.5   // 新被占领星球额外半径权重因子
            for (src in mySources) {
                for (tgt in priorityEnemies) {
                    val dist = src.position.distance(tgt.position)
                    if (dist <= 0.0) continue
                    var radiusFactor = tgt.radius
                    // Boost for newly captured planets with similar radius
                    if (capturedByEnemy.contains(tgt) && kotlin.math.abs(tgt.radius - avgEnemyRadius) / avgEnemyRadius < 0.1) {
                        radiusFactor *= capturedBoost
                    }
                    // 进一步放大刚派出飞船行星的影响
                    if (tgt.transporter != null) {
                        radiusFactor *= 1.3  // 发射后防御薄弱，优先级提高
                    }
                    val weight = (radiusFactor / Math.pow(dist, distancePower)) * enemyPriorityCoef
                    val ships = (src.nShips * 0.6).coerceAtLeast(1.0)
                    candidatePairs.add(Action(player, src.id, tgt.id, ships) to weight)
                }
            }
        }

        // --------- 中立目标候选 ---------
        if (considerNeutral && neutralPlanets.isNotEmpty()) {
            for (src in mySources) {
                for (tgt in neutralPlanets) {
                    val dist = src.position.distance(tgt.position)
                    if (dist <= 0.0) continue
                    val weight = (tgt.radius / Math.pow(dist, distancePower)) * neutralPriorityCoef
                    val required = tgt.nShips * 1.1
                    val ships = minOf(src.nShips * 0.2, required).coerceAtLeast(1.0)
                    candidatePairs.add(Action(player, src.id, tgt.id, ships) to weight)
                }
            }
        }

        if (candidatePairs.isEmpty()) return Action.DO_NOTHING

        // 先按启发式权重取前 20 个动作，减少评估开销
        val topCandidates = candidatePairs
                .sortedByDescending { it.second }
                .take(candidateLimit)   // 根据局面规模动态调整候选数目

        // --------- Actor-Critic 评估选择最佳动作 ---------
        var bestScore = Double.NEGATIVE_INFINITY
        var bestAction = topCandidates.first().first
        for ((act, _) in topCandidates) {
            val score = evaluateActionActorCritic(gameState, act)
            if (score > bestScore) {
                bestScore = score
                bestAction = act
            }
        }

        // 如果执行了中立攻击则重置计数器
        val bestTarget = gameState.planets[bestAction.destinationPlanetId]
        if (bestTarget.owner == Player.Neutral) {
            ticksSinceNeutralAttack = 0
        }

        return bestAction
    }

    private fun mutate(v: FloatArray, mutProb: Double): FloatArray {

        val n = v.size
        val x = FloatArray(n)
        // pointwise probability of additional mutations
        // choose element of vector to mutate
        var ix = random.nextInt(n)
        if (!flipAtLeastOneValue) {
            // setting this to -1 means it will never match the first clause in the if statement in the loop
            // leaving it at the randomly chosen value ensures that at least one bit (or more generally value) is always flipped
            ix = -1
        }
        // copy all the values faithfully apart from the chosen one
        for (i in 0 until n) {
            if (i == ix || random.nextDouble() < mutProb) {
                x[i] = random.nextFloat()
            } else {
                x[i] = v[i]
            }
        }
        return x
    }

    // random point in n-dimensional space in unit hypercube; n = sequenceLength
    private fun randomPoint(): FloatArray {
        val p = FloatArray(sequenceLength)
        for (i in p.indices) {
            p[i] = random.nextFloat()
        }
        return p
    }

    private fun shiftLeftAndRandomAppend(v: FloatArray, shiftBy: Int): FloatArray {
        val p = FloatArray(v.size)
        for (i in 0 until p.size - shiftBy) {
            p[i] = v[i + shiftBy]
        }
        // TODO: this is a bit of a hack, but it should work when shiftBy is 2, which it is for now
        p[p.size - 1] = random.nextFloat()
        p[p.size - 2] = random.nextFloat()
        return p
    }

    private fun evalSeq(state: GameState, seq: FloatArray): Double {
        val wrapper = GameStateWrapper(state.deepCopy(), params, player, opponentModel)
        wrapper.runForwardModel(seq)

        // ---------- RL 补充评估：对执行完序列后的状态做若干 rollout ----------
        val rolloutDepth = 40
        val nRollouts = 4
        var extraValue = 0.0
        for (i in 0 until nRollouts) {
            val fm = ForwardModel(wrapper.forwardModel.state.deepCopy(), params)
            var steps = 0
            while (steps < rolloutDepth && !fm.isTerminal()) {
                val myAct = randomGreedyAction(fm.state)
                val oppAct = opponentModel.getAction(fm.state)
                fm.step(mapOf(player to myAct, player.opponent() to oppAct))
                steps++
            }
            extraValue += fm.getShips(player) - fm.getShips(player.opponent())
        }
        extraValue /= nRollouts
        // 综合：原先得分 + RL 预估的未来价值权重 0.5
        return wrapper.scoreDifference() + 0.5 * extraValue
    }

    // rollout 中用于快速行动的策略
    private fun randomGreedyAction(state: GameState): Action {
        val myPlanets = state.planets.filter { it.owner == player && it.transporter == null && it.nShips > 3 }
        if (myPlanets.isEmpty()) return Action.DO_NOTHING
        // 选兵力最多行星
        val source = myPlanets.maxByOrNull { it.nShips } ?: return Action.DO_NOTHING

        val neutralTargets = state.planets.filter { it.owner == Player.Neutral }
        val enemyTargets = state.planets.filter { it.owner == player.opponent() }

        val target = if (neutralTargets.isNotEmpty()) {
            // 选择 growth/distance 比最高中立
            neutralTargets.maxByOrNull { it.growthRate / source.position.distance(it.position) }!!
        } else if (enemyTargets.isNotEmpty()) {
            enemyTargets.minByOrNull {
                val dist = source.position.distance(it.position)
                val futureDef = it.nShips + it.growthRate * (dist / params.transporterSpeed)
                futureDef
            }!!
        } else {
            return Action.DO_NOTHING
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

    // ===== Actor-Critic 评估 =====
    private fun evaluateActionActorCritic(state: GameState, action: Action): Double {
        // 基础特征值
        val featuresBefore = featureVector(state)
        val valueBefore = criticValue(featuresBefore)

        // rollout 一步 + 若干深度
        val fm = ForwardModel(state.deepCopy(), params)
        val oppA0 = opponentModel.getAction(state)
        fm.step(mapOf(player to action, player.opponent() to oppA0))

        var steps = 0
        while (steps < rolloutDepth && !fm.isTerminal()) {
            val myAct = randomGreedyAction(fm.state)
            val oppAct = opponentModel.getAction(fm.state)
            fm.step(mapOf(player to myAct, player.opponent() to oppAct))
            steps++
        }

        val reward = (fm.getShips(player) - fm.getShips(player.opponent())) - (state.planets.filter { it.owner == player }.sumOf { it.nShips } - state.planets.filter { it.owner == player.opponent() }.sumOf { it.nShips })

        val featuresAfter = featureVector(fm.state)
        val valueAfter = criticValue(featuresAfter)

        val target = reward + criticGamma * valueAfter
        val tdError = target - valueBefore

        // critic update
        for (i in criticWeights.indices) {
            criticWeights[i] += criticAlpha * tdError * featuresBefore[i]
        }

        return target // 作为当前动作的评价值
    }

    // --------- Critic helpers ---------
    private fun featureVector(state: GameState): DoubleArray {
        val shipDiff = state.planets.filter { it.owner == player }.sumOf { it.nShips } - state.planets.filter { it.owner == player.opponent() }.sumOf { it.nShips }
        val planetDiff = state.planets.count { it.owner == player } - state.planets.count { it.owner == player.opponent() }
        return doubleArrayOf(shipDiff, planetDiff.toDouble(), 1.0)
    }

    private fun criticValue(features: DoubleArray): Double {
        var v = 0.0
        for (i in criticWeights.indices) {
            v += criticWeights[i] * features[i]
        }
        return v
    }
}

fun main() {
    val gameParams = GameParams(numPlanets = 10)
    val gameState = GameStateFactory(gameParams).createGame()
    val agent = SimpleEvoAgent()
    agent.prepareToPlayAs(Player.Player1, gameParams)
    println(agent.getAgentType())
    val action = agent.getAction(gameState)
    println(action)
}
