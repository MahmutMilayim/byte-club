using System.Collections.Generic;
using UnityEngine;

public class BestPassEvaluator : MonoBehaviour
{
    [Header("Refs")]
    public FrameSnapshotApplier applier;
    public ShotGeometryConfig config;
    public MetersToWorldMapper mapper;

    [Header("Pass rules")]
    public bool onlyOpponentsBlockPass = true;
    public float passBlockRadius = 0.85f;
    public float gkPassBlockRadius = 1.2f;

    [Header("Shot probability sampling")]
    [Range(10, 120)] public int goalSamples = 48;

    [Header("Log")]
    public bool logResult = true;

    public DecisionResult LastResult { get; private set; }

    private struct GoalBlocker
    {
        public Vector2 pos;
        public float radius;
    }

    private void OnEnable()
    {
        if (applier != null)
            applier.OnSnapshotApplied += _ => EvaluateNow();
    }

    private void OnDisable()
    {
        if (applier != null)
            applier.OnSnapshotApplied -= _ => EvaluateNow();
    }

    [ContextMenu("Evaluate Best Pass Now")]
    public DecisionResult EvaluateNow()
    {
        var dto = applier != null ? applier.LastDTO : null;
        return Evaluate(dto);
    }

    public DecisionResult Evaluate(FrameSnapshotDTO dto)
    {
        if (dto == null || dto.shooter == null || config == null || mapper == null)
        {
            Debug.LogWarning("BestPassEvaluator: missing dto/shooter/config/mapper.");
            LastResult = null;
            return null;
        }

        string shooterTeam = dto.shooter.teamId;
        int shooterId = dto.shooter.playerId;
        Vector2 shooterPos = new Vector2(dto.shooter.x, dto.shooter.y);

        float shooterOpen = ComputeShotProbabilityForPosition(dto, shooterPos, shooterId);

        var result = new DecisionResult
        {
            frameIndex = dto.frameIndex,
            initialShooterId = shooterId,
            initialShooterOpenRatio = shooterOpen,
            finalShooterId = shooterId,
            finalShooterOpenRatio = shooterOpen,
            usePass = false,
            passReceiverId = -1,
            passReceiverMeters = default
        };

        int bestReceiverId = -1;
        float bestReceiverOpen = -1f;
        Vector2 bestReceiverPos = default;

        if (dto.players != null)
        {
            foreach (var p in dto.players)
            {
                if (p == null) continue;
                if (p.teamId != shooterTeam) continue;
                if (p.id == shooterId) continue;

                Vector2 teammatePos = new Vector2(p.x, p.y);
                bool passClear = IsPassClear(dto, shooterPos, teammatePos, shooterTeam);

                var option = new PassOptionResult
                {
                    receiverId = p.id,
                    passClear = passClear,
                    receiverMeters = teammatePos,
                    shotOpenRatio = -1f
                };

                if (passClear)
                {
                    float teammateOpen = ComputeShotProbabilityForPosition(dto, teammatePos, p.id);
                    option.shotOpenRatio = teammateOpen;

                    if (teammateOpen > bestReceiverOpen)
                    {
                        bestReceiverOpen = teammateOpen;
                        bestReceiverId = p.id;
                        bestReceiverPos = teammatePos;
                    }
                }

                result.passOptions.Add(option);

                if (logResult)
                {
                    if (!option.passClear)
                    {
                        Debug.Log($"[PASS-OPTION] receiver={option.receiverId} passClear=FALSE shotOpen=N/A");
                    }
                    else
                    {
                        Debug.Log($"[PASS-OPTION] receiver={option.receiverId} passClear=TRUE shotOpen={option.shotOpenRatio * 100f:F1}%");
                    }
                }
            }
        }

        // Yeni karar kuralı:
        // Eğer en iyi temiz pas opsiyonu shooter'dan daha iyiyse PASS
        if (bestReceiverId != -1 && bestReceiverOpen > shooterOpen)
        {
            result.usePass = true;
            result.passReceiverId = bestReceiverId;
            result.passReceiverMeters = bestReceiverPos;
            result.finalShooterId = bestReceiverId;
            result.finalShooterOpenRatio = bestReceiverOpen;
        }

        LastResult = result;

        if (logResult)
        {
            float shooterPct = shooterOpen * 100f;

            if (!result.usePass)
            {
                if (bestReceiverId == -1)
                {
                    Debug.Log($"[PASS] frame={dto.frameIndex} shooterOpen={shooterPct:F1}% bestPass=NONE decision=SHOT");
                }
                else
                {
                    float bestPct = bestReceiverOpen * 100f;
                    Debug.Log(
                        $"[PASS] frame={dto.frameIndex} shooterOpen={shooterPct:F1}% " +
                        $"bestPassTo={bestReceiverId} bestPassOpen={bestPct:F1}% decision=SHOT"
                    );
                }
            }
            else
            {
                float bestPct = bestReceiverOpen * 100f;
                Debug.Log(
                    $"[PASS] frame={dto.frameIndex} shooterOpen={shooterPct:F1}% " +
                    $"bestPassTo={bestReceiverId} bestPassOpen={bestPct:F1}% decision=PASS"
                );
            }
        }

        return result;
    }

    public float ComputeShotProbabilityForPosition(FrameSnapshotDTO dto, Vector2 shooterPos, int shooterId)
    {
        if (dto == null || config == null || mapper == null)
            return 0f;

        string targetGoal = (dto.targetGoal ?? "TOP").Trim().ToUpperInvariant();
        float goalY = (targetGoal == "BOTTOM") ? 0f : config.fieldLength;

        Vector2 goalLeft = new Vector2(config.goalCenterX - config.goalHalfWidth, goalY);
        Vector2 goalRight = new Vector2(config.goalCenterX + config.goalHalfWidth, goalY);

        List<GoalBlocker> blockers = BuildShotBlockers(dto, shooterId);

        int openCount = 0;

        for (int i = 0; i < goalSamples; i++)
        {
            float t = (i + 0.5f) / goalSamples;
            Vector2 target = Vector2.Lerp(goalLeft, goalRight, t);

            if (!IsBlocked(shooterPos, target, blockers))
                openCount++;
        }

        int blockedCount = goalSamples - openCount;
        int totalCount = openCount + blockedCount;

        if (totalCount <= 0)
            return 0f;

        float probability = (float)openCount / totalCount;

        if (logResult)
        {
            Debug.Log(
                $"[SHOT-PROB] shooterId={shooterId} open={openCount} blocked={blockedCount} total={totalCount} prob={probability:P1}"
            );
        }

        return probability;
    }

    private List<GoalBlocker> BuildShotBlockers(FrameSnapshotDTO dto, int shooterId)
    {
        var blockers = new List<GoalBlocker>(32);

        // Goalkeeper blocker
        if (dto.goalkeeper != null && dto.goalkeeper.playerId != shooterId)
        {
            blockers.Add(new GoalBlocker
            {
                pos = new Vector2(dto.goalkeeper.x, dto.goalkeeper.y),
                radius = config.gkBlockRadius
            });
        }

        // Shooter hariç herkes blocker
        if (dto.players != null)
        {
            foreach (var p in dto.players)
            {
                if (p == null) continue;
                if (p.id == shooterId) continue;

                blockers.Add(new GoalBlocker
                {
                    pos = new Vector2(p.x, p.y),
                    radius = config.playerBlockRadius
                });
            }
        }

        return blockers;
    }

    private bool IsBlocked(Vector2 shooterPos, Vector2 targetPos, List<GoalBlocker> blockers)
    {
        for (int i = 0; i < blockers.Count; i++)
        {
            if (ShotOpennessEvaluator_SegmentCircle.SegmentIntersectsCircle(
                    shooterPos,
                    targetPos,
                    blockers[i].pos,
                    blockers[i].radius))
            {
                return true;
            }
        }

        return false;
    }

    private bool IsPassClear(FrameSnapshotDTO dto, Vector2 from, Vector2 to, string shooterTeamId)
    {
        if (dto.players != null)
        {
            foreach (var p in dto.players)
            {
                if (p == null) continue;

                // Pası sadece rakipler bloklasın istiyorsan
                if (onlyOpponentsBlockPass && p.teamId == shooterTeamId)
                    continue;

                // receiver kendisi blocker sayılmasın
                if ((new Vector2(p.x, p.y) - to).sqrMagnitude < 0.0001f)
                    continue;

                if (ShotOpennessEvaluator_SegmentCircle.SegmentIntersectsCircle(
                        from,
                        to,
                        new Vector2(p.x, p.y),
                        passBlockRadius))
                {
                    return false;
                }
            }
        }

        if (dto.goalkeeper != null)
        {
            if (!onlyOpponentsBlockPass || dto.goalkeeper.teamId != shooterTeamId)
            {
                if (ShotOpennessEvaluator_SegmentCircle.SegmentIntersectsCircle(
                        from,
                        to,
                        new Vector2(dto.goalkeeper.x, dto.goalkeeper.y),
                        gkPassBlockRadius))
                {
                    return false;
                }
            }
        }

        return true;
    }

    public bool TryGetRandomOpenGoalTarget(FrameSnapshotDTO dto, Vector2 shooterPos, int shooterId, out Vector2 targetMeters)
{
    targetMeters = default;

    if (dto == null || config == null || mapper == null)
        return false;

    string targetGoal = (dto.targetGoal ?? "TOP").Trim().ToUpperInvariant();
    float goalY = (targetGoal == "BOTTOM") ? 0f : config.fieldLength;

    Vector2 goalLeft = new Vector2(config.goalCenterX - config.goalHalfWidth, goalY);
    Vector2 goalRight = new Vector2(config.goalCenterX + config.goalHalfWidth, goalY);

    List<GoalBlocker> blockers = BuildShotBlockers(dto, shooterId);
    List<Vector2> openTargets = new List<Vector2>();

    for (int i = 0; i < goalSamples; i++)
    {
        float t = (i + 0.5f) / goalSamples;
        Vector2 sampleTarget = Vector2.Lerp(goalLeft, goalRight, t);

        if (!IsBlocked(shooterPos, sampleTarget, blockers))
            openTargets.Add(sampleTarget);
    }

    if (openTargets.Count == 0)
        return false;

    int randomIndex = Random.Range(0, openTargets.Count);
    targetMeters = openTargets[randomIndex];
    return true;
}
}