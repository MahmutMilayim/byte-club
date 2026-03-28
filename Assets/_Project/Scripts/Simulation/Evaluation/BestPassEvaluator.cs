using System.Collections.Generic;
using UnityEngine;

public class BestPassEvaluator : MonoBehaviour
{
    [Header("Refs")]
    public FrameSnapshotApplier applier;
    public ShotGeometryConfig config;

    [Header("Pass rules")]
    public bool onlyOpponentsBlockPass = true;
    public float passBlockRadius = 0.85f;
    public float gkPassBlockRadius = 1.2f;

    [Header("Log")]
    public bool logResult = true;

    public DecisionResult LastResult { get; private set; }

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
        if (dto == null || dto.shooter == null || config == null)
        {
            Debug.LogWarning("BestPassEvaluator: missing dto/shooter/config.");
            LastResult = null;
            return null;
        }

        string shooterTeam = dto.shooter.teamId;
        Vector2 shooterPos = new Vector2(dto.shooter.x, dto.shooter.y);

        float shooterOpen = EvaluateShotAt(dto, shooterPos, shooterTeam);

        var result = new DecisionResult
        {
            frameIndex = dto.frameIndex,
            initialShooterId = dto.shooter.playerId,
            initialShooterOpenRatio = shooterOpen,
            finalShooterId = dto.shooter.playerId,
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
                if (p.id == dto.shooter.playerId) continue;

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
                    float teammateOpen = EvaluateShotAt(dto, teammatePos, shooterTeam);
                    option.shotOpenRatio = teammateOpen;

                    if (teammateOpen > bestReceiverOpen)
                    {
                        bestReceiverOpen = teammateOpen;
                        bestReceiverId = p.id;
                        bestReceiverPos = teammatePos;
                    }
                }

                result.passOptions.Add(option);
            }
        }

        // Yeni kural:
        // Eğer herhangi bir temiz pas opsiyonunun shotOpenRatio'su shooter'dan büyükse, pass seç.
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
                    Debug.Log(
                        $"[PASS] frame={dto.frameIndex} shooterOpen={shooterPct:F1}% bestPass=NONE decision=SHOT"
                    );
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

    private float EvaluateShotAt(FrameSnapshotDTO dto, Vector2 shooterPos, string shooterTeamId)
    {
        string targetGoal = (dto.targetGoal ?? "TOP").Trim().ToUpperInvariant();
        float goalY = (targetGoal == "BOTTOM") ? 0f : config.fieldLength;

        Vector2 leftPost = new Vector2(config.goalCenterX - config.goalHalfWidth, goalY);
        Vector2 rightPost = new Vector2(config.goalCenterX + config.goalHalfWidth, goalY);

        var blockers = BuildShotBlockers(dto, shooterTeamId);
        var res = ShotOpennessEvaluator.Evaluate(shooterPos, leftPost, rightPost, blockers);

        return res.openRatio;
    }

    private List<ShotOpennessEvaluator.Blocker> BuildShotBlockers(FrameSnapshotDTO dto, string shooterTeamId)
    {
        var blockers = new List<ShotOpennessEvaluator.Blocker>(32);

        if (dto.goalkeeper != null && dto.goalkeeper.teamId != shooterTeamId)
        {
            blockers.Add(new ShotOpennessEvaluator.Blocker
            {
                pos = new Vector2(dto.goalkeeper.x, dto.goalkeeper.y),
                radius = config.gkBlockRadius
            });
        }

        if (dto.players != null)
        {
            foreach (var p in dto.players)
            {
                if (p == null) continue;
                if (p.teamId == shooterTeamId) continue;

                blockers.Add(new ShotOpennessEvaluator.Blocker
                {
                    pos = new Vector2(p.x, p.y),
                    radius = config.playerBlockRadius
                });
            }
        }

        return blockers;
    }

    private bool IsPassClear(FrameSnapshotDTO dto, Vector2 from, Vector2 to, string shooterTeamId)
    {
        if (dto.players != null)
        {
            foreach (var p in dto.players)
            {
                if (p == null) continue;

                if (onlyOpponentsBlockPass && p.teamId == shooterTeamId)
                    continue;

                // receiver kendisi blocker sayılmasın
                if ((new Vector2(p.x, p.y) - to).sqrMagnitude < 0.0001f)
                    continue;

                if (ShotOpennessEvaluator_SegmentCircle.SegmentIntersectsCircle(from, to, new Vector2(p.x, p.y), passBlockRadius))
                    return false;
            }
        }

        if (dto.goalkeeper != null)
        {
            if (!onlyOpponentsBlockPass || dto.goalkeeper.teamId != shooterTeamId)
            {
                if (ShotOpennessEvaluator_SegmentCircle.SegmentIntersectsCircle(from, to, new Vector2(dto.goalkeeper.x, dto.goalkeeper.y), gkPassBlockRadius))
                    return false;
            }
        }

        return true;
    }
}