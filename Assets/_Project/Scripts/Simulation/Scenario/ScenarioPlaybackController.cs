using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ScenarioPlaybackController : MonoBehaviour
{
    [Header("Refs")]
    public FrameSnapshotApplier applier;
    public MetersToWorldMapper mapper;
    public ShotGeometryConfig config;

    [Header("Timing")]
    public float preActionDelay = 0.35f;
    public float passToShootDelay = 0.9f;
    public float resetDelay = 1.0f;

    [Header("Decision")]
    public float passMarginPercent = 5f;
    public bool onlyOpponentsBlockPass = true;
    public float passBlockRadius = 0.85f;
    public float gkPassBlockRadius = 1.2f;

    [Header("Debug")]
    public bool playOnSnapshotApplied = true;
    public bool logDecision = true;

    private Coroutine _playRoutine;

    private void OnEnable()
    {
        if (applier != null && playOnSnapshotApplied)
            applier.OnSnapshotApplied += HandleSnapshotApplied;
    }

    private void OnDisable()
    {
        if (applier != null && playOnSnapshotApplied)
            applier.OnSnapshotApplied -= HandleSnapshotApplied;
    }

    private void HandleSnapshotApplied(FrameSnapshotDTO dto)
    {
        if (_playRoutine != null)
            StopCoroutine(_playRoutine);

        _playRoutine = StartCoroutine(PlayScenario(dto));
    }

    [ContextMenu("Play Scenario Now")]
    public void PlayScenarioNow()
    {
        var dto = applier != null ? applier.LastDTO : null;
        if (dto == null)
        {
            Debug.LogWarning("ScenarioPlaybackController: No snapshot available.");
            return;
        }

        if (_playRoutine != null)
            StopCoroutine(_playRoutine);

        _playRoutine = StartCoroutine(PlayScenario(dto));
    }

    private IEnumerator PlayScenario(FrameSnapshotDTO dto)
    {
        if (dto == null || dto.shooter == null || mapper == null || config == null || applier == null)
        {
            Debug.LogWarning("ScenarioPlaybackController: Missing dto/applier/mapper/config.");
            yield break;
        }

        ResetAllPlayersToIdle();

        if (!applier.TryGetPlayerAnimationDriver(dto.shooter.playerId, out var shooterDriver))
        {
            Debug.LogWarning($"ScenarioPlaybackController: Shooter driver not found for {dto.shooter.playerId}.");
            yield break;
        }

        string shooterTeam = dto.shooter.teamId;
        Vector2 shooterPos = new Vector2(dto.shooter.x, dto.shooter.y);

        float shooterOpen = EvaluateShotAt(dto, shooterPos, shooterTeam);

        int bestReceiverId = -1;
        float bestReceiverOpen = -1f;
        Vector2 bestReceiverPos = default;

        if (dto.players != null)
        {
            foreach (var p in dto.players)
            {
                if (p == null) continue;
                if (p.teamId != shooterTeam) continue;

                Vector2 teammatePos = new Vector2(p.x, p.y);

                if (!IsPassClear(dto, shooterPos, teammatePos, shooterTeam))
                    continue;

                float teammateOpen = EvaluateShotAt(dto, teammatePos, shooterTeam);

                if (teammateOpen > bestReceiverOpen)
                {
                    bestReceiverOpen = teammateOpen;
                    bestReceiverId = p.id;
                    bestReceiverPos = teammatePos;
                }
            }
        }

        float shooterPct = shooterOpen * 100f;
        float bestPassPct = bestReceiverOpen * 100f;

        bool usePass = bestReceiverId != -1 && bestPassPct >= shooterPct + passMarginPercent;

        if (logDecision)
        {
            if (usePass)
            {
                Debug.Log($"[ScenarioPlayback] PASS selected | shooter={dto.shooter.playerId} shot={shooterPct:F1}% -> receiver={bestReceiverId} shot={bestPassPct:F1}%");
            }
            else
            {
                Debug.Log($"[ScenarioPlayback] SHOOT selected | shooter={dto.shooter.playerId} shot={shooterPct:F1}%");
            }
        }

        yield return new WaitForSeconds(preActionDelay);

        Vector3 goalWorld = GetGoalWorld(dto.targetGoal);

        if (!usePass)
        {
            shooterDriver.FaceTowards(goalWorld);
            shooterDriver.PlayShoot();

            yield return new WaitForSeconds(resetDelay);
            shooterDriver.SetIdle();
            yield break;
        }

        if (!applier.TryGetPlayerAnimationDriver(bestReceiverId, out var receiverDriver))
        {
            Debug.LogWarning($"ScenarioPlaybackController: Receiver driver not found for {bestReceiverId}. Falling back to direct shot.");
            shooterDriver.FaceTowards(goalWorld);
            shooterDriver.PlayShoot();

            yield return new WaitForSeconds(resetDelay);
            shooterDriver.SetIdle();
            yield break;
        }

        Vector3 receiverWorld = mapper.ToWorld(bestReceiverPos.x, bestReceiverPos.y);

        shooterDriver.FaceTowards(receiverWorld);
        shooterDriver.PlayPass();

        yield return new WaitForSeconds(passToShootDelay);

        receiverDriver.FaceTowards(goalWorld);
        receiverDriver.PlayShoot();

        yield return new WaitForSeconds(resetDelay);

        shooterDriver.SetIdle();
        receiverDriver.SetIdle();
    }

    private void ResetAllPlayersToIdle()
    {
        if (applier == null) return;

        // FrameSnapshotApplier internal dictionary public değil, bu yüzden DTO üzerinden resetleyelim.
        var dto = applier.LastDTO;
        if (dto == null) return;

        if (dto.shooter != null && applier.TryGetPlayerAnimationDriver(dto.shooter.playerId, out var shooterDriver))
            shooterDriver.SetIdle();

        if (dto.goalkeeper != null && applier.TryGetPlayerAnimationDriver(dto.goalkeeper.playerId, out var gkDriver))
            gkDriver.SetIdle();

        if (dto.players != null)
        {
            foreach (var p in dto.players)
            {
                if (p == null) continue;
                if (applier.TryGetPlayerAnimationDriver(p.id, out var driver))
                    driver.SetIdle();
            }
        }
    }

    private Vector3 GetGoalWorld(string targetGoal)
    {
        string tg = (targetGoal ?? "TOP").Trim().ToUpperInvariant();
        float goalX = config.goalCenterX;
        float goalY = (tg == "BOTTOM") ? 0f : config.fieldLength;
        return mapper.ToWorld(goalX, goalY);
    }

    private float EvaluateShotAt(FrameSnapshotDTO dto, Vector2 shooterPos, string shooterTeamId)
    {
        string targetGoal = (dto.targetGoal ?? "TOP").Trim().ToUpperInvariant();
        float goalY = (targetGoal == "BOTTOM") ? 0f : config.fieldLength;

        Vector2 leftPost = new Vector2(config.goalCenterX - config.goalHalfWidth, goalY);
        Vector2 rightPost = new Vector2(config.goalCenterX + config.goalHalfWidth, goalY);

        var blockers = BuildShotBlockers(dto, shooterTeamId);
        var result = ShotOpennessEvaluator.Evaluate(shooterPos, leftPost, rightPost, blockers);

        return result.openRatio;
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