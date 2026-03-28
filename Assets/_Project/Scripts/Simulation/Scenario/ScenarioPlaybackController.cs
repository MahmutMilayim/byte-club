using System.Collections;
using UnityEngine;

public class ScenarioPlaybackController : MonoBehaviour
{
    [Header("Refs")]
    public FrameSnapshotApplier applier;
    public MetersToWorldMapper mapper;
    public ShotGeometryConfig config;
    public BestPassEvaluator bestPassEvaluator;

    [Header("Timing")]
    public float preActionDelay = 0.35f;
    public float passToShootDelay = 0.9f;
    public float resetDelay = 1.0f;

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
        if (dto == null || dto.shooter == null || mapper == null || config == null || applier == null || bestPassEvaluator == null)
        {
            Debug.LogWarning("ScenarioPlaybackController: Missing dto/applier/mapper/config/bestPassEvaluator.");
            yield break;
        }

        ResetAllPlayersToIdle();

        if (!applier.TryGetPlayerAnimationDriver(dto.shooter.playerId, out var shooterDriver))
        {
            Debug.LogWarning($"ScenarioPlaybackController: Shooter driver not found for {dto.shooter.playerId}.");
            yield break;
        }

        // Kararı evaluator versin
        DecisionResult decision = bestPassEvaluator.LastResult;
        if (decision == null)
        {
            Debug.LogWarning("ScenarioPlaybackController: decision result is null. Make sure BestPassEvaluator ran before playback.");
            yield break;
        }

        yield return new WaitForSeconds(preActionDelay);

        Vector3 goalWorld = GetGoalWorld(dto.targetGoal);

        // Direkt şut
        if (!decision.usePass)
        {
            if (logDecision)
            {
                Debug.Log(
                    $"[ScenarioPlayback] SHOOT selected | shooter={decision.initialShooterId} shot={decision.initialShooterOpenRatio * 100f:F1}%"
                );
            }

            shooterDriver.FaceTowards(goalWorld);
            shooterDriver.PlayShoot();

            yield return new WaitForSeconds(resetDelay);
            shooterDriver.SetIdle();
            yield break;
        }

        // Pass + shoot
        if (!applier.TryGetPlayerAnimationDriver(decision.passReceiverId, out var receiverDriver))
        {
            Debug.LogWarning(
                $"ScenarioPlaybackController: Receiver driver not found for {decision.passReceiverId}. Falling back to direct shot."
            );

            shooterDriver.FaceTowards(goalWorld);
            shooterDriver.PlayShoot();

            yield return new WaitForSeconds(resetDelay);
            shooterDriver.SetIdle();
            yield break;
        }

        if (logDecision)
        {
            Debug.Log(
                $"[ScenarioPlayback] PASS selected | shooter={decision.initialShooterId} shot={decision.initialShooterOpenRatio * 100f:F1}% " +
                $"-> receiver={decision.passReceiverId} shot={decision.finalShooterOpenRatio * 100f:F1}%"
            );
        }

        Vector3 receiverWorld = mapper.ToWorld(decision.passReceiverMeters.x, decision.passReceiverMeters.y);

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
}