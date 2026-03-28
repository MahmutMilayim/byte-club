using System.Collections;
using UnityEngine;

public class BallPlaybackController : MonoBehaviour
{
    [Header("Refs")]
    public FrameSnapshotApplier applier;
    public MetersToWorldMapper mapper;

    [Header("Timing")]
    public float passDuration = 0.55f;
    public float shotDuration = 0.65f;

    [Header("Arc")]
    public float passArcHeight = 0.35f;
    public float shotArcHeight = 0.15f;

    [Header("Offsets")]
    public float ballHeightOffset = 0.12f;
    public float forwardOffset = 0.35f;

    private Coroutine _ballRoutine;

    public void StopPlayback()
    {
        if (_ballRoutine != null)
        {
            StopCoroutine(_ballRoutine);
            _ballRoutine = null;
        }
    }

    public void SnapBallInFrontOfPlayer(Transform playerTransform)
    {
        GameObject ball = GetBallObject();
        if (ball == null || playerTransform == null)
            return;

        Vector3 forward = playerTransform.forward;
        forward.y = 0f;
        if (forward.sqrMagnitude < 0.0001f)
            forward = Vector3.forward;

        forward.Normalize();

        Vector3 pos = playerTransform.position + forward * forwardOffset;
        pos.y += ballHeightOffset;

        ball.transform.position = pos;
        ball.SetActive(true);
    }

    public void PlayPass(Vector3 fromWorld, Vector3 toWorld)
    {
        StopPlayback();
        _ballRoutine = StartCoroutine(AnimateBall(fromWorld, toWorld, passDuration, passArcHeight));
    }

    public void PlayShot(Vector3 fromWorld, Vector3 toWorld)
    {
        StopPlayback();
        _ballRoutine = StartCoroutine(AnimateBall(fromWorld, toWorld, shotDuration, shotArcHeight));
    }

    private IEnumerator AnimateBall(Vector3 fromWorld, Vector3 toWorld, float duration, float arcHeight)
    {
        GameObject ball = GetBallObject();
        if (ball == null)
            yield break;

        fromWorld.y += ballHeightOffset;
        toWorld.y += ballHeightOffset;

        float elapsed = 0f;

        while (elapsed < duration)
        {
            float t = elapsed / duration;

            Vector3 pos = Vector3.Lerp(fromWorld, toWorld, t);
            pos.y += Mathf.Sin(t * Mathf.PI) * arcHeight;

            ball.transform.position = pos;

            elapsed += Time.deltaTime;
            yield return null;
        }

        ball.transform.position = toWorld;
        _ballRoutine = null;
    }

    private GameObject GetBallObject()
    {
        if (applier == null)
        {
            Debug.LogWarning("BallPlaybackController: applier missing.");
            return null;
        }

        GameObject ball = applier.GetBallObject();
        if (ball == null)
        {
            Debug.LogWarning("BallPlaybackController: no runtime ball found.");
            return null;
        }

        return ball;
    }
}