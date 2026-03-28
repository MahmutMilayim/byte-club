using UnityEngine;

[DisallowMultipleComponent]
public class PlayerAnimationDriver : MonoBehaviour
{
    private static readonly int SpeedHash = Animator.StringToHash("Speed");
    private static readonly int PassHash = Animator.StringToHash("Pass");
    private static readonly int ShootHash = Animator.StringToHash("Shoot");

    [Header("Refs")]
    [SerializeField] private Animator animator;

    [Header("Tuning")]
    [SerializeField] private float defaultJogSpeed = 1f;
    [SerializeField] private bool resetToIdleOnEnable = true;
    [SerializeField] private bool faceActionDirection = true;
    [SerializeField] private float rotateYOnly = 1f;

    public Animator Animator => animator;

    private void Reset()
    {
        animator = GetComponent<Animator>();
    }

    private void Awake()
    {
        if (animator == null)
            animator = GetComponent<Animator>();

        if (animator == null)
            Debug.LogError($"PlayerAnimationDriver on '{name}' could not find Animator.");
    }

    private void OnEnable()
    {
        if (resetToIdleOnEnable)
            SetIdle();
    }

    public void SetIdle()
    {
        if (animator == null) return;
        animator.SetFloat(SpeedHash, 0f);
    }

    public void SetJog(float speed = -1f)
    {
        if (animator == null) return;

        if (speed < 0f)
            speed = defaultJogSpeed;

        animator.SetFloat(SpeedHash, speed);
    }

    public void PlayPass()
    {
        if (animator == null) return;

        animator.SetFloat(SpeedHash, 0f);
        animator.ResetTrigger(ShootHash);
        animator.SetTrigger(PassHash);
    }

    public void PlayShoot()
    {
        if (animator == null) return;

        animator.SetFloat(SpeedHash, 0f);
        animator.ResetTrigger(PassHash);
        animator.SetTrigger(ShootHash);
    }

    public void FaceTowards(Vector3 worldTarget)
    {
        if (!faceActionDirection) return;

        Vector3 dir = worldTarget - transform.position;
        dir.y = 0f;

        if (dir.sqrMagnitude < 0.0001f)
            return;

        Quaternion look = Quaternion.LookRotation(dir.normalized, Vector3.up);

        if (rotateYOnly > 0.5f)
            transform.rotation = Quaternion.Euler(0f, look.eulerAngles.y, 0f);
        else
            transform.rotation = look;
    }

    public void FaceDirection(Vector3 worldDirection)
    {
        if (!faceActionDirection) return;

        worldDirection.y = 0f;
        if (worldDirection.sqrMagnitude < 0.0001f)
            return;

        Quaternion look = Quaternion.LookRotation(worldDirection.normalized, Vector3.up);

        if (rotateYOnly > 0.5f)
            transform.rotation = Quaternion.Euler(0f, look.eulerAngles.y, 0f);
        else
            transform.rotation = look;
    }

    public void TeleportAndIdle(Vector3 worldPosition)
    {
        transform.position = worldPosition;
        SetIdle();
    }
}