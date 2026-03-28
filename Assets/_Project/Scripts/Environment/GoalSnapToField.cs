using UnityEngine;

public class GoalSnapToField : MonoBehaviour
{
    public MetersToWorldMapper mapper;
    public bool topGoal = true;
    public float groundY = 0f;

    [Tooltip("Modelin kendi yönü yanlışsa buradan düzelt. Genelde 90, -90 veya 180 gerekir.")]
    public float yawOffset = 0f;

    [ContextMenu("Snap Goal To Field")]
    public void SnapGoalToField()
    {
        if (mapper == null)
        {
            Debug.LogWarning("GoalSnapToField: mapper is missing.");
            return;
        }

        float meterX = 34f;
        float meterY = topGoal ? 105f : 0f;

        Vector3 world = mapper.ToWorld(meterX, meterY);
        transform.position = new Vector3(world.x, groundY, world.z);

        Vector3 fieldCenter = mapper.ToWorld(34f, 52.5f);
        Vector3 dir = fieldCenter - transform.position;
        dir.y = 0f;

        if (dir.sqrMagnitude > 0.0001f)
        {
            transform.rotation =
                Quaternion.LookRotation(dir.normalized, Vector3.up) *
                Quaternion.Euler(0f, yawOffset, 0f);
        }

        Debug.Log($"{name} snapped -> topGoal={topGoal}, world={transform.position}, yawOffset={yawOffset}");
    }
}