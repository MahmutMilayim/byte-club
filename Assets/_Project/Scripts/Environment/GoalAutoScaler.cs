using UnityEngine;

[ExecuteAlways]
public class GoalAutoScaler : MonoBehaviour
{
    [Header("Real goal dimensions (meters)")]
    public float targetWidth = 7.32f;

    [Tooltip("Which local axis represents the goal width? Usually X.")]
    public WidthAxis widthAxis = WidthAxis.X;

    public enum WidthAxis
    {
        X,
        Y,
        Z
    }

    [ContextMenu("Scale Goal To Real Width")]
    public void ScaleGoalToRealWidth()
    {
        Renderer[] renderers = GetComponentsInChildren<Renderer>();
        if (renderers == null || renderers.Length == 0)
        {
            Debug.LogWarning("GoalAutoScaler: No Renderer found.");
            return;
        }

        Bounds combined = renderers[0].bounds;
        for (int i = 1; i < renderers.Length; i++)
            combined.Encapsulate(renderers[i].bounds);

        float currentWidth = widthAxis switch
        {
            WidthAxis.X => combined.size.x,
            WidthAxis.Y => combined.size.y,
            WidthAxis.Z => combined.size.z,
            _ => combined.size.x
        };

        if (currentWidth <= 0.0001f)
        {
            Debug.LogWarning("GoalAutoScaler: Current width is too small.");
            return;
        }

        float factor = targetWidth / currentWidth;
        transform.localScale *= factor;

        Debug.Log($"GoalAutoScaler: currentWidth={currentWidth:F3}, factor={factor:F3}, newScale={transform.localScale}");
    }
}