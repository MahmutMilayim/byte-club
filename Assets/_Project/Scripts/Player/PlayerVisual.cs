using UnityEngine;

public class PlayerVisual : MonoBehaviour
{
    [SerializeField] private Renderer bodyRenderer;
    [SerializeField] private Renderer headRenderer;

    public PlayerDefinitionSO Definition { get; private set; }

    public void SetRenderers(Renderer body, Renderer head)
    {
        bodyRenderer = body;
        headRenderer = head;
    }

    public void ApplyDefinition(PlayerDefinitionSO def)
    {
        Definition = def;
        if (def == null || def.team == null) return;

        if (bodyRenderer != null)
        {
            var m = new Material(bodyRenderer.sharedMaterial);
            m.color = def.team.primaryColor;     // BODY = primary
            bodyRenderer.material = m;
        }

        if (headRenderer != null)
        {
            var m = new Material(headRenderer.sharedMaterial);
            m.color = def.team.secondaryColor;  // HEAD = secondary
            headRenderer.material = m;
        }
    }
}