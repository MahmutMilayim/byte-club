using UnityEngine;

public class PlayerVisual : MonoBehaviour
{
    [SerializeField] private Renderer bodyRenderer;
    [SerializeField] private Renderer headRenderer;

    private MaterialPropertyBlock _mpbBody;
    private MaterialPropertyBlock _mpbHead;

    public PlayerDefinitionSO Definition { get; private set; }

    public void SetRenderers(Renderer body, Renderer head)
    {
        bodyRenderer = body;
        headRenderer = head;

        _mpbBody ??= new MaterialPropertyBlock();
        _mpbHead ??= new MaterialPropertyBlock();
    }

    public void ApplyDefinition(PlayerDefinitionSO def)
    {
        Definition = def;
        if (def == null || def.team == null) return;

        if (bodyRenderer != null)
        {
            bodyRenderer.GetPropertyBlock(_mpbBody);
            _mpbBody.SetColor("_BaseColor", def.team.primaryColor);  // URP Lit
            _mpbBody.SetColor("_Color", def.team.primaryColor);      // fallback
            bodyRenderer.SetPropertyBlock(_mpbBody);
        }

        if (headRenderer != null)
        {
            headRenderer.GetPropertyBlock(_mpbHead);
            _mpbHead.SetColor("_BaseColor", def.team.secondaryColor);
            _mpbHead.SetColor("_Color", def.team.secondaryColor);
            headRenderer.SetPropertyBlock(_mpbHead);
        }
    }
}