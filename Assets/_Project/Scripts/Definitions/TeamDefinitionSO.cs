using UnityEngine;

[CreateAssetMenu(menuName = "Football/Team Definition", fileName = "TeamDef_")]
public class TeamDefinitionSO : ScriptableObject
{
    [Header("Identity")]
    public string teamId;      // "HOME", "AWAY"
    public string teamName;    // "Red Team"
    public string shortCode;   // "RED"

    [Header("Colors")]
    public Color primaryColor = Color.red;      // body
    public Color secondaryColor = Color.white;  // head

    [Header("Optional")]
    public Sprite logo;
}