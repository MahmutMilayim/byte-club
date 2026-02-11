using UnityEngine;

public enum PlayerRole { GK, DEF, MID, FWD }

[CreateAssetMenu(menuName = "Football/Player Definition", fileName = "PlayerDef_")]
public class PlayerDefinitionSO : ScriptableObject
{
    [Header("Identity")]
    public string playerId;       // dış snapshot id
    public string playerName;
    [Range(0, 99)] public int jerseyNumber;

    [Header("Team / Role")]
    public TeamDefinitionSO team;
    public PlayerRole role;

    [Header("Static params")]
    [Min(0f)] public float interceptRadiusMeters = 1.25f;

    [Header("Prefab")]
    public GameObject visualPrefab; // otomatik atanacak
}