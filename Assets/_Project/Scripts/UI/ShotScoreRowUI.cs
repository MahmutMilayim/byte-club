using TMPro;
using UnityEngine;

public class ShotScoreRowUI : MonoBehaviour
{
    [Header("Refs")]
    public TMP_Text playerIdText;
    public TMP_Text roleText;
    public TMP_Text scoreText;

    public void Bind(int playerId, string roleLabel, float scorePercent)
    {
        if (playerIdText != null)
            playerIdText.text = $"Player {playerId}";

        if (roleText != null)
            roleText.text = roleLabel;

        if (scoreText != null)
            scoreText.text = $"{scorePercent:F1}%";
    }

    public void BindEmpty()
    {
        if (playerIdText != null)
            playerIdText.text = "-";

        if (roleText != null)
            roleText.text = "-";

        if (scoreText != null)
            scoreText.text = "-";
    }
}