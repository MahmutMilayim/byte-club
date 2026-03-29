using System.Collections.Generic;
using UnityEngine;

public class ShotScorePanelController : MonoBehaviour
{
    [Header("Refs")]
    public BestPassEvaluator bestPassEvaluator;
    public Transform contentRoot;
    public GameObject rowPrefab;

    [Header("Options")]
    public int maxRows = 5;
    public bool refreshOnEnable = true;

    private readonly List<GameObject> _spawnedRows = new();

    private void OnEnable()
    {
        if (refreshOnEnable)
            Refresh();
    }

    [ContextMenu("Refresh Score Panel")]
    public void Refresh()
    {
        ClearRows();

        if (bestPassEvaluator == null || contentRoot == null || rowPrefab == null)
        {
            Debug.LogWarning("ShotScorePanelController: Missing refs.");
            return;
        }

        List<ShotScoreEntry> entries = bestPassEvaluator.GetTopScoreEntries(maxRows);

        for (int i = 0; i < maxRows; i++)
        {
            GameObject row = Instantiate(rowPrefab, contentRoot);
            _spawnedRows.Add(row);

            ShotScoreRowUI rowUI = row.GetComponent<ShotScoreRowUI>();
            if (rowUI == null)
                continue;

            if (i < entries.Count)
            {
                ShotScoreEntry entry = entries[i];
                string roleLabel = entry.isInitialShooter ? "Shooter" : "Receiver";
                rowUI.Bind(entry.playerId, roleLabel, entry.scorePercent);
            }
            else
            {
                rowUI.BindEmpty();
            }
        }
    }

    private void ClearRows()
    {
        for (int i = 0; i < _spawnedRows.Count; i++)
        {
            if (_spawnedRows[i] != null)
                Destroy(_spawnedRows[i]);
        }

        _spawnedRows.Clear();
    }
}