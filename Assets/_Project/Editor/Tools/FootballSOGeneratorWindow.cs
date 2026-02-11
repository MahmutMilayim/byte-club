#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

public class FootballSOGeneratorWindow : EditorWindow
{
    [Header("Output (base folders)")]
    [SerializeField] private string teamsFolder = "Assets/_Project/ScriptableObjects/Teams";
    [SerializeField] private string playersFolder = "Assets/_Project/ScriptableObjects/Players";
    [SerializeField] private string prefabsFolder = "Assets/_Project/Prefabs/Player";

    [Header("Manual Generate Teams")]
    [SerializeField] private string homeTeamId = "HOME";
    [SerializeField] private string homeTeamName = "Home Team";
    [SerializeField] private string homeShortCode = "HOME";

    [SerializeField] private string awayTeamId = "AWAY";
    [SerializeField] private string awayTeamName = "Away Team";
    [SerializeField] private string awayShortCode = "AWAY";

    [Header("Manual Generate Players (counts)")]
    [SerializeField] private int homeCount = 11;
    [SerializeField] private int awayCount = 11;

    [Header("Manual IDs")]
    [SerializeField] private bool autoGenerateIds = true;
    [SerializeField] private string idPrefixHome = "H";
    [SerializeField] private string idPrefixAway = "A";
    [SerializeField] private int idStartNumber = 1;
    [SerializeField] private int idDigits = 4;
    [SerializeField] private TextAsset customIdListText; // one id per line

    [Header("Manual Player Defaults")]
    [SerializeField] private float defaultInterceptRadiusMeters = 1.25f;
    [SerializeField] private string homeNamePrefix = "Home Player ";
    [SerializeField] private string awayNamePrefix = "Away Player ";
    [SerializeField] private int homeJerseyStart = 1;
    [SerializeField] private int awayJerseyStart = 1;

    [Header("Prefab (snowman)")]
    [SerializeField] private bool generateSnowmanPrefabIfMissing = true;
    [SerializeField] private string snowmanPrefabName = "SnowmanPlayer.prefab";

    [Header("Auto Spawn (Scene)")]
    [SerializeField] private bool autoSpawnInScene = false;
    [SerializeField] private string runtimeRootName = "FootballRuntime";
    [SerializeField] private float spawnSpacing = 2.2f;
    [SerializeField] private Vector3 spawnOrigin = new Vector3(-12f, 0f, -6f);

    [Header("JSON Import")]
    [SerializeField] private TextAsset rosterJson;
    [SerializeField] private bool importAlsoGeneratesSnowmanPrefab = true;

    private int TotalPlayers => Mathf.Max(0, homeCount) + Mathf.Max(0, awayCount);

    [MenuItem("Tools/Football/Generate Team & Player SOs")]
    public static void Open()
    {
        var w = GetWindow<FootballSOGeneratorWindow>("Football Generator");
        w.minSize = new Vector2(620, 760);
    }

    private void OnGUI()
    {
        EditorGUILayout.LabelField("Football Generator", EditorStyles.boldLabel);
        EditorGUILayout.Space(8);

        // Output
        EditorGUILayout.LabelField("Output", EditorStyles.boldLabel);
        teamsFolder = EditorGUILayout.TextField("Teams Base Folder", teamsFolder);
        playersFolder = EditorGUILayout.TextField("Players Base Folder", playersFolder);
        prefabsFolder = EditorGUILayout.TextField("Prefabs Folder", prefabsFolder);

        EditorGUILayout.Space(10);

        // Manual teams
        EditorGUILayout.LabelField("Manual Generate (Teams)", EditorStyles.boldLabel);
        DrawTeamBlock("HOME", ref homeTeamId, ref homeTeamName, ref homeShortCode);
        EditorGUILayout.Space(6);
        DrawTeamBlock("AWAY", ref awayTeamId, ref awayTeamName, ref awayShortCode);

        // Manual counts
        EditorGUILayout.Space(10);
        EditorGUILayout.LabelField("Manual Generate (Players)", EditorStyles.boldLabel);
        homeCount = EditorGUILayout.IntField("Home Count", homeCount);
        awayCount = EditorGUILayout.IntField("Away Count", awayCount);
        EditorGUILayout.LabelField("Total Players", TotalPlayers.ToString());

        // Spawn toggle
        EditorGUILayout.Space(8);
        autoSpawnInScene = EditorGUILayout.Toggle("Spawn After Generate/Import", autoSpawnInScene);

        // IDs
        EditorGUILayout.Space(8);
        autoGenerateIds = EditorGUILayout.Toggle("Auto-generate IDs", autoGenerateIds);
        if (autoGenerateIds)
        {
            idPrefixHome = EditorGUILayout.TextField("Home Prefix", idPrefixHome);
            idPrefixAway = EditorGUILayout.TextField("Away Prefix", idPrefixAway);
            idStartNumber = EditorGUILayout.IntField("Start Number", idStartNumber);
            idDigits = EditorGUILayout.IntSlider("Digits", idDigits, 1, 8);
        }
        else
        {
            customIdListText = (TextAsset)EditorGUILayout.ObjectField("Custom ID List", customIdListText, typeof(TextAsset), false);
            EditorGUILayout.HelpBox($"Text dosyasında {TotalPlayers} satır ID olmalı.", MessageType.Info);
        }

        // Defaults
        EditorGUILayout.Space(8);
        defaultInterceptRadiusMeters = EditorGUILayout.FloatField("Intercept Radius (m)", defaultInterceptRadiusMeters);
        homeNamePrefix = EditorGUILayout.TextField("Home Name Prefix", homeNamePrefix);
        awayNamePrefix = EditorGUILayout.TextField("Away Name Prefix", awayNamePrefix);
        homeJerseyStart = EditorGUILayout.IntField("Home Jersey Start", homeJerseyStart);
        awayJerseyStart = EditorGUILayout.IntField("Away Jersey Start", awayJerseyStart);

        // Prefab option
        EditorGUILayout.Space(8);
        generateSnowmanPrefabIfMissing = EditorGUILayout.Toggle("Auto-create Snowman Prefab", generateSnowmanPrefabIfMissing);
        snowmanPrefabName = EditorGUILayout.TextField("Snowman Prefab Name", snowmanPrefabName);

        EditorGUILayout.Space(12);

        using (new EditorGUI.DisabledScope(!IsConfigValid(out var reason)))
        {
            if (GUILayout.Button("MANUAL GENERATE (Teams + Players + Prefab)", GUILayout.Height(42)))
                GenerateAllManual();
        }
        if (!IsConfigValid(out var warn))
            EditorGUILayout.HelpBox(warn, MessageType.Warning);

        // JSON Import
        EditorGUILayout.Space(14);
        EditorGUILayout.LabelField("JSON Import", EditorStyles.boldLabel);
        rosterJson = (TextAsset)EditorGUILayout.ObjectField("Roster JSON", rosterJson, typeof(TextAsset), false);
        importAlsoGeneratesSnowmanPrefab = EditorGUILayout.Toggle("Import also ensures Snowman Prefab", importAlsoGeneratesSnowmanPrefab);

        using (new EditorGUI.DisabledScope(rosterJson == null))
        {
            if (GUILayout.Button("IMPORT JSON (Create/Update SOs)", GUILayout.Height(36)))
                ImportFromJson();
        }
    }

    private void DrawTeamBlock(string label, ref string id, ref string name, ref string code)
    {
        EditorGUILayout.LabelField(label, EditorStyles.miniBoldLabel);
        id = EditorGUILayout.TextField("Team Id", id);
        name = EditorGUILayout.TextField("Team Name", name);
        code = EditorGUILayout.TextField("Short Code", code);
    }

    private bool IsConfigValid(out string reason)
    {
        if (!IsAssetsPath(teamsFolder) || !IsAssetsPath(playersFolder) || !IsAssetsPath(prefabsFolder))
        {
            reason = "Folders must be inside Assets/. Örn: Assets/_Project/...";
            return false;
        }

        if (homeCount < 0 || awayCount < 0 || TotalPlayers <= 0)
        {
            reason = "Counts must be non-negative and total > 0.";
            return false;
        }

        if (!autoGenerateIds)
        {
            if (customIdListText == null)
            {
                reason = "Auto IDs kapalıysa Custom ID List vermelisin.";
                return false;
            }

            var ids = ParseIds(customIdListText.text);
            if (ids.Count != TotalPlayers)
            {
                reason = $"Custom ID List {TotalPlayers} satır olmalı. Found {ids.Count}.";
                return false;
            }
        }

        if (defaultInterceptRadiusMeters < 0f)
        {
            reason = "Intercept radius must be >= 0.";
            return false;
        }

        reason = "";
        return true;
    }

    // -----------------------------
    // MANUAL GENERATE
    // -----------------------------
    private void GenerateAllManual()
    {
        EnsureFolder(teamsFolder);
        EnsureFolder(playersFolder);
        EnsureFolder(prefabsFolder);

        AssetDatabase.Refresh();

        var ok = EditorUtility.DisplayDialog(
            "Manual Generate",
            $"Create/update:\n- 2 Team SO\n- {TotalPlayers} Player SO\n- Snowman prefab (optional)\n\nContinue?",
            "Yes",
            "Cancel"
        );
        if (!ok) return;

        try
        {
            // Teams -> write into Teams/<TeamName>/
            var homeTeam = CreateOrUpdateTeamInTeamFolder(homeTeamId, homeTeamName, homeShortCode);
            var awayTeam = CreateOrUpdateTeamInTeamFolder(awayTeamId, awayTeamName, awayShortCode);

            // Prefab
            GameObject snowmanPrefab = null;
            if (generateSnowmanPrefabIfMissing)
            {
                var prefabPath = $"{prefabsFolder}/{SanitizeFileName(snowmanPrefabName)}";
                snowmanPrefab = CreateOrUpdateSnowmanPrefab(prefabPath);
            }

            // IDs
            var ids = autoGenerateIds ? GenerateIds() : ParseIds(customIdListText.text);

            // Players -> write into Players/<TeamName>/
            int idx = 0;
            idx = CreatePlayersForTeamInTeamFolder(homeTeam, homeCount, ids, idx, homeNamePrefix, homeJerseyStart, snowmanPrefab);
            idx = CreatePlayersForTeamInTeamFolder(awayTeam, awayCount, ids, idx, awayNamePrefix, awayJerseyStart, snowmanPrefab);

            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();

            if (autoSpawnInScene)
                SpawnPlayersInActiveScene();
        }
        catch (Exception ex)
        {
            Debug.LogError(ex);
            EditorUtility.DisplayDialog("Generation Failed", ex.Message, "OK");
            return;
        }

        EditorUtility.DisplayDialog("Done", "Manual generate completed.", "OK");
    }

    private TeamDefinitionSO CreateOrUpdateTeamInTeamFolder(string teamId, string teamName, string shortCode)
    {
        string folderName = SanitizeFolderName(!string.IsNullOrWhiteSpace(teamName) ? teamName : teamId);
        string teamOutFolder = $"{teamsFolder}/{folderName}";
        EnsureFolder(teamOutFolder);

        string path = $"{teamOutFolder}/TeamDef_{SanitizeFileName(teamId)}.asset";
        var existing = AssetDatabase.LoadAssetAtPath<TeamDefinitionSO>(path);
        TeamDefinitionSO so = existing != null ? existing : ScriptableObject.CreateInstance<TeamDefinitionSO>();

        so.teamId = teamId;
        so.teamName = teamName;
        so.shortCode = shortCode;

        if (existing == null) AssetDatabase.CreateAsset(so, path);
        else EditorUtility.SetDirty(so);

        return so;
    }

    private int CreatePlayersForTeamInTeamFolder(
        TeamDefinitionSO team,
        int count,
        List<string> ids,
        int startIndex,
        string namePrefix,
        int jerseyStart,
        GameObject prefabToAssign
    )
    {
        string teamFolderName = SanitizeFolderName(!string.IsNullOrWhiteSpace(team.teamName) ? team.teamName : team.teamId);
        string playerOutFolder = $"{playersFolder}/{teamFolderName}";
        EnsureFolder(playerOutFolder);

        for (int i = 0; i < count; i++)
        {
            string id = ids[startIndex + i];
            string safeId = SanitizeFileName(id);
            string path = $"{playerOutFolder}/PlayerDef_{safeId}.asset";

            var existing = AssetDatabase.LoadAssetAtPath<PlayerDefinitionSO>(path);
            PlayerDefinitionSO so = existing != null ? existing : ScriptableObject.CreateInstance<PlayerDefinitionSO>();

            so.playerId = id;
            so.playerName = $"{namePrefix}{i + 1}";
            so.jerseyNumber = Mathf.Clamp(jerseyStart + i, 0, 99);
            so.team = team;
            so.role = (i == 0) ? PlayerRole.GK : PlayerRole.FWD; // manual'da basit
            so.interceptRadiusMeters = Mathf.Max(0f, defaultInterceptRadiusMeters);

            if (prefabToAssign != null)
                so.visualPrefab = prefabToAssign;

            if (existing == null) AssetDatabase.CreateAsset(so, path);
            else EditorUtility.SetDirty(so);
        }

        return startIndex + count;
    }

    // -----------------------------
    // PREFAB (Snowman)
    // -----------------------------
    private GameObject CreateOrUpdateSnowmanPrefab(string prefabPath)
    {
        var existingPrefab = AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath);
        if (existingPrefab != null) return existingPrefab;

        var root = new GameObject("SnowmanPlayer");

        var body = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        body.name = "Body";
        body.transform.SetParent(root.transform, false);
        body.transform.localScale = new Vector3(1.0f, 1.0f, 1.0f);
        body.transform.localPosition = new Vector3(0f, 0.5f, 0f);

        var head = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        head.name = "Head";
        head.transform.SetParent(root.transform, false);
        head.transform.localScale = new Vector3(0.6f, 0.6f, 0.6f);
        head.transform.localPosition = new Vector3(0f, 1.3f, 0f);

        // Remove primitive colliders
        DestroyImmediate(body.GetComponent<Collider>());
        DestroyImmediate(head.GetComponent<Collider>());

        // One capsule collider on root
        var capsule = root.AddComponent<CapsuleCollider>();
        capsule.center = new Vector3(0f, 1.0f, 0f);
        capsule.height = 2.0f;
        capsule.radius = 0.4f;

        // PlayerVisual
        var pv = root.AddComponent<PlayerVisual>();
        pv.SetRenderers(body.GetComponent<Renderer>(), head.GetComponent<Renderer>());

        // Save prefab
        var savedPrefab = PrefabUtility.SaveAsPrefabAsset(root, prefabPath);
        DestroyImmediate(root);
        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();
        return savedPrefab;
    }

    // -----------------------------
    // SPAWN
    // -----------------------------
    private void SpawnPlayersInActiveScene()
    {
        GameObject root = GameObject.Find(runtimeRootName);
        if (root == null)
        {
            root = new GameObject(runtimeRootName);
            Undo.RegisterCreatedObjectUndo(root, "Create FootballRuntime Root");
        }

        // Clear existing children (idempotent)
        for (int i = root.transform.childCount - 1; i >= 0; i--)
        {
            var child = root.transform.GetChild(i).gameObject;
            Undo.DestroyObjectImmediate(child);
        }

        // Load all PlayerDefinitionSO under playersFolder
        string[] guids = AssetDatabase.FindAssets("t:PlayerDefinitionSO", new[] { playersFolder });
        var defs = new List<PlayerDefinitionSO>(guids.Length);
        foreach (var guid in guids)
        {
            string path = AssetDatabase.GUIDToAssetPath(guid);
            var def = AssetDatabase.LoadAssetAtPath<PlayerDefinitionSO>(path);
            if (def != null) defs.Add(def);
        }

        if (defs.Count == 0)
        {
            EditorUtility.DisplayDialog("Spawn", "No PlayerDefinitionSO found to spawn.", "OK");
            return;
        }

        // Sort by team name then jersey
        defs.Sort((a, b) =>
        {
            string ta = a.team != null ? a.team.teamName : "";
            string tb = b.team != null ? b.team.teamName : "";
            int t = string.CompareOrdinal(ta, tb);
            if (t != 0) return t;
            return a.jerseyNumber.CompareTo(b.jerseyNumber);
        });

        var prefab = defs[0].visualPrefab;
        if (prefab == null)
        {
            EditorUtility.DisplayDialog("Spawn", "visualPrefab is null on PlayerDefinitionSO. Generate/import first.", "OK");
            return;
        }

        int col = 0;
        int row = 0;
        string lastTeam = defs[0].team != null ? defs[0].team.teamName : "";

        for (int i = 0; i < defs.Count; i++)
        {
            var def = defs[i];
            string currentTeam = def.team != null ? def.team.teamName : "";

            if (i > 0 && currentTeam != lastTeam)
            {
                row++;
                col = 0;
                lastTeam = currentTeam;
            }

            Vector3 pos = spawnOrigin + new Vector3(col * spawnSpacing, 0f, row * spawnSpacing);

            GameObject instance = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
            Undo.RegisterCreatedObjectUndo(instance, "Spawn Player");
            instance.name = $"P_{def.playerId}_{def.playerName}";
            instance.transform.SetParent(root.transform, true);
            instance.transform.position = pos;

            var pv = instance.GetComponent<PlayerVisual>();
            if (pv != null)
                pv.ApplyDefinition(def);

            col++;
        }

        Selection.activeObject = root;
    }

    // -----------------------------
    // JSON IMPORT
    // -----------------------------
    [Serializable] private class RosterRoot { public TeamJson[] teams; public PlayerJson[] players; }
    [Serializable] private class TeamJson { public string teamId; public string teamName; public string shortCode; }
    [Serializable] private class PlayerJson
    {
        public string playerId;
        public string playerName;
        public int jerseyNumber;
        public string role;                 // "GK"/"DEF"/"MID"/"FWD"
        public float interceptRadiusMeters; // optional
        public string teamId;
    }

    private void ImportFromJson()
    {
        if (rosterJson == null)
        {
            EditorUtility.DisplayDialog("Import", "Please assign a roster JSON TextAsset.", "OK");
            return;
        }

        EnsureFolder(teamsFolder);
        EnsureFolder(playersFolder);
        EnsureFolder(prefabsFolder);

        GameObject snowmanPrefab = null;
        if (importAlsoGeneratesSnowmanPrefab)
        {
            var prefabPath = $"{prefabsFolder}/{SanitizeFileName(snowmanPrefabName)}";
            snowmanPrefab = CreateOrUpdateSnowmanPrefab(prefabPath);
        }

        RosterRoot root;
        try
        {
            root = JsonUtility.FromJson<RosterRoot>(rosterJson.text);
        }
        catch
        {
            EditorUtility.DisplayDialog("Import Failed", "JSON parse failed. Check format.", "OK");
            return;
        }

        if (root == null || root.teams == null || root.players == null)
        {
            EditorUtility.DisplayDialog("Import Failed", "JSON must contain 'teams' and 'players' arrays.", "OK");
            return;
        }

        var teamMap = new Dictionary<string, TeamDefinitionSO>();

        try
        {
            // Teams
            foreach (var t in root.teams)
            {
                if (string.IsNullOrWhiteSpace(t.teamId)) continue;
                var teamSo = CreateOrUpdateTeamInTeamFolder(t.teamId, t.teamName, t.shortCode);
                teamMap[t.teamId] = teamSo;
            }

            // Players -> Players/<TeamName>/
            foreach (var p in root.players)
            {
                if (string.IsNullOrWhiteSpace(p.playerId) || string.IsNullOrWhiteSpace(p.teamId))
                    continue;

                if (!teamMap.TryGetValue(p.teamId, out var teamSo) || teamSo == null)
                {
                    Debug.LogWarning($"Player {p.playerId} references unknown teamId '{p.teamId}'. Skipped.");
                    continue;
                }

                var role = ParseRole(p.role);
                var radius = (p.interceptRadiusMeters > 0f) ? p.interceptRadiusMeters : defaultInterceptRadiusMeters;

                CreateOrUpdatePlayerInTeamFolder(
                    playerId: p.playerId,
                    playerName: p.playerName,
                    jerseyNumber: p.jerseyNumber,
                    role: role,
                    interceptRadius: radius,
                    team: teamSo,
                    prefabToAssign: snowmanPrefab
                );
            }

            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
        }
        catch (Exception ex)
        {
            Debug.LogError(ex);
            EditorUtility.DisplayDialog("Import Failed", ex.Message, "OK");
            return;
        }

        EditorUtility.DisplayDialog("Import", "JSON imported successfully (Teams & Players updated).", "OK");

        if (autoSpawnInScene)
            SpawnPlayersInActiveScene();
    }

    private void CreateOrUpdatePlayerInTeamFolder(
        string playerId,
        string playerName,
        int jerseyNumber,
        PlayerRole role,
        float interceptRadius,
        TeamDefinitionSO team,
        GameObject prefabToAssign
    )
    {
        string folderName = SanitizeFolderName(!string.IsNullOrWhiteSpace(team.teamName) ? team.teamName : team.teamId);
        string playerOutFolder = $"{playersFolder}/{folderName}";
        EnsureFolder(playerOutFolder);

        string safeId = SanitizeFileName(playerId);
        string path = $"{playerOutFolder}/PlayerDef_{safeId}.asset";

        var existing = AssetDatabase.LoadAssetAtPath<PlayerDefinitionSO>(path);
        PlayerDefinitionSO so = existing != null ? existing : ScriptableObject.CreateInstance<PlayerDefinitionSO>();

        so.playerId = playerId;
        so.playerName = string.IsNullOrWhiteSpace(playerName) ? playerId : playerName;
        so.jerseyNumber = Mathf.Clamp(jerseyNumber, 0, 99);
        so.team = team;
        so.role = role;
        so.interceptRadiusMeters = Mathf.Max(0f, interceptRadius);

        if (prefabToAssign != null)
            so.visualPrefab = prefabToAssign;

        if (existing == null) AssetDatabase.CreateAsset(so, path);
        else EditorUtility.SetDirty(so);
    }

    private PlayerRole ParseRole(string role)
    {
        if (string.IsNullOrWhiteSpace(role)) return PlayerRole.FWD;
        role = role.Trim().ToUpperInvariant();

        switch (role)
        {
            case "GK": return PlayerRole.GK;
            case "DEF": return PlayerRole.DEF;
            case "MID": return PlayerRole.MID;
            case "FWD": return PlayerRole.FWD;
            default: return PlayerRole.FWD;
        }
    }

    // -----------------------------
    // HELPERS
    // -----------------------------
    private List<string> GenerateIds()
    {
        var ids = new List<string>(TotalPlayers);

        for (int i = 0; i < homeCount; i++)
            ids.Add($"{idPrefixHome}{(idStartNumber + i).ToString().PadLeft(idDigits, '0')}");

        for (int i = 0; i < awayCount; i++)
            ids.Add($"{idPrefixAway}{(idStartNumber + i).ToString().PadLeft(idDigits, '0')}");

        return ids;
    }

    private static List<string> ParseIds(string text)
    {
        var ids = new List<string>();
        if (string.IsNullOrWhiteSpace(text)) return ids;

        var lines = text.Split(new[] { "\r\n", "\n" }, StringSplitOptions.RemoveEmptyEntries);
        foreach (var raw in lines)
        {
            var s = raw.Trim();
            if (!string.IsNullOrEmpty(s)) ids.Add(s);
        }
        return ids;
    }

    private static bool IsAssetsPath(string path)
    {
        if (string.IsNullOrWhiteSpace(path)) return false;
        path = path.Replace("\\", "/");
        return path == "Assets" || path.StartsWith("Assets/");
    }

    private static void EnsureFolder(string unityPath)
    {
        unityPath = unityPath.Replace("\\", "/").TrimEnd('/');

        if (string.IsNullOrWhiteSpace(unityPath)) return;
        if (unityPath == "Assets") return;

        if (!unityPath.StartsWith("Assets/"))
            throw new Exception($"Path must start with 'Assets/': {unityPath}");

        if (AssetDatabase.IsValidFolder(unityPath)) return;

        var parts = unityPath.Split('/');
        string current = "Assets";

        for (int i = 1; i < parts.Length; i++)
        {
            string next = $"{current}/{parts[i]}";
            if (!AssetDatabase.IsValidFolder(next))
                AssetDatabase.CreateFolder(current, parts[i]);
            current = next;
        }
    }

    private static string SanitizeFolderName(string name)
    {
        if (string.IsNullOrWhiteSpace(name)) return "Team";
        name = name.Trim();

        foreach (var c in Path.GetInvalidFileNameChars())
            name = name.Replace(c.ToString(), "");

        name = name.Replace(" ", "_");
        return string.IsNullOrWhiteSpace(name) ? "Team" : name;
    }

    private static string SanitizeFileName(string name)
    {
        if (string.IsNullOrWhiteSpace(name)) return "id";
        name = name.Trim();

        foreach (var c in Path.GetInvalidFileNameChars())
            name = name.Replace(c.ToString(), "");

        return string.IsNullOrWhiteSpace(name) ? "id" : name;
    }
}
#endif