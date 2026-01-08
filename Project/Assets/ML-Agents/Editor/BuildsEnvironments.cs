using UnityEditor;
using UnityEngine;
using System.IO;

public class BuildEnvironments
{
    [MenuItem("ML-Agents/Build All Environments")]
    static void BuildAllEnvironments()
    {
        string[] environments = new string[]
        {
            "3DBall",
            "Basic", 
            "Crawler",
            "FoodCollector",
            "GridWorld",
            "Hallway",
            "PushBlock",
            "Pyramids",
            "Walker",
            "WallJump",
            "Worm"
        };

        int successCount = 0;
        int failCount = 0;

        foreach (string env in environments)
        {
            if (BuildEnvironment(env))
                successCount++;
            else
                failCount++;
        }

        Debug.Log($"========================================");
        Debug.Log($"BUILD COMPLETE!");
        Debug.Log($"Success: {successCount}, Failed: {failCount}");
        Debug.Log($"========================================");
    }

    static bool BuildEnvironment(string envName)
    {
        string scenePath = $"Assets/ML-Agents/Examples/{envName}/Scenes/{envName}.unity";
        
        if (!File.Exists(scenePath))
        {
            Debug.LogWarning($"❌ Scene not found: {scenePath}");
            return false;
        }

        string buildPath = Path.Combine(
            Directory.GetParent(Application.dataPath).Parent.FullName,
            "builds",
            envName,
            $"{envName}.exe"
        );

        Directory.CreateDirectory(Path.GetDirectoryName(buildPath));

        BuildPlayerOptions buildPlayerOptions = new BuildPlayerOptions
        {
            scenes = new[] { scenePath },
            locationPathName = buildPath,
            target = BuildTarget.StandaloneWindows64,
            options = BuildOptions.EnableHeadlessMode  // No graphics!
        };

        Debug.Log($"Building {envName}...");
        
        var report = BuildPipeline.BuildPlayer(buildPlayerOptions);
        
        if (report.summary.result == UnityEditor.Build.Reporting.BuildResult.Succeeded)
        {
            Debug.Log($"Built {envName} → {buildPath}");
            return true;
        }
        else
        {
            Debug.LogError($"Failed to build {envName}");
            return false;
        }
    }
}
