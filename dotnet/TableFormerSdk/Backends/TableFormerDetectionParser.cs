using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using TableFormerSdk.Models;

namespace TableFormerSdk.Backends;

internal static class TableFormerDetectionParser
{
    // Lowered threshold to match Python pipeline behavior (was 0.5f)
    // Python typically uses 0.25-0.35 for better recall on table cells
    private const float ScoreThreshold = 0.25f;
    // TEMPORANEO: Force debug sempre abilitato
    private const bool _debugLoggingEnabled = true; // Environment.GetEnvironmentVariable("TABLEFORMER_DEBUG") == "1";

    public static IReadOnlyList<TableRegion> Parse(float[] logits, IReadOnlyList<long> logitsShape, float[] boxes, IReadOnlyList<long> boxesShape, int imageWidth, int imageHeight)
    {
        DebugLog($"Parse called - imageSize: {imageWidth}x{imageHeight}, logitsShape: [{string.Join(",", logitsShape)}], boxesShape: [{string.Join(",", boxesShape)}]");

        // Handle both formats:
        // - Full DETR: logits [batch, queries, classes], boxes [batch, queries, 4]
        // - Simplified: logits [batch, classes], boxes [batch, 4]
        if (logitsShape.Count < 2 || boxesShape.Count < 2)
        {
            DebugLog($"Early return - invalid shapes (logitsShape.Count={logitsShape.Count}, boxesShape.Count={boxesShape.Count})");
            return Array.Empty<TableRegion>();
        }

        var batch = (int)logitsShape[0];
        if (batch != 1)
        {
            throw new NotSupportedException("Only batch size 1 is supported");
        }

        int queries, classes;
        if (logitsShape.Count == 2)
        {
            // Simplified format: [batch, classes] - treat as single query
            queries = 1;
            classes = checked((int)logitsShape[1]);
            DebugLog($"Simplified format detected - queries=1, classes={classes}");
        }
        else
        {
            // Full DETR format: [batch, queries, classes]
            queries = checked((int)logitsShape[1]);
            classes = checked((int)logitsShape[2]);
            DebugLog($"Full DETR format detected - queries={queries}, classes={classes}");
        }
        var results = new List<TableRegion>();
        var allScores = new List<float>();
        int filteredCount = 0;

        for (int q = 0; q < queries; q++)
        {
            int offset, boxIndex;
            if (logitsShape.Count == 2)
            {
                // Simplified format: logits are flat [classes], boxes are flat [4]
                offset = 0;
                boxIndex = 0;
            }
            else
            {
                // Full DETR format: logits [queries, classes], boxes [queries, 4]
                offset = q * classes;
                boxIndex = q * 4;
            }

            float bestLogit = float.NegativeInfinity;
            int bestClass = -1;

            for (int c = 0; c < classes; c++)
            {
                float current = logits[offset + c];
                if (current > bestLogit)
                {
                    bestLogit = current;
                    bestClass = c;
                }
            }

            float score = Sigmoid(bestLogit);
            allScores.Add(score);

            if (score < ScoreThreshold)
            {
                filteredCount++;
                continue;
            }

            if (boxIndex + 3 >= boxes.Length)
            {
                break;
            }

            float cx = boxes[boxIndex];
            float cy = boxes[boxIndex + 1];
            float bw = boxes[boxIndex + 2];
            float bh = boxes[boxIndex + 3];

            float x = (cx - (bw / 2f)) * imageWidth;
            float y = (cy - (bh / 2f)) * imageHeight;
            float width = bw * imageWidth;
            float height = bh * imageHeight;

            x = Math.Clamp(x, 0, imageWidth);
            y = Math.Clamp(y, 0, imageHeight);
            width = Math.Clamp(width, 0, imageWidth - x);
            height = Math.Clamp(height, 0, imageHeight - y);

            results.Add(new TableRegion(x, y, width, height, $"class_{bestClass}"));
        }

        // Debug logging
        if (_debugLoggingEnabled && allScores.Count > 0)
        {
            var topScores = allScores.OrderByDescending(s => s).Take(10).ToArray();
            DebugLog($"Queries: {queries}, Filtered: {filteredCount}, Detected: {results.Count}");
            DebugLog($"Top 10 confidence scores: {string.Join(", ", topScores.Select(s => s.ToString("F3")))}");
            DebugLog($"Min/Max/Avg confidence: {allScores.Min():F3}/{allScores.Max():F3}/{allScores.Average():F3}");
        }

        return results;
    }

    private static void DebugLog(string message)
    {
        var logPath = "/tmp/tableformer-debug.log";
        try
        {
            File.AppendAllText(logPath, $"[{DateTime.Now:HH:mm:ss.fff}] [TableFormerParser] {message}\n");
        }
        catch
        {
            // Ignore logging errors
        }
    }

    private static float Sigmoid(float value) => 1f / (1f + MathF.Exp(-value));
}
