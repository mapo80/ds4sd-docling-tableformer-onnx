using System;
using System.Collections.Generic;
using TableFormerSdk.Models;

namespace TableFormerSdk.Backends;

internal static class TableFormerDetectionParser
{
    private const float ScoreThreshold = 0.5f;

    public static IReadOnlyList<TableRegion> Parse(float[] logits, IReadOnlyList<long> logitsShape, float[] boxes, IReadOnlyList<long> boxesShape, int imageWidth, int imageHeight)
    {
        if (logitsShape.Count < 3 || boxesShape.Count < 3)
        {
            return Array.Empty<TableRegion>();
        }

        var batch = (int)logitsShape[0];
        if (batch != 1)
        {
            throw new NotSupportedException("Only batch size 1 is supported");
        }

        int queries = checked((int)logitsShape[1]);
        int classes = checked((int)logitsShape[2]);
        var results = new List<TableRegion>();

        for (int q = 0; q < queries; q++)
        {
            int offset = q * classes;
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
            if (score < ScoreThreshold)
            {
                continue;
            }

            int boxIndex = q * 4;
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

        return results;
    }

    private static float Sigmoid(float value) => 1f / (1f + MathF.Exp(-value));
}
