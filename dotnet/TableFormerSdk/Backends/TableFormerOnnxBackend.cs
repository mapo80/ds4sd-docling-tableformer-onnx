using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using TableFormerSdk.Models;

namespace TableFormerSdk.Backends;

internal sealed class TableFormerOnnxBackend : ITableFormerBackend, IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;

    public TableFormerOnnxBackend(string modelPath)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
        {
            throw new ArgumentException("Model path is empty", nameof(modelPath));
        }

        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            IntraOpNumThreads = 0,
            InterOpNumThreads = 1
        };

        _session = new InferenceSession(modelPath, options);
        options.Dispose();
        _inputName = _session.InputMetadata.Keys.First();
    }

    public IReadOnlyList<TableRegion> Infer(SKBitmap image, string sourcePath)
    {
        var tensor = Preprocess(image);
        var input = NamedOnnxValue.CreateFromTensor(_inputName, tensor);
        using var results = _session.Run(new[] { input });
        (input as IDisposable)?.Dispose();

        var logitsTensor = results.First(x => string.Equals(x.Name, "logits", StringComparison.OrdinalIgnoreCase)).AsTensor<float>();
        var boxesTensor = results.First(x => string.Equals(x.Name, "pred_boxes", StringComparison.OrdinalIgnoreCase)).AsTensor<float>();

        var logits = logitsTensor.ToArray();
        var boxes = boxesTensor.ToArray();
        var logitsShape = ToLongArray(logitsTensor.Dimensions);
        var boxesShape = ToLongArray(boxesTensor.Dimensions);

        return TableFormerDetectionParser.Parse(logits, logitsShape, boxes, boxesShape, image.Width, image.Height);
    }

    private static DenseTensor<float> Preprocess(SKBitmap bitmap)
    {
        int width = bitmap.Width;
        int height = bitmap.Height;
        var data = new DenseTensor<float>(new[] { 1, 3, height, width });
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                var color = bitmap.GetPixel(x, y);
                data[0, 0, y, x] = color.Red / 255f;
                data[0, 1, y, x] = color.Green / 255f;
                data[0, 2, y, x] = color.Blue / 255f;
            }
        }

        return data;
    }

    public void Dispose() => _session.Dispose();

    private static long[] ToLongArray(ReadOnlySpan<int> dimensions)
    {
        var result = new long[dimensions.Length];
        for (int i = 0; i < dimensions.Length; i++)
        {
            result[i] = dimensions[i];
        }

        return result;
    }
}
