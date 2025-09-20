using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using TableFormerSdk.Models;

namespace TableFormerSdk.Backends;

internal sealed class TableFormerOnnxBackend : ITableFormerBackend, IDisposable
{
    private readonly IOnnxSessionAdapter _session;

    public TableFormerOnnxBackend(string modelPath)
        : this(new InferenceSessionAdapter(modelPath))
    {
    }

    internal TableFormerOnnxBackend(IOnnxSessionAdapter session)
    {
        _session = session ?? throw new ArgumentNullException(nameof(session));
    }

    public IReadOnlyList<TableRegion> Infer(SKBitmap image, string sourcePath)
    {
        var tensor = Preprocess(image);
        var outputs = _session.Run(tensor).ToList();

        var logitsOutput = FindOutput(outputs, "logits");
        var boxesOutput = FindOutput(outputs, "pred_boxes");

        return TableFormerDetectionParser.Parse(
            logitsOutput.Data,
            logitsOutput.Shape,
            boxesOutput.Data,
            boxesOutput.Shape,
            image.Width,
            image.Height);
    }

    private static OnnxOutput FindOutput(IEnumerable<OnnxOutput> outputs, string name)
    {
        foreach (var output in outputs)
        {
            if (string.Equals(output.Name, name, StringComparison.OrdinalIgnoreCase))
            {
                return output;
            }
        }

        throw new InvalidOperationException($"ONNX output '{name}' was not found");
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
}

internal interface IOnnxSessionAdapter : IDisposable
{
    IEnumerable<OnnxOutput> Run(DenseTensor<float> tensor);
}

internal sealed record OnnxOutput(string Name, float[] Data, long[] Shape);

[ExcludeFromCodeCoverage]
internal sealed class InferenceSessionAdapter : IOnnxSessionAdapter
{
    private readonly InferenceSession _session;
    private readonly string _inputName;

    public InferenceSessionAdapter(string modelPath)
        : this(modelPath, configureOptions: null)
    {
    }

    internal InferenceSessionAdapter(string modelPath, Action<SessionOptions>? configureOptions)
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

        configureOptions?.Invoke(options);

        _session = new InferenceSession(modelPath, options);
        options.Dispose();
        _inputName = _session.InputMetadata.Keys.First();
    }

    public IEnumerable<OnnxOutput> Run(DenseTensor<float> tensor)
    {
        var input = NamedOnnxValue.CreateFromTensor(_inputName, tensor);
        using var results = _session.Run(new[] { input });
        (input as IDisposable)?.Dispose();

        foreach (var value in results)
        {
            var tensorValue = value.AsTensor<float>();
            var shape = ToLongArray(tensorValue.Dimensions);
            var data = tensorValue.ToArray();
            (value as IDisposable)?.Dispose();
            yield return new OnnxOutput(value.Name, data, shape);
        }
    }

    private static long[] ToLongArray(ReadOnlySpan<int> dimensions)
    {
        var result = new long[dimensions.Length];
        for (int i = 0; i < dimensions.Length; i++)
        {
            result[i] = dimensions[i];
        }

        return result;
    }

    public void Dispose() => _session.Dispose();

    public static InferenceSessionAdapter CreateOrt(string modelPath)
    {
        return new InferenceSessionAdapter(modelPath, ConfigureOrtSessionOptions);
    }

    private static void ConfigureOrtSessionOptions(SessionOptions options)
    {
        options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL;
        options.AddSessionConfigEntry("session.load_model_format", "ORT");
    }
}
