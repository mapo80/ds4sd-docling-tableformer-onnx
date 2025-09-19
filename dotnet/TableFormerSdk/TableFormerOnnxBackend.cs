using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using System.Collections.Generic;
using System.Linq;

namespace TableFormerSdk;

internal sealed class TableFormerOnnxBackend : ITableFormerBackend, IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;

    public TableFormerOnnxBackend(string modelPath)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path is empty", nameof(modelPath));

        var opts = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            IntraOpNumThreads = 0,
            InterOpNumThreads = 1
        };
        _session = new InferenceSession(modelPath, opts);
        opts.Dispose();
        _inputName = _session.InputMetadata.Keys.First();
    }

    public IReadOnlyList<TableRegion> Infer(SKBitmap image, string sourcePath)
    {
        var tensor = Preprocess(image);
        var input = NamedOnnxValue.CreateFromTensor(_inputName, tensor);
        using var results = _session.Run(new[] { input });
        (input as IDisposable)?.Dispose();
        // TODO: parse outputs into table structure regions.
        return new List<TableRegion>();
    }

    private static DenseTensor<float> Preprocess(SKBitmap bmp)
    {
        int w = bmp.Width;
        int h = bmp.Height;
        var data = new DenseTensor<float>(new[] { 1, 3, h, w });
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                var c = bmp.GetPixel(x, y);
                data[0, 0, y, x] = c.Red / 255f;
                data[0, 1, y, x] = c.Green / 255f;
                data[0, 2, y, x] = c.Blue / 255f;
            }
        }
        return data;
    }

    public void Dispose() => _session.Dispose();
}
