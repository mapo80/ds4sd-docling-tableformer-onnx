using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using System.Linq;
using System.Collections.Generic;

namespace LayoutSdk;

internal sealed class OnnxRuntimeBackend : ILayoutBackend, IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;

    public OnnxRuntimeBackend(string modelPath)
    {
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

    public IReadOnlyList<BoundingBox> Infer(SKBitmap image)
    {
        var tensor = Preprocess(image);
        var input = NamedOnnxValue.CreateFromTensor(_inputName, tensor);
        using var results = _session.Run(new[] { input });
        (input as IDisposable)?.Dispose();
        // TODO: parse outputs into BoundingBox list
        return new List<BoundingBox>();
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
