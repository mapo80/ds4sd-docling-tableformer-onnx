using OpenVinoSharp;
using SkiaSharp;
using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;

namespace TableFormerSdk;

internal sealed class OpenVinoBackend : ITableFormerBackend, IDisposable
{
    private readonly Core _core;
    private readonly Model _model;
    private readonly CompiledModel _compiled;
    private readonly InferRequest _request;
    private readonly string _inputName;

    public OpenVinoBackend(string modelPath)
    {
        // copy OpenVINO native libraries next to the executable so P/Invoke can resolve them
        var nuget = Environment.GetEnvironmentVariable("NUGET_PACKAGES")
                   ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".nuget", "packages");
        var runtimes = Path.Combine(nuget, "openvino.runtime.ubuntu.24-x86_64");
        if (Directory.Exists(runtimes))
        {
            var latest = Directory.GetDirectories(runtimes).OrderByDescending(p => p).FirstOrDefault();
            if (latest != null)
            {
                var native = Path.Combine(latest, "runtimes", "ubuntu.24-x86_64", "native");
                var execDir = AppContext.BaseDirectory;
                foreach (var src in Directory.GetFiles(native))
                {
                    var dest = Path.Combine(execDir, Path.GetFileName(src));
                    if (!File.Exists(dest))
                    {
                        File.Copy(src, dest);
                    }
                    try { System.Runtime.InteropServices.NativeLibrary.Load(dest); } catch { }
                }
            }
        }

        _core = new Core();
        _model = _core.read_model(modelPath);
        _compiled = _core.compile_model(_model, "CPU");
        _request = _compiled.create_infer_request();
        _inputName = _model.inputs()[0].get_any_name();
    }

    public IReadOnlyList<TableRegion> Infer(SKBitmap image, string sourcePath)
    {
        int w = image.Width;
        int h = image.Height;
        float[] data = new float[1 * 3 * h * w];
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                var c = image.GetPixel(x, y);
                int idx = y * w + x;
                data[idx] = c.Red / 255f;
                data[idx + h * w] = c.Green / 255f;
                data[idx + 2 * h * w] = c.Blue / 255f;
            }
        }
        using var tensor = new Tensor(new Shape(new long[] { 1, 3, h, w }), data);
        _request.set_tensor(_inputName, tensor);
        _request.infer();
        return new List<TableRegion>();
    }

    public void Dispose()
    {
        _request.Dispose();
        _compiled.Dispose();
        _model.Dispose();
        _core.Dispose();
    }
}
