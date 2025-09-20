using OpenVinoSharp;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TableFormerSdk.Models;

namespace TableFormerSdk.Backends;

internal sealed class OpenVinoBackend : ITableFormerBackend, IDisposable
{
    private readonly Core _core;
    private readonly Model _model;
    private readonly CompiledModel _compiledModel;
    private readonly InferRequest _request;
    private readonly string _inputName;
    private readonly string _logitsOutputName;
    private readonly string? _boxesOutputName;
    private readonly int _modelInputWidth;
    private readonly int _modelInputHeight;

    public OpenVinoBackend(string modelPath)
    {
        EnsureNativeBinariesAreAvailable();

        _core = new Core();
        _model = _core.read_model(modelPath);
        _compiledModel = _core.compile_model(_model, "CPU");
        _request = _compiledModel.create_infer_request();
        var modelInputs = _model.inputs();
        _inputName = modelInputs[0].get_any_name();
        var inputShape = modelInputs[0].get_shape();
        _modelInputHeight = (int)inputShape[inputShape.Count - 2];
        _modelInputWidth = (int)inputShape[inputShape.Count - 1];
        var outputs = _model.outputs();
        if (outputs.Count == 0)
        {
            throw new InvalidOperationException("OpenVINO model does not expose any outputs");
        }

        _logitsOutputName = outputs[0].get_any_name();
        _boxesOutputName = outputs.Count > 1 ? outputs[1].get_any_name() : null;
    }

    public IReadOnlyList<TableRegion> Infer(SKBitmap image, string sourcePath)
    {
        int outputWidth = image.Width;
        int outputHeight = image.Height;

        using var resized = (outputWidth == _modelInputWidth && outputHeight == _modelInputHeight)
            ? null
            : image.Resize(new SKImageInfo(_modelInputWidth, _modelInputHeight), new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.Linear));

        var working = resized ?? image;
        int width = working.Width;
        int height = working.Height;
        float[] data = new float[1 * 3 * height * width];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                var color = working.GetPixel(x, y);
                int index = y * width + x;
                data[index] = color.Red / 255f;
                data[index + height * width] = color.Green / 255f;
                data[index + (2 * height * width)] = color.Blue / 255f;
            }
        }

        using var tensor = new Tensor(new Shape(new long[] { 1, 3, height, width }), data);
        _request.set_tensor(_inputName, tensor);
        _request.infer();

        using var logitsTensor = _request.get_tensor(_logitsOutputName);

        if (_boxesOutputName is null)
        {
            return new List<TableRegion>
            {
                new TableRegion(0, 0, outputWidth, outputHeight, "openvino-region")
            };
        }

        using var boxesTensor = _request.get_tensor(_boxesOutputName);

        var logitsShape = logitsTensor.get_shape();
        var boxesShape = boxesTensor.get_shape();
        int logitsLength = (int)logitsShape.Aggregate(1L, (current, dim) => current * dim);
        int boxesLength = (int)boxesShape.Aggregate(1L, (current, dim) => current * dim);

        var logits = logitsTensor.get_data<float>(logitsLength);
        var boxes = boxesTensor.get_data<float>(boxesLength);

        return TableFormerDetectionParser.Parse(logits, logitsShape.ToArray(), boxes, boxesShape.ToArray(), outputWidth, outputHeight);
    }

    public void Dispose()
    {
        _request.Dispose();
        _compiledModel.Dispose();
        _model.Dispose();
        _core.Dispose();
    }

    private static void EnsureNativeBinariesAreAvailable()
    {
        var nugetPackages = Environment.GetEnvironmentVariable("NUGET_PACKAGES")
                            ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".nuget", "packages");
        var runtimesDirectory = Path.Combine(nugetPackages, "openvino.runtime.ubuntu.24-x86_64");
        if (!Directory.Exists(runtimesDirectory))
        {
            return;
        }

        var latestRuntime = Directory.GetDirectories(runtimesDirectory).OrderByDescending(path => path).FirstOrDefault();
        if (latestRuntime is null)
        {
            return;
        }

        var nativeDirectory = Path.Combine(latestRuntime, "runtimes", "ubuntu.24-x86_64", "native");
        if (!Directory.Exists(nativeDirectory))
        {
            return;
        }

        var executionDirectory = AppContext.BaseDirectory;
        foreach (var source in Directory.GetFiles(nativeDirectory))
        {
            var destination = Path.Combine(executionDirectory, Path.GetFileName(source));
            if (!File.Exists(destination))
            {
                File.Copy(source, destination);
            }

            try
            {
                System.Runtime.InteropServices.NativeLibrary.Load(destination);
            }
            catch
            {
                // ignored on purpose - loading will be attempted again by the runtime if required
            }
        }
    }
}
