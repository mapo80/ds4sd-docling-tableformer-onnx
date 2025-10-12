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
        // TEMPORANEO: Force debug sempre abilitato
        var debugEnabled = true; // Environment.GetEnvironmentVariable("TABLEFORMER_DEBUG") == "1";

        if (debugEnabled)
        {
            DebugLog($"Input image size: {image.Width}x{image.Height}");
        }

        var tensor = Preprocess(image);

        if (debugEnabled)
        {
            DebugLog($"Preprocessed tensor shape: [{string.Join(", ", tensor.Dimensions.ToArray())}]");
            // Sample some values from the tensor
            DebugLog($"Sample tensor values (first 5 pixels, R channel): {tensor[0,0,0,0]:F3}, {tensor[0,0,0,1]:F3}, {tensor[0,0,0,2]:F3}, {tensor[0,0,0,3]:F3}, {tensor[0,0,0,4]:F3}");

            // Save preprocessed image for visual inspection
            SavePreprocessedImage(tensor, sourcePath);
        }

        var input = NamedOnnxValue.CreateFromTensor(_inputName, tensor);
        using var results = _session.Run(new[] { input });
        (input as IDisposable)?.Dispose();

        var logitsTensor = results.First(x => string.Equals(x.Name, "logits", StringComparison.OrdinalIgnoreCase)).AsTensor<float>();
        var boxesTensor = results.First(x => string.Equals(x.Name, "pred_boxes", StringComparison.OrdinalIgnoreCase)).AsTensor<float>();

        if (debugEnabled)
        {
            DebugLog($"Logits shape: [{string.Join(", ", logitsTensor.Dimensions.ToArray())}]");
            DebugLog($"Boxes shape: [{string.Join(", ", boxesTensor.Dimensions.ToArray())}]");

            var logitsArray = logitsTensor.ToArray();
            if (logitsArray.Length > 0)
            {
                DebugLog($"First 10 logits: {string.Join(", ", logitsArray.Take(10).Select(x => x.ToString("F3")))}");
            }

            var boxesArray = boxesTensor.ToArray();
            if (boxesArray.Length > 0)
            {
                DebugLog($"First 10 box values: {string.Join(", ", boxesArray.Take(10).Select(x => x.ToString("F3")))}");
            }
        }

        var logits = logitsTensor.ToArray();
        var boxes = boxesTensor.ToArray();
        var logitsShape = ToLongArray(logitsTensor.Dimensions);
        var boxesShape = ToLongArray(boxesTensor.Dimensions);

        return TableFormerDetectionParser.Parse(logits, logitsShape, boxes, boxesShape, image.Width, image.Height);
    }

    private static DenseTensor<float> Preprocess(SKBitmap bitmap)
    {
        // ImageNet normalization parameters (HuggingFace AutoImageProcessor standard)
        const float mean_r = 0.485f;
        const float mean_g = 0.456f;
        const float mean_b = 0.406f;
        const float std_r = 0.229f;
        const float std_g = 0.224f;
        const float std_b = 0.225f;
        const float inv255 = 1.0f / 255.0f;

        // Target size for TableFormer
        const int targetSize = 448;

        // TEMPORANEO: Test senza letterboxing - resize diretto a 448x448
        using var resized = bitmap.Resize(new SKImageInfo(targetSize, targetSize), SKSamplingOptions.Default);
        if (resized == null)
        {
            throw new InvalidOperationException("Failed to resize bitmap");
        }

        var data = new DenseTensor<float>(new[] { 1, 3, targetSize, targetSize });
        for (int y = 0; y < targetSize; y++)
        {
            for (int x = 0; x < targetSize; x++)
            {
                var color = resized.GetPixel(x, y);
                // Normalize to [0, 1] then apply mean/std normalization
                data[0, 0, y, x] = (color.Red * inv255 - mean_r) / std_r;
                data[0, 1, y, x] = (color.Green * inv255 - mean_g) / std_g;
                data[0, 2, y, x] = (color.Blue * inv255 - mean_b) / std_b;
            }
        }

        return data;
    }

    private static void DebugLog(string message)
    {
        var logPath = "/tmp/tableformer-debug.log";
        try
        {
            File.AppendAllText(logPath, $"[{DateTime.Now:HH:mm:ss.fff}] [TableFormerOnnx] {message}\n");
        }
        catch
        {
            // Ignore logging errors
        }
    }

    private static void SavePreprocessedImage(DenseTensor<float> tensor, string sourcePath)
    {
        try
        {
            // Denormalize tensor back to image for visualization
            const float mean_r = 0.485f;
            const float mean_g = 0.456f;
            const float mean_b = 0.406f;
            const float std_r = 0.229f;
            const float std_g = 0.224f;
            const float std_b = 0.225f;

            int width = tensor.Dimensions[3];
            int height = tensor.Dimensions[2];

            using var bitmap = new SKBitmap(width, height);

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    // Denormalize from tensor
                    float r = (tensor[0, 0, y, x] * std_r + mean_r) * 255f;
                    float g = (tensor[0, 1, y, x] * std_g + mean_g) * 255f;
                    float b = (tensor[0, 2, y, x] * std_b + mean_b) * 255f;

                    // Clamp to valid range
                    byte rByte = (byte)Math.Clamp(r, 0, 255);
                    byte gByte = (byte)Math.Clamp(g, 0, 255);
                    byte bByte = (byte)Math.Clamp(b, 0, 255);

                    bitmap.SetPixel(x, y, new SKColor(rByte, gByte, bByte));
                }
            }

            var debugPath = Path.Combine(Path.GetDirectoryName(sourcePath) ?? "/tmp",
                Path.GetFileNameWithoutExtension(sourcePath) + "_preprocessed.png");

            using var image = SKImage.FromBitmap(bitmap);
            using var data = image.Encode(SKEncodedImageFormat.Png, 100);
            using var stream = File.OpenWrite(debugPath);
            data.SaveTo(stream);

            DebugLog($"Saved preprocessed image to: {debugPath}");
        }
        catch (Exception ex)
        {
            DebugLog($"Failed to save preprocessed image: {ex.Message}");
        }
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
