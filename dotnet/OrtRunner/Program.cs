using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

class Program
{
    record OrtConfig(string ModelPath, string InputFolder, string OutputFolder,
        int IntraOpNumThreads, int InterOpNumThreads, bool SequentialExecution,
        string ExecutionProvider, int BatchSize, int Warmup, int RunsPerImage);

    static void Main()
    {
        var cfgPath = Path.Combine(AppContext.BaseDirectory, "OrtConfig.json");
        if (!File.Exists(cfgPath))
            cfgPath = Path.Combine(Directory.GetCurrentDirectory(), "dotnet/OrtRunner/OrtConfig.json");
        var baseDir = Path.GetDirectoryName(cfgPath)!;
        var modelPath = Path.GetFullPath(Path.Combine(baseDir, cfg.ModelPath));
        var inputFolder = Path.GetFullPath(Path.Combine(baseDir, cfg.InputFolder));
        var outputFolder = Path.GetFullPath(Path.Combine(baseDir, cfg.OutputFolder));
        Directory.CreateDirectory(outputFolder);
        var so = new SessionOptions();
        so.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        so.ExecutionMode = cfg.SequentialExecution ? ExecutionMode.ORT_SEQUENTIAL : ExecutionMode.ORT_PARALLEL;
        if (cfg.IntraOpNumThreads > 0) so.IntraOpNumThreads = cfg.IntraOpNumThreads;
        if (cfg.InterOpNumThreads > 0) so.InterOpNumThreads = cfg.InterOpNumThreads;
        if (!string.Equals(cfg.ExecutionProvider, "CPU", StringComparison.OrdinalIgnoreCase))
        {
            if (cfg.ExecutionProvider.Equals("OpenVINO", StringComparison.OrdinalIgnoreCase))
                so.AppendExecutionProvider_OpenVINO();
            else if (cfg.ExecutionProvider.Equals("DirectML", StringComparison.OrdinalIgnoreCase))
                so.AppendExecutionProvider_DML();
        }

        using var session = new InferenceSession(modelPath, so);
        foreach (var kv in session.InputMetadata)
            Console.WriteLine($"Input: {kv.Key} -> [{string.Join(',', kv.Value.Dimensions)}]");
        foreach (var kv in session.OutputMetadata)
            Console.WriteLine($"Output: {kv.Key} -> [{string.Join(',', kv.Value.Dimensions)}]");

        var meta = session.InputMetadata.First().Value;
        if (meta.Dimensions.Length != 4 || meta.Dimensions[2] <= 0 || meta.Dimensions[3] <= 0)
            throw new InvalidOperationException("Model input height/width must be static");
        int h = meta.Dimensions[2];
        int w = meta.Dimensions[3];
        float[] mean = { 0.485f, 0.456f, 0.406f };
        float[] std = { 0.229f, 0.224f, 0.225f };

        var timingsPath = Path.Combine(outputFolder, "timings.csv");
        using var tw = new StreamWriter(timingsPath);
        long sizeBytes = new FileInfo(modelPath).Length;
        File.WriteAllText(Path.Combine(outputFolder, "model_info.json"),
            $"{{\n  \"model_size_bytes\": {sizeBytes}\n}}\n");

        foreach (var imgPath in Directory.GetFiles(inputFolder))
        {
            var ext = Path.GetExtension(imgPath).ToLower();
            if (ext != ".png" && ext != ".jpg" && ext != ".jpeg" && ext != ".bmp" && ext != ".tif" && ext != ".tiff")
                continue;
            var tensor = ImageToTensor.ToCHWFloat(imgPath, h, w, mean, std);
            var inputName = session.InputNames[0];
            for (int i = 0; i < cfg.Warmup; i++)
                session.Run(new[] { NamedOnnxValue.CreateFromTensor(inputName, tensor) });
            var sw = System.Diagnostics.Stopwatch.StartNew();
            for (int r = 0; r < cfg.RunsPerImage; r++)
                session.Run(new[] { NamedOnnxValue.CreateFromTensor(inputName, tensor) });
            sw.Stop();
            var ms = sw.Elapsed.TotalMilliseconds / Math.Max(1, cfg.RunsPerImage);
            tw.WriteLine($"{Path.GetFileName(imgPath)},{ms:F3}");
        }
    }
}
