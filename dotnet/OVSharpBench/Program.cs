using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json;
using OpenVinoSharp;

record BenchConfig(string ModelPath, string OutputBase, string VariantName, int Warmup, int Runs);

class Program
{
    static void Main(string[] args)
    {
        string cfgPath = Path.Combine(AppContext.BaseDirectory, "BenchConfig.json");
        if (args.Length > 0) cfgPath = args[0];
        var cfg = JsonSerializer.Deserialize<BenchConfig>(File.ReadAllText(cfgPath))!;
        string baseDir = Path.GetDirectoryName(cfgPath)!;
        string modelPath = Path.GetFullPath(Path.Combine(baseDir, cfg.ModelPath));
        string outputBase = Path.GetFullPath(Path.Combine(baseDir, cfg.OutputBase));
        string runDir = Path.Combine(outputBase, cfg.VariantName, $"run-{DateTime.UtcNow:yyyyMMdd-HHmmss}");
        Directory.CreateDirectory(runDir);
        var logs = new List<string>();
        void Log(string m){ Console.WriteLine(m); logs.Add(m); }

        using var core = new Core();
        Log($"Model: {modelPath}");
        using var model = core.read_model(modelPath);
        using var compiled = core.compile_model(model, "CPU");
        using var request = compiled.create_infer_request();
        var inputs = model.inputs();
        string inputName = inputs[0].get_any_name();
        using var tensor = request.get_tensor(inputName);
        var shape = tensor.get_shape();
        int len = (int)shape.Aggregate(1L, (a, b) => a * (long)b);
        float[] data = new float[len];
        tensor.set_data(data);
        for (int i = 0; i < cfg.Warmup; i++) request.infer();

        var timings = new List<double>();
        var sw = new Stopwatch();
        for (int i = 0; i < cfg.Runs; i++)
        {
            sw.Restart();
            request.infer();
            sw.Stop();
            double ms = sw.Elapsed.TotalMilliseconds;
            timings.Add(ms);
            Log($"Run {i}: {ms:F3} ms");
        }

        File.WriteAllLines(Path.Combine(runDir, "timings.csv"),
            new[] { "run,ms" }.Concat(timings.Select((t, i) => $"{i},{t:F6}")));

        var sorted = timings.OrderBy(t => t).ToList();
        double mean = timings.Average();
        double median = sorted.Count % 2 == 1 ? sorted[sorted.Count / 2] : (sorted[sorted.Count / 2 - 1] + sorted[sorted.Count / 2]) / 2.0;
        double p95 = sorted[(int)Math.Floor(0.95 * (sorted.Count - 1))];
        File.WriteAllText(Path.Combine(runDir, "summary.json"),
            JsonSerializer.Serialize(new { count = timings.Count, mean_ms = mean, median_ms = median, p95_ms = p95 }, new JsonSerializerOptions { WriteIndented = true }));

        long size = new FileInfo(modelPath).Length;
        string precision = modelPath.Contains("fp16") ? "fp16" : "fp32";
        File.WriteAllText(Path.Combine(runDir, "model_info.json"),
            JsonSerializer.Serialize(new { model_path = modelPath, model_size_bytes = size, ep = "OpenVINO", device = "CPU", precision }, new JsonSerializerOptions { WriteIndented = true }));

        var env = new
        {
            dotnet = Environment.Version.ToString(),
            openvino = "OpenVINO.CSharp.API 2025.1.0.2",
            os = System.Runtime.InteropServices.RuntimeInformation.OSDescription,
            cpu = System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture.ToString()
        };
        File.WriteAllText(Path.Combine(runDir, "env.json"), JsonSerializer.Serialize(env, new JsonSerializerOptions { WriteIndented = true }));
        File.WriteAllText(Path.Combine(runDir, "config.json"), JsonSerializer.Serialize(cfg, new JsonSerializerOptions { WriteIndented = true }));

        var manifest = new Dictionary<string, string>();
        foreach (var f in Directory.GetFiles(runDir))
        {
            using var sha = SHA256.Create();
            using var fs = File.OpenRead(f);
            manifest[Path.GetFileName(f)] = Convert.ToHexString(sha.ComputeHash(fs)).ToLower();
        }
        File.WriteAllText(Path.Combine(runDir, "manifest.json"), JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true }));
        File.WriteAllLines(Path.Combine(runDir, "logs.txt"), logs);
    }
}
