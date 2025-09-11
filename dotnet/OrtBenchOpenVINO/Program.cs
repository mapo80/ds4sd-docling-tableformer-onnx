using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

record BenchConfig(string ModelPath, string ImagesFolder, string OutputBase, string VariantName,
    int TargetH, int TargetW, int Warmup, int RunsPerImage, int IntraOpNumThreads,
    int InterOpNumThreads, bool SequentialExecution, string OpenVINODevice, bool UseSynthetic);

class Program
{
    static void Main(string[] args)
    {
        string cfgPath = Path.Combine(AppContext.BaseDirectory, "BenchConfig.json");
        if (!File.Exists(cfgPath))
            cfgPath = Path.Combine(Directory.GetCurrentDirectory(), "dotnet/OrtBenchOpenVINO/BenchConfig.json");
        var cfg = JsonSerializer.Deserialize<BenchConfig>(File.ReadAllText(cfgPath))!;
        string baseDir = Path.GetDirectoryName(cfgPath)!;
        string modelPath = Path.GetFullPath(Path.Combine(baseDir, "..", cfg.ModelPath));
        string imagesFolder = Path.GetFullPath(Path.Combine(baseDir, "..", cfg.ImagesFolder));
        string outputBase = Path.GetFullPath(Path.Combine(baseDir, "..", cfg.OutputBase));

        var opts = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
        if (cfg.SequentialExecution) opts.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
        if (cfg.IntraOpNumThreads >= 0) opts.IntraOpNumThreads = cfg.IntraOpNumThreads;
        if (cfg.InterOpNumThreads >= 0) opts.InterOpNumThreads = cfg.InterOpNumThreads;

        string device = string.IsNullOrWhiteSpace(cfg.OpenVINODevice) ? "CPU_FP32" : cfg.OpenVINODevice;
        string epUsed = $"OpenVINO:{device}";
        try { opts.AppendExecutionProvider_OpenVINO(device); }
        catch (Exception ex) { Console.WriteLine($"[WARN] OpenVINO non disponibile ({ex.Message}). Fallback a CPU."); epUsed = "CPU"; }

        string variant = (cfg.VariantName ?? "custom") + "-ov";
        string timestamp = DateTime.UtcNow.ToString("yyyyMMdd-HHmmss");
        string runDir = Path.Combine(outputBase, variant, $"run-{timestamp}");
        Directory.CreateDirectory(runDir);
        var logs = new List<string>();
        void Log(string m){ Console.WriteLine(m); logs.Add(m); }

        using var session = new InferenceSession(modelPath, opts);
        var inputMeta = session.InputMetadata.First();
        var dims = inputMeta.Value.Dimensions;
        int h = dims.Length > 2 && dims[2] > 0 ? dims[2] : cfg.TargetH;
        int w = dims.Length > 3 && dims[3] > 0 ? dims[3] : cfg.TargetW;
        if (h <= 0 || w <= 0) throw new InvalidOperationException("Invalid target size");
        string inputName = inputMeta.Key;
        bool isInt8 = inputMeta.Value.ElementType == typeof(sbyte);

        string csvPath = Path.Combine(runDir, "timings.csv");
        var timings = new List<double>();
        using (var sw = new StreamWriter(csvPath))
        {
            sw.WriteLine("filename,ms");
            List<string> images = cfg.UseSynthetic
                ? new List<string> { "synthetic" }
                : Directory.EnumerateFiles(imagesFolder)
                    .Where(p => new[]{".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
                        .Contains(Path.GetExtension(p).ToLower()))
                    .ToList();

            if (images.Count > 0)
            {
                if (isInt8)
                {
                    DenseTensor<sbyte> wTensor;
                    if (cfg.UseSynthetic)
                    {
                        wTensor = new DenseTensor<sbyte>(new[] {1,3,h,w});
                        var rnd = new Random(0);
                        var span = wTensor.Buffer.Span;
                        for (int i = 0; i < span.Length; i++) span[i] = (sbyte)rnd.Next(-128,128);
                    }
                    else
                    {
                        wTensor = ImageToTensor.ToCHWInt8(images[0], h, w);
                    }
                    var warm = new[] { NamedOnnxValue.CreateFromTensor(inputName, wTensor) };
                    for (int i = 0; i < Math.Max(1, cfg.Warmup); i++) session.Run(warm);
                }
                else
                {
                    DenseTensor<float> wTensor;
                    if (cfg.UseSynthetic)
                    {
                        wTensor = new DenseTensor<float>(new[] {1,3,h,w});
                        var rnd = new Random(0);
                        var span = wTensor.Buffer.Span;
                        for (int i = 0; i < span.Length; i++) span[i] = (float)rnd.NextDouble();
                    }
                    else
                    {
                        wTensor = ImageToTensor.ToCHWFloat(images[0], h, w);
                    }
                    var warm = new[] { NamedOnnxValue.CreateFromTensor(inputName, wTensor) };
                    for (int i = 0; i < Math.Max(1, cfg.Warmup); i++) session.Run(warm);
                }
            }

            for (int idx = 0; idx < images.Count; idx++)
            {
                if (idx == 0) continue;
                string img = images[idx];
                if (isInt8)
                {
                    DenseTensor<sbyte> tensor;
                    string name;
                    if (cfg.UseSynthetic)
                    {
                        tensor = new DenseTensor<sbyte>(new[] {1,3,h,w});
                        var rnd = new Random(0);
                        var span = tensor.Buffer.Span;
                        for (int i = 0; i < span.Length; i++) span[i] = (sbyte)rnd.Next(-128,128);
                        name = "synthetic";
                    }
                    else
                    {
                        tensor = ImageToTensor.ToCHWInt8(img, h, w);
                        name = Path.GetFileName(img);
                    }
                    var cont = new[] { NamedOnnxValue.CreateFromTensor(inputName, tensor) };
                    var swatch = System.Diagnostics.Stopwatch.StartNew();
                    for (int r = 0; r < cfg.RunsPerImage; r++) session.Run(cont);
                    swatch.Stop();
                    double ms = swatch.Elapsed.TotalMilliseconds / Math.Max(1, cfg.RunsPerImage);
                    sw.WriteLine($"{name},{ms:F3}");
                    timings.Add(ms);
                    Log($"{name} {ms:F3} ms");
                }
                else
                {
                    DenseTensor<float> tensor;
                    string name;
                    if (cfg.UseSynthetic)
                    {
                        tensor = new DenseTensor<float>(new[] {1,3,h,w});
                        var rnd = new Random(0);
                        var span = tensor.Buffer.Span;
                        for (int i = 0; i < span.Length; i++) span[i] = (float)rnd.NextDouble();
                        name = "synthetic";
                    }
                    else
                    {
                        tensor = ImageToTensor.ToCHWFloat(img, h, w);
                        name = Path.GetFileName(img);
                    }
                    var cont = new[] { NamedOnnxValue.CreateFromTensor(inputName, tensor) };
                    var swatch = System.Diagnostics.Stopwatch.StartNew();
                    for (int r = 0; r < cfg.RunsPerImage; r++) session.Run(cont);
                    swatch.Stop();
                    double ms = swatch.Elapsed.TotalMilliseconds / Math.Max(1, cfg.RunsPerImage);
                    sw.WriteLine($"{name},{ms:F3}");
                    timings.Add(ms);
                    Log($"{name} {ms:F3} ms");
                }
            }
        }

        var mean = timings.Count > 0 ? timings.Average() : 0.0;
        var ordered = timings.OrderBy(x => x).ToList();
        double median = 0.0;
        if (ordered.Count > 0)
        {
            if (ordered.Count % 2 == 1) median = ordered[ordered.Count / 2];
            else median = 0.5 * (ordered[ordered.Count / 2 - 1] + ordered[ordered.Count / 2]);
        }
        double p95 = 0.0;
        if (ordered.Count > 0)
        {
            double idx = 0.95 * (ordered.Count - 1);
            int lo = (int)Math.Floor(idx);
            int hi = (int)Math.Ceiling(idx);
            p95 = lo == hi ? ordered[lo] : ordered[lo] + (idx - lo) * (ordered[hi] - ordered[lo]);
        }
        File.WriteAllText(Path.Combine(runDir, "summary.json"),
            JsonSerializer.Serialize(new { count = timings.Count, mean_ms = mean, median_ms = median, p95_ms = p95 },
                new JsonSerializerOptions { WriteIndented = true }));

        long sizeBytes = new FileInfo(modelPath).Length;
        string precision = modelPath.ToLower().Contains("int8") || (cfg.VariantName?.ToLower().Contains("int8") ?? false) ? "int8" : "fp32";
        File.WriteAllText(Path.Combine(runDir, "model_info.json"),
            JsonSerializer.Serialize(new { model_path = modelPath, model_size_bytes = sizeBytes,
                ep = epUsed.Split(':')[0], device = epUsed.Contains(':') ? epUsed.Split(':')[1] : "CPU", precision },
                new JsonSerializerOptions { WriteIndented = true }));

        string cpu = GetCpuName();
        string ortVersion = OrtEnv.Instance().GetVersionString();
        var env = new { python = (string?)null, dotnet = Environment.Version.ToString(), ort = ortVersion,
            os = System.Runtime.InteropServices.RuntimeInformation.OSDescription, cpu };
        File.WriteAllText(Path.Combine(runDir, "env.json"), JsonSerializer.Serialize(env, new JsonSerializerOptions{WriteIndented=true}));

        File.WriteAllText(Path.Combine(runDir, "config.json"), JsonSerializer.Serialize(cfg, new JsonSerializerOptions{WriteIndented=true}));
        File.WriteAllLines(Path.Combine(runDir, "logs.txt"), logs);

        var manifest = new Dictionary<string,string>();
        foreach (var file in Directory.GetFiles(runDir).Where(f => Path.GetFileName(f) != "manifest.json"))
        {
            using var stream = File.OpenRead(file);
            manifest[Path.GetFileName(file)] = BitConverter.ToString(SHA256.HashData(stream)).Replace("-","" ).ToLowerInvariant();
        }
        File.WriteAllText(Path.Combine(runDir, "manifest.json"), JsonSerializer.Serialize(manifest, new JsonSerializerOptions{WriteIndented=true}));
    }

    static string GetCpuName()
    {
        try
        {
            if (System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Linux))
            {
                foreach (var line in File.ReadLines("/proc/cpuinfo"))
                    if (line.StartsWith("model name"))
                        return line.Split(':')[1].Trim();
            }
        }
        catch {}
        return Environment.GetEnvironmentVariable("PROCESSOR_IDENTIFIER") ?? "unknown";
    }
}
