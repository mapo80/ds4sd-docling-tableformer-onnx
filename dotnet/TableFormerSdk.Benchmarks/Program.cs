using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text.Json;
using System.Collections.Generic;
using System.Linq;
using TableFormerSdk;
using SkiaSharp;

static Dictionary<string,string> ParseArgs(string[] args)
{
    var dict = new Dictionary<string,string>();
    for (int i=0;i<args.Length;i++)
    {
        var a = args[i];
        if (a.StartsWith("--"))
        {
            var key = a;
            string val = (i+1<args.Length && !args[i+1].StartsWith("--"))? args[++i]:"true";
            dict[key]=val;
        }
    }
    return dict;
}

static string ResizeToTemp(string path, int w, int h)
{
    using var bmp = SKBitmap.Decode(path);
    using var resized = bmp.Resize(new SKImageInfo(w, h), SKFilterQuality.High);
    var tmp = Path.GetTempFileName() + ".png";
    using var img = SKImage.FromBitmap(resized);
    using var data = img.Encode(SKEncodedImageFormat.Png, 90);
    using var fs = File.OpenWrite(tmp);
    data.SaveTo(fs);
    return tmp;
}

var par = ParseArgs(args);
if (!par.TryGetValue("--variant-name", out var variant))
{
    Console.Error.WriteLine("--variant-name is required");
    return;
}
if (!par.TryGetValue("--engine", out var engStr))
{
    Console.Error.WriteLine("--engine is required");
    return;
}
var engine = Enum.Parse<TableFormerRuntime>(engStr, true);
var variantEnum = Enum.Parse<TableFormerModelVariant>(variant, true);
string images = par.GetValueOrDefault("--images", "./dataset");
string output = par.GetValueOrDefault("--output", "results");
int warmup = int.Parse(par.GetValueOrDefault("--warmup", "1"));
int runsPerImage = int.Parse(par.GetValueOrDefault("--runs-per-image", "1"));
int targetH = int.Parse(par.GetValueOrDefault("--target-h", "640"));
int targetW = int.Parse(par.GetValueOrDefault("--target-w", "640"));

var ts = DateTime.UtcNow.ToString("yyyyMMdd-HHmmss");
var runDir = Path.Combine(output, variant, $"run-{ts}");
Directory.CreateDirectory(runDir);

var options = new TableFormerSdkOptions(new TableFormerModelPaths(
    "models/tableformer-fast.onnx",
    "models/tableformer-accurate.onnx"));
using var sdk = new TableFormerSdk.TableFormerSdk(options);

// gather image files
var files = Directory.Exists(images)
    ? Directory.GetFiles(images)
        .Where(f => f.EndsWith(".jpg", true, CultureInfo.InvariantCulture) ||
                    f.EndsWith(".jpeg", true, CultureInfo.InvariantCulture) ||
                    f.EndsWith(".png", true, CultureInfo.InvariantCulture) ||
                    f.EndsWith(".bmp", true, CultureInfo.InvariantCulture) ||
                    f.EndsWith(".tif", true, CultureInfo.InvariantCulture) ||
                    f.EndsWith(".tiff", true, CultureInfo.InvariantCulture))
        .OrderBy(f=>f)
        .ToList()
    : new List<string>();
if (files.Count==0)
{
    using var bmp = new SKBitmap(targetW,targetH);
    using var img = SKImage.FromBitmap(bmp);
    var tmp = Path.GetTempFileName()+".png";
    using var data = img.Encode(SKEncodedImageFormat.Png, 90);
    using var fs = File.OpenWrite(tmp);
    data.SaveTo(fs);
    files.Add(tmp);
}

// warmup
var warmPath = ResizeToTemp(files[0], targetW, targetH);
for (int i=0;i<warmup;i++)
    sdk.Process(warmPath, false, engine, variantEnum);

var timings = new List<double>();
using (var csv = new StreamWriter(Path.Combine(runDir, "timings.csv")))
{
    csv.WriteLine("filename,ms");
    foreach (var file in files)
    {
        var prepPath = ResizeToTemp(file, targetW, targetH);
        for (int r=0;r<runsPerImage;r++)
        {
            var sw = Stopwatch.StartNew();
            sdk.Process(prepPath, false, engine, variantEnum);
            sw.Stop();
            var ms = sw.Elapsed.TotalMilliseconds;
            csv.WriteLine($"{Path.GetFileName(file)},{ms:F3}");
            timings.Add(ms);
        }
    }
}

static double Percentile(List<double> seq, double p)
{
    if (seq.Count==0) return double.NaN;
    var ordered = seq.OrderBy(x=>x).ToList();
    var idx = (int)Math.Ceiling(p/100.0*ordered.Count)-1;
    idx = Math.Clamp(idx,0,ordered.Count-1);
    return ordered[idx];
}

var summary = new {
    count = timings.Count,
    mean_ms = timings.Count>0?timings.Average():double.NaN,
    median_ms = Percentile(timings,50),
    p95_ms = Percentile(timings,95)
};
File.WriteAllText(Path.Combine(runDir,"summary.json"), JsonSerializer.Serialize(summary, new JsonSerializerOptions{WriteIndented=true}));

string modelPath = options.Onnx.GetModelPath(variantEnum);
var modelInfo = new {
    model_path = modelPath,
    model_size_bytes = File.Exists(modelPath) ? new FileInfo(modelPath).Length : 0,
    ep = "CPU",
    device = "CPU",
    precision = modelPath.ToLowerInvariant().Contains("fp16")?"fp16":"fp32"
};
File.WriteAllText(Path.Combine(runDir,"model_info.json"), JsonSerializer.Serialize(modelInfo,new JsonSerializerOptions{WriteIndented=true}));

var env = new {
    dotnet = Environment.Version.ToString(),
    os = Environment.OSVersion.ToString()
};
File.WriteAllText(Path.Combine(runDir,"env.json"), JsonSerializer.Serialize(env,new JsonSerializerOptions{WriteIndented=true}));

File.WriteAllText(Path.Combine(runDir,"config.json"), JsonSerializer.Serialize(new {
    engine = engine.ToString(),
    images,
    variant,
    warmup,
    runs_per_image = runsPerImage,
    target_h = targetH,
    target_w = targetW
}, new JsonSerializerOptions{WriteIndented=true}));

static string Sha256Of(string path)
{
    using var sha = SHA256.Create();
    using var s = File.OpenRead(path);
    return Convert.ToHexString(sha.ComputeHash(s)).ToLowerInvariant();
}

var manifest = new {
    files = new[]{"timings.csv","summary.json","model_info.json","env.json","config.json"}
        .Select(f=> new {file=f, sha256=Sha256Of(Path.Combine(runDir,f))})
};
File.WriteAllText(Path.Combine(runDir,"manifest.json"), JsonSerializer.Serialize(manifest,new JsonSerializerOptions{WriteIndented=true}));

File.WriteAllText(Path.Combine(runDir,"logs.txt"), $"RUN {variant} ok, N={timings.Count}\n");

Console.WriteLine($"OK: {runDir}");
