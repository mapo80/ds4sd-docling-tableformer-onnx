using SkiaSharp;
using System.Globalization;
using TableFormerSdk;
using TableFormerSdk.Configuration;
using TableFormerSdk.Enums;
using TableFormerSdk.Models;

var imagePath = ResolveSampleImagePath();
if (!File.Exists(imagePath))
{
    Console.Error.WriteLine($"Sample image not found: {imagePath}");
    return 1;
}

var outputDir = Path.Combine(ResolveRepoRoot(), "results", "tableformer-release-verification");
Directory.CreateDirectory(outputDir);

var options = new TableFormerSdkOptions();
Console.WriteLine("Docling TableFormer SDK - release verification");
Console.WriteLine($"Sample image: {imagePath}");
Console.WriteLine($"Output directory: {outputDir}");
Console.WriteLine();

var preparedImagePath = PrepareImageForInference(imagePath, outputDir);
if (!string.Equals(preparedImagePath, imagePath, StringComparison.Ordinal))
{
    Console.WriteLine($"Preprocessed image saved to {preparedImagePath}");
    Console.WriteLine();
}

Console.WriteLine("Available runtimes from packaged catalog:");
foreach (var runtime in options.AvailableRuntimes)
{
    Console.WriteLine($" - {runtime}");
}
Console.WriteLine();

if (options.ModelCatalog is ReleaseModelCatalog releaseCatalog)
{
    foreach (var runtime in options.AvailableRuntimes)
    {
        foreach (TableFormerModelVariant variant in Enum.GetValues<TableFormerModelVariant>())
        {
            if (!releaseCatalog.SupportsVariant(runtime, variant))
            {
                continue;
            }

            var artifact = releaseCatalog.GetArtifact(runtime, variant);
            Console.WriteLine($"Artifact for {runtime}/{variant}: {artifact.ModelPath}");
            if (!string.IsNullOrWhiteSpace(artifact.WeightsPath))
            {
                Console.WriteLine($"  Weights: {artifact.WeightsPath}");
            }
        }
    }

    Console.WriteLine();
}

var requestedRuntimes = new[]
{
    TableFormerRuntime.Onnx,
    TableFormerRuntime.Ort,
    TableFormerRuntime.OpenVino
};

using var sdk = new TableFormerSdk.TableFormerSdk(options);
var metrics = new List<ResultMetrics>();

foreach (var runtime in requestedRuntimes)
{
    if (!options.AvailableRuntimes.Contains(runtime))
    {
        Console.WriteLine($"Runtime {runtime} not available in release package, skipping.");
        Console.WriteLine();
        continue;
    }

    foreach (TableFormerModelVariant variant in Enum.GetValues<TableFormerModelVariant>())
    {
        Console.WriteLine($"Running {runtime} / {variant}...");
        try
        {
            var result = sdk.Process(preparedImagePath, overlay: true, variant: variant, runtime: runtime);
            var overlayFile = SaveOverlayIfPresent(result, outputDir, runtime, variant, imagePath);
            var metric = BuildMetrics(result, runtime, variant, overlayFile);
            metrics.Add(metric);
            PrintMetrics(metric);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  Failed: {ex.GetType().Name}: {ex.Message}");
        }

        Console.WriteLine();
    }
}

if (metrics.Count == 0)
{
    Console.Error.WriteLine("No runtimes were executed. Ensure the NuGet package models are available.");
    return 1;
}

var summaryPath = Path.Combine(outputDir, "summary.csv");
WriteSummaryCsv(summaryPath, metrics);
Console.WriteLine($"Summary written to {summaryPath}");

return 0;

static string ResolveRepoRoot()
{
    var current = AppContext.BaseDirectory;
    for (var i = 0; i < 5; i++)
    {
        current = Path.GetFullPath(Path.Combine(current, ".."));
    }

    return current;
}

static string ResolveSampleImagePath()
{
    var repoRoot = ResolveRepoRoot();
    return Path.Combine(repoRoot, "dataset", "FinTabNet", "images", "HAL.2004.page_82.pdf_125315.png");
}

static string PrepareImageForInference(string sourcePath, string workDir)
{
    const int targetSize = 448;

    using var bitmap = SKBitmap.Decode(sourcePath) ?? throw new InvalidOperationException($"Unable to decode {sourcePath}");
    if (bitmap.Width == targetSize && bitmap.Height == targetSize)
    {
        return sourcePath;
    }

    var scale = Math.Min((float)targetSize / bitmap.Width, (float)targetSize / bitmap.Height);
    var scaledWidth = Math.Max(1, (int)MathF.Round(bitmap.Width * scale));
    var scaledHeight = Math.Max(1, (int)MathF.Round(bitmap.Height * scale));

    using var resized = bitmap.Resize(new SKImageInfo(scaledWidth, scaledHeight), new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.None))
        ?? throw new InvalidOperationException("Unable to resize bitmap for preprocessing");

    using var canvasBitmap = new SKBitmap(targetSize, targetSize);
    using (var canvas = new SKCanvas(canvasBitmap))
    {
        canvas.Clear(SKColors.White);
        var offsetX = (targetSize - scaledWidth) / 2f;
        var offsetY = (targetSize - scaledHeight) / 2f;
        canvas.DrawBitmap(resized, new SKRect(offsetX, offsetY, offsetX + scaledWidth, offsetY + scaledHeight));
        canvas.Flush();
    }

    var preprocessedPath = Path.Combine(workDir, Path.GetFileNameWithoutExtension(sourcePath) + "_prepared.png");
    using var skImage = SKImage.FromBitmap(canvasBitmap);
    using var data = skImage.Encode(SKEncodedImageFormat.Png, 100);
    using (var stream = File.Open(preprocessedPath, FileMode.Create, FileAccess.Write, FileShare.None))
    {
        data.SaveTo(stream);
    }

    return preprocessedPath;
}

static ResultMetrics BuildMetrics(
    TableStructureResult result,
    TableFormerRuntime runtime,
    TableFormerModelVariant variant,
    string? overlayFile)
{
    var regions = result.Regions;
    var regionCount = regions.Count;
    var averageWidth = regionCount > 0 ? regions.Average(r => r.Width) : 0f;
    var averageHeight = regionCount > 0 ? regions.Average(r => r.Height) : 0f;
    var averageArea = regionCount > 0 ? regions.Average(r => r.Width * r.Height) : 0f;

    return new ResultMetrics(
        runtime,
        variant,
        result.InferenceTime.TotalMilliseconds,
        regionCount,
        averageWidth,
        averageHeight,
        averageArea,
        overlayFile,
        result.PerformanceSnapshot);
}

static void PrintMetrics(ResultMetrics metric)
{
    Console.WriteLine(
        $"  Regions: {metric.RegionCount}, average size: {metric.AverageWidth:F1}x{metric.AverageHeight:F1} ({metric.AverageArea:F1} px^2)");
    Console.WriteLine($"  Inference time: {metric.InferenceMilliseconds:F1} ms (runtime {metric.Snapshot.Runtime})");
    Console.WriteLine($"  Overlay: {(metric.OverlayPath is null ? "not generated" : metric.OverlayPath)}");
}

static string? SaveOverlayIfPresent(
    TableStructureResult result,
    string outputDir,
    TableFormerRuntime runtime,
    TableFormerModelVariant variant,
    string imagePath)
{
    if (result.OverlayImage is not { } overlay)
    {
        return null;
    }

    var filename = string.Format(
        CultureInfo.InvariantCulture,
        "{0}_{1}_{2}.png",
        Path.GetFileNameWithoutExtension(imagePath) ?? "image",
        runtime.ToString().ToLowerInvariant(),
        variant.ToString().ToLowerInvariant());

    var overlayPath = Path.Combine(outputDir, filename);
    using var data = overlay.Encode(SKEncodedImageFormat.Png, 100);
    using var stream = File.Open(overlayPath, FileMode.Create, FileAccess.Write, FileShare.None);
    data.SaveTo(stream);
    overlay.Dispose();
    return overlayPath;
}

static void WriteSummaryCsv(string path, IEnumerable<ResultMetrics> metrics)
{
    using var writer = new StreamWriter(path);
    writer.WriteLine("runtime,variant,inference_ms,regions,avg_width,avg_height,avg_area,overlay_path,total_samples");
    foreach (var metric in metrics)
    {
        writer.WriteLine(string.Join(",", new[]
        {
            metric.Runtime.ToString(),
            metric.Variant.ToString(),
            metric.InferenceMilliseconds.ToString("F1", CultureInfo.InvariantCulture),
            metric.RegionCount.ToString(CultureInfo.InvariantCulture),
            metric.AverageWidth.ToString("F1", CultureInfo.InvariantCulture),
            metric.AverageHeight.ToString("F1", CultureInfo.InvariantCulture),
            metric.AverageArea.ToString("F1", CultureInfo.InvariantCulture),
            metric.OverlayPath ?? string.Empty,
            metric.Snapshot.TotalSampleCount.ToString(CultureInfo.InvariantCulture)
        }));
    }
}

file sealed record ResultMetrics(
    TableFormerRuntime Runtime,
    TableFormerModelVariant Variant,
    double InferenceMilliseconds,
    int RegionCount,
    float AverageWidth,
    float AverageHeight,
    float AverageArea,
    string? OverlayPath,
    TableFormerSdk.Performance.TableFormerPerformanceSnapshot Snapshot);
