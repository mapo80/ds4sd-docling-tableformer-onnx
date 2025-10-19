using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using SkiaSharp;

var options = VisualizerOptions.Parse(args);
return VisualizerRunner.Run(options);

internal sealed record VisualizerOptions(
    DirectoryInfo DatasetDirectory,
    FileInfo BenchmarkResultsPath,
    DirectoryInfo OutputDirectory)
{
    public static VisualizerOptions Parse(IEnumerable<string> args)
    {
        var repositoryRoot = ResolveRepositoryRoot();
        var dataset = new DirectoryInfo(Path.Combine(repositoryRoot, "dataset", "FinTabNet", "benchmark"));
        var benchmarkResults = new FileInfo(Path.Combine(repositoryRoot, "results", "tableformer_docling_fintabnet_dotnet.json"));
        var output = new DirectoryInfo(Path.Combine(repositoryRoot, "results", "visualizations"));

        var enumerator = args.GetEnumerator();
        while (enumerator.MoveNext())
        {
            var current = enumerator.Current;
            switch (current)
            {
                case "--dataset":
                    if (!enumerator.MoveNext())
                    {
                        throw new ArgumentException("Missing value for --dataset");
                    }
                    dataset = new DirectoryInfo(enumerator.Current ?? string.Empty);
                    break;
                case "--results":
                    if (!enumerator.MoveNext())
                    {
                        throw new ArgumentException("Missing value for --results");
                    }
                    benchmarkResults = new FileInfo(enumerator.Current ?? string.Empty);
                    break;
                case "--output":
                    if (!enumerator.MoveNext())
                    {
                        throw new ArgumentException("Missing value for --output");
                    }
                    output = new DirectoryInfo(enumerator.Current ?? string.Empty);
                    break;
                default:
                    throw new ArgumentException($"Unknown argument '{current}'");
            }
        }

        return new VisualizerOptions(dataset, benchmarkResults, output);
    }

    private static string ResolveRepositoryRoot()
    {
        var current = AppContext.BaseDirectory;
        for (var i = 0; i < 5; i++)
        {
            current = Path.GetFullPath(Path.Combine(current, ".."));
        }
        return current;
    }
}

internal static class VisualizerRunner
{
    public static int Run(VisualizerOptions options)
    {
        if (!options.DatasetDirectory.Exists)
        {
            Console.Error.WriteLine($"Dataset directory not found: {options.DatasetDirectory.FullName}");
            return 1;
        }

        if (!options.BenchmarkResultsPath.Exists)
        {
            Console.Error.WriteLine($"Benchmark results file not found: {options.BenchmarkResultsPath.FullName}");
            Console.WriteLine("Run the benchmark first using TableFormerTorchSharpSdk.Benchmarks");
            return 1;
        }

        options.OutputDirectory.Create();

        // Load benchmark results
        var resultsJson = File.ReadAllText(options.BenchmarkResultsPath.FullName);
        using var resultsDoc = JsonDocument.Parse(resultsJson);

        if (!resultsDoc.RootElement.TryGetProperty("predictions", out var predictions))
        {
            Console.Error.WriteLine("Invalid benchmark results format: missing 'predictions'");
            return 1;
        }

        var imageFiles = options.DatasetDirectory
            .EnumerateFiles("*.png", SearchOption.TopDirectoryOnly)
            .ToDictionary(f => f.Name, f => f, StringComparer.Ordinal);

        var processedCount = 0;

        foreach (var prediction in predictions.EnumerateObject())
        {
            var imageName = prediction.Name;
            if (!imageFiles.TryGetValue(imageName, out var imageFile))
            {
                Console.WriteLine($"Skipping {imageName}: image file not found");
                continue;
            }

            Console.WriteLine($"Processing {imageName}...");

            // Load the original image
            using var originalBitmap = SKBitmap.Decode(imageFile.FullName);
            if (originalBitmap == null)
            {
                Console.Error.WriteLine($"Failed to load image: {imageName}");
                continue;
            }

            // Create a new bitmap with the same dimensions
            using var bitmap = new SKBitmap(originalBitmap.Width, originalBitmap.Height);
            using var canvas = new SKCanvas(bitmap);

            // Draw the original image
            canvas.DrawBitmap(originalBitmap, 0, 0);

            // Get tables
            if (!prediction.Value.TryGetProperty("tables", out var tables))
            {
                Console.WriteLine($"No tables found for {imageName}");
                continue;
            }

            var tableIndex = 0;
            foreach (var table in tables.EnumerateArray())
            {
                DrawTableBoundingBoxes(canvas, table, tableIndex++, originalBitmap.Width, originalBitmap.Height);
            }

            // Save the annotated image
            var outputPath = Path.Combine(options.OutputDirectory.FullName, $"{Path.GetFileNameWithoutExtension(imageName)}_annotated.png");
            using var image = SKImage.FromBitmap(bitmap);
            using var data = image.Encode(SKEncodedImageFormat.Png, 100);
            using var stream = File.OpenWrite(outputPath);
            data.SaveTo(stream);

            Console.WriteLine($"Saved annotated image to {outputPath}");
            processedCount++;
        }

        Console.WriteLine($"Processed {processedCount} images");
        Console.WriteLine($"Output directory: {options.OutputDirectory.FullName}");

        return 0;
    }

    private static void DrawTableBoundingBoxes(SKCanvas canvas, JsonElement table, int tableIndex, int imageWidth, int imageHeight)
    {
        // Define colors for different cell types
        var cellBorderPaint = new SKPaint
        {
            Style = SKPaintStyle.Stroke,
            StrokeWidth = 2,
            Color = SKColors.Red,
            IsAntialias = true
        };

        var headerPaint = new SKPaint
        {
            Style = SKPaintStyle.Stroke,
            StrokeWidth = 3,
            Color = SKColors.Blue,
            IsAntialias = true
        };

        var textBboxPaint = new SKPaint
        {
            Style = SKPaintStyle.Stroke,
            StrokeWidth = 1,
            Color = new SKColor(0, 255, 0, 128), // Semi-transparent green
            IsAntialias = true
        };

        var tableBboxPaint = new SKPaint
        {
            Style = SKPaintStyle.Stroke,
            StrokeWidth = 4,
            Color = SKColors.Purple,
            IsAntialias = true
        };

        // Draw predict_details if available
        if (table.TryGetProperty("predict_details", out var predictDetails))
        {
            // Draw table bounding box
            if (predictDetails.TryGetProperty("table_bbox", out var tableBbox))
            {
                var rect = ParseBoundingBoxArray(tableBbox, imageWidth, imageHeight);
                canvas.DrawRect(rect, tableBboxPaint);
            }

            // Draw prediction bounding boxes (cell predictions)
            if (predictDetails.TryGetProperty("prediction_bboxes_page", out var predBboxes))
            {
                foreach (var bbox in predBboxes.EnumerateArray())
                {
                    var rect = ParseBoundingBoxArray(bbox, imageWidth, imageHeight);
                    canvas.DrawRect(rect, cellBorderPaint);
                }
            }
        }

        // Draw tf_responses (docling cell responses)
        if (table.TryGetProperty("tf_responses", out var tfResponses))
        {
            foreach (var response in tfResponses.EnumerateArray())
            {
                // Determine if this is a header cell
                var isHeader = false;
                if (response.TryGetProperty("column_header", out var colHeader) && colHeader.GetBoolean())
                {
                    isHeader = true;
                }
                if (response.TryGetProperty("row_header", out var rowHeader) && rowHeader.GetBoolean())
                {
                    isHeader = true;
                }

                var paint = isHeader ? headerPaint : cellBorderPaint;

                // Draw cell bounding box
                if (response.TryGetProperty("bbox", out var cellBbox))
                {
                    var rect = ParseBoundingBoxDict(cellBbox, imageWidth, imageHeight);
                    canvas.DrawRect(rect, paint);
                }

                // Draw text cell bounding boxes
                if (response.TryGetProperty("text_cell_bboxes", out var textBboxes))
                {
                    foreach (var textBbox in textBboxes.EnumerateArray())
                    {
                        var rect = ParseBoundingBoxDict(textBbox, imageWidth, imageHeight);
                        canvas.DrawRect(rect, textBboxPaint);
                    }
                }
            }
        }

        cellBorderPaint.Dispose();
        headerPaint.Dispose();
        textBboxPaint.Dispose();
        tableBboxPaint.Dispose();
    }

    private static SKRect ParseBoundingBoxArray(JsonElement bbox, int imageWidth, int imageHeight)
    {
        // Array format: [left, top, right, bottom]
        var values = bbox.EnumerateArray().Select(e => e.GetDouble()).ToArray();
        if (values.Length != 4)
        {
            throw new InvalidOperationException("Bounding box array must have 4 elements");
        }

        return new SKRect(
            (float)values[0],
            (float)values[1],
            (float)values[2],
            (float)values[3]
        );
    }

    private static SKRect ParseBoundingBoxDict(JsonElement bbox, int imageWidth, int imageHeight)
    {
        // Dictionary format: {"l": left, "t": top, "r": right, "b": bottom}
        var left = bbox.GetProperty("l").GetDouble();
        var top = bbox.GetProperty("t").GetDouble();
        var right = bbox.GetProperty("r").GetDouble();
        var bottom = bbox.GetProperty("b").GetDouble();

        return new SKRect(
            (float)left,
            (float)top,
            (float)right,
            (float)bottom
        );
    }
}
