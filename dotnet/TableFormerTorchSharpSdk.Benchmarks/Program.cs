using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;

using TableFormerTorchSharpSdk.Artifacts;
using TableFormerTorchSharpSdk.Decoding;
using TableFormerTorchSharpSdk.Matching;
using TableFormerTorchSharpSdk.PagePreparation;
using TableFormerTorchSharpSdk.Results;
using TableFormerTorchSharpSdk.Tensorization;
using TableFormerTorchSharpSdk.Utilities;

var options = BenchmarkOptions.Parse(args);
return await BenchmarkRunner.RunAsync(options);

internal sealed record BenchmarkOptions(
    DirectoryInfo DatasetDirectory,
    FileInfo OutputPath,
    FileInfo ReferencePath,
    DirectoryInfo ArtifactCacheDirectory,
    int NumThreads,
    bool SkipReferenceCheck)
{
    public static BenchmarkOptions Parse(IEnumerable<string> args)
    {
        var repositoryRoot = ResolveRepositoryRoot();
        var dataset = new DirectoryInfo(Path.Combine(repositoryRoot, "dataset", "FinTabNet", "benchmark"));
        var output = new FileInfo(Path.Combine(repositoryRoot, "results", "tableformer_docling_fintabnet_dotnet.json"));
        var reference = new FileInfo(Path.Combine(repositoryRoot, "results", "tableformer_docling_fintabnet.json"));
        var cacheDirectory = new DirectoryInfo(Path.Combine(repositoryRoot, "dotnet", "artifacts_benchmark_cache"));
        var numThreads = Environment.ProcessorCount;
        var skipReferenceCheck = false;

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
                case "--output":
                    if (!enumerator.MoveNext())
                    {
                        throw new ArgumentException("Missing value for --output");
                    }

                    output = new FileInfo(enumerator.Current ?? string.Empty);
                    break;
                case "--reference":
                    if (!enumerator.MoveNext())
                    {
                        throw new ArgumentException("Missing value for --reference");
                    }

                    reference = new FileInfo(enumerator.Current ?? string.Empty);
                    break;
                case "--cache-dir":
                    if (!enumerator.MoveNext())
                    {
                        throw new ArgumentException("Missing value for --cache-dir");
                    }

                    cacheDirectory = new DirectoryInfo(enumerator.Current ?? string.Empty);
                    break;
                case "--num-threads":
                    if (!enumerator.MoveNext())
                    {
                        throw new ArgumentException("Missing value for --num-threads");
                    }

                    if (!int.TryParse(enumerator.Current, NumberStyles.Integer, CultureInfo.InvariantCulture, out numThreads))
                    {
                        throw new ArgumentException("--num-threads must be an integer value");
                    }

                    if (numThreads <= 0)
                    {
                        throw new ArgumentOutOfRangeException(nameof(numThreads), numThreads, "Thread count must be positive.");
                    }

                    break;
                case "--skip-reference-check":
                    skipReferenceCheck = true;
                    break;
                default:
                    throw new ArgumentException($"Unknown argument '{current}'");
            }
        }

        return new BenchmarkOptions(dataset, output, reference, cacheDirectory, numThreads, skipReferenceCheck);
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

internal static class BenchmarkRunner
{
    public static async Task<int> RunAsync(BenchmarkOptions options)
    {
        if (!options.DatasetDirectory.Exists)
        {
            Console.Error.WriteLine($"Dataset directory not found: {options.DatasetDirectory.FullName}");
            return 1;
        }

        options.OutputPath.Directory?.Create();

        TorchSharp.torch.set_num_threads(options.NumThreads);
        TorchSharp.torch.set_num_interop_threads(Math.Max(1, Math.Min(options.NumThreads, Environment.ProcessorCount)));

        using var bootstrapper = new TableFormerArtifactBootstrapper(options.ArtifactCacheDirectory);
        var bootstrapResult = await bootstrapper.EnsureArtifactsAsync().ConfigureAwait(false);
        var initializationSnapshot = await bootstrapResult.InitializePredictorAsync().ConfigureAwait(false);

        using var neuralModel = new TableFormerTorchSharpSdk.Model.TableFormerNeuralModel(
            bootstrapResult.ConfigSnapshot,
            initializationSnapshot,
            bootstrapResult.ModelDirectory);

        var decoder = new TableFormerSequenceDecoder(initializationSnapshot);
        var cellMatcher = new TableFormerCellMatcher(bootstrapResult.ConfigSnapshot);
        var cropper = new TableFormerTableCropper();
        var tensorizer = TableFormerImageTensorizer.FromConfig(bootstrapResult.ConfigSnapshot);
        var preparer = new TableFormerPageInputPreparer();
        var postProcessor = new TableFormerMatchingPostProcessor();
        var assembler = new TableFormerDoclingResponseAssembler();

        var predictions = new Dictionary<string, Dictionary<string, object?>>(StringComparer.Ordinal);
        var timings = new Dictionary<string, double>(StringComparer.Ordinal);

        var imageFiles = options.DatasetDirectory
            .EnumerateFiles("*.png", SearchOption.TopDirectoryOnly)
            .OrderBy(file => file.Name, StringComparer.Ordinal)
            .ToList();

        if (imageFiles.Count == 0)
        {
            Console.Error.WriteLine($"No PNG files found in {options.DatasetDirectory.FullName}");
            return 1;
        }

        foreach (var imageFile in imageFiles)
        {
            var decodedImage = TableFormerDecodedPageImage.Decode(imageFile);
            var stopwatch = Stopwatch.StartNew();
            var pageSnapshot = preparer.PreparePageInput(decodedImage);
            var cropSnapshot = cropper.PrepareTableCrops(decodedImage, pageSnapshot.TableBoundingBoxes);
            var tables = new List<Dictionary<string, object?>>(capacity: cropSnapshot.TableCrops.Count);

            for (var tableIndex = 0; tableIndex < cropSnapshot.TableCrops.Count; tableIndex++)
            {
                var crop = cropSnapshot.TableCrops[tableIndex];
                using var tensorSnapshot = tensorizer.CreateTensor(crop);
                using var prediction = neuralModel.Predict(tensorSnapshot.Tensor);

                var decoded = decoder.Decode(prediction);
                var matchingResult = cellMatcher.MatchCells(pageSnapshot, crop, decoded);
                var matchingDetails = matchingResult.ToMatchingDetails();
                var processed = pageSnapshot.Tokens.Count > 0
                    ? postProcessor.Process(matchingDetails.ToMutable(), correctOverlappingCells: false)
                    : matchingDetails;
                var assembled = assembler.Assemble(processed, decoded, sortRowColIndexes: true);

                tables.Add(assembled.ToDictionary());
            }

            stopwatch.Stop();
            timings[imageFile.Name] = stopwatch.Elapsed.TotalMilliseconds;

            predictions[imageFile.Name] = new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["num_tables"] = tables.Count,
                ["tables"] = tables,
            };
        }

        var benchmarkReport = new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["predictions"] = predictions,
            ["timings_ms"] = timings,
            ["summary"] = new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["num_documents"] = imageFiles.Count,
                ["total_ms"] = timings.Values.Sum(),
                ["average_ms"] = timings.Values.Average(),
            },
        };

        var json = JsonSerializer.Serialize(benchmarkReport, new JsonSerializerOptions
        {
            WriteIndented = true,
        });
        await File.WriteAllTextAsync(options.OutputPath.FullName, json).ConfigureAwait(false);

        Console.WriteLine($"Benchmark report saved to {options.OutputPath.FullName}");
        foreach (var pair in timings)
        {
            Console.WriteLine($"{pair.Key}: {pair.Value.ToString("F2", CultureInfo.InvariantCulture)} ms");
        }

        Console.WriteLine(
            $"Average: {timings.Values.Average().ToString("F2", CultureInfo.InvariantCulture)} ms " +
            $"over {imageFiles.Count} documents");

        if (!options.SkipReferenceCheck && options.ReferencePath.Exists)
        {
            var referenceJson = await File.ReadAllTextAsync(options.ReferencePath.FullName).ConfigureAwait(false);
            using var referenceDocument = JsonDocument.Parse(referenceJson);
            var referenceCanonical = JsonCanonicalizer.GetCanonicalJson(referenceDocument.RootElement);

            using var actualDocument = JsonDocument.Parse(JsonSerializer.Serialize(predictions));
            var actualCanonical = JsonCanonicalizer.GetCanonicalJson(actualDocument.RootElement);

            if (!string.Equals(referenceCanonical, actualCanonical, StringComparison.Ordinal))
            {
                Console.Error.WriteLine(
                    "TorchSharp benchmark output diverges from canonical reference. " +
                    "Use --skip-reference-check to collect timings anyway.");
                return 2;
            }
        }

        return 0;
    }
}
