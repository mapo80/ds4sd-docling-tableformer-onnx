using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

using TableFormerTorchSharpSdk.Artifacts;
using TableFormerTorchSharpSdk.Decoding;
using TableFormerTorchSharpSdk.Matching;
using TableFormerTorchSharpSdk.PagePreparation;
using TableFormerTorchSharpSdk.Results;
using TableFormerTorchSharpSdk.Tensorization;
using TableFormerTorchSharpSdk.Utilities;

var options = BenchmarkOptions.Parse(args);
return await BenchmarkRunner.RunAsync(options).ConfigureAwait(false);

internal sealed record BenchmarkOptions(
    string RepositoryRoot,
    DirectoryInfo DatasetDirectory,
    DirectoryInfo RunsDirectory,
    FileInfo OutputPath,
    FileInfo ReferencePath,
    DirectoryInfo ArtifactCacheDirectory,
    int RequestedThreads,
    int Iterations,
    string Label,
    FileInfo? BaselinePath,
    FileInfo MarkdownReportPath,
    bool SkipReferenceCheck)
{
    public static BenchmarkOptions Parse(IEnumerable<string> args)
    {
        var repositoryRoot = ResolveRepositoryRoot();
        var dataset = new DirectoryInfo(Path.Combine(repositoryRoot, "dataset", "FinTabNet", "benchmark"));
        var runsDirectory = new DirectoryInfo(Path.Combine(repositoryRoot, "results", "perf_runs"));
        var output = new FileInfo(Path.Combine(repositoryRoot, "results", "tableformer_docling_fintabnet_dotnet.json"));
        var reference = new FileInfo(Path.Combine(repositoryRoot, "results", "tableformer_docling_fintabnet.json"));
        var cacheDirectory = new DirectoryInfo(Path.Combine(repositoryRoot, "dotnet", "artifacts_benchmark_cache"));
        var requestedThreads = Environment.ProcessorCount;
        var iterations = 1;
        var label = "optimized";
        FileInfo? baseline = null;
        var reportPath = new FileInfo(Path.Combine(repositoryRoot, "results", "performance_report.md"));
        var skipReferenceCheck = false;

        var enumerator = args.GetEnumerator();
        while (enumerator.MoveNext())
        {
            var current = enumerator.Current;
            switch (current)
            {
                case "--dataset":
                    dataset = new DirectoryInfo(ReadValue("--dataset"));
                    break;
                case "--runs-dir":
                    runsDirectory = new DirectoryInfo(ReadValue("--runs-dir"));
                    break;
                case "--output":
                    output = new FileInfo(ReadValue("--output"));
                    break;
                case "--reference":
                    reference = new FileInfo(ReadValue("--reference"));
                    break;
                case "--cache-dir":
                    cacheDirectory = new DirectoryInfo(ReadValue("--cache-dir"));
                    break;
                case "--num-threads":
                    if (!int.TryParse(ReadValue("--num-threads"), NumberStyles.Integer, CultureInfo.InvariantCulture, out requestedThreads))
                    {
                        throw new ArgumentException("--num-threads must be an integer value");
                    }

                    if (requestedThreads <= 0)
                    {
                        throw new ArgumentOutOfRangeException(nameof(requestedThreads), requestedThreads, "Thread count must be positive.");
                    }

                    break;
                case "--iterations":
                    if (!int.TryParse(ReadValue("--iterations"), NumberStyles.Integer, CultureInfo.InvariantCulture, out iterations))
                    {
                        throw new ArgumentException("--iterations must be an integer value");
                    }

                    if (iterations <= 0)
                    {
                        throw new ArgumentOutOfRangeException(nameof(iterations), iterations, "Iteration count must be positive.");
                    }

                    break;
                case "--label":
                    label = ReadValue("--label");
                    if (string.IsNullOrWhiteSpace(label))
                    {
                        throw new ArgumentException("Label cannot be empty.");
                    }

                    break;
                case "--baseline":
                    baseline = new FileInfo(ReadValue("--baseline"));
                    break;
                case "--report":
                    reportPath = new FileInfo(ReadValue("--report"));
                    break;
                case "--skip-reference-check":
                    skipReferenceCheck = true;
                    break;
                default:
                    throw new ArgumentException($"Unknown argument '{current}'");
            }
        }

        return new BenchmarkOptions(
            repositoryRoot,
            dataset,
            runsDirectory,
            output,
            reference,
            cacheDirectory,
            requestedThreads,
            iterations,
            label,
            baseline,
            reportPath,
            skipReferenceCheck);

        string ReadValue(string name)
        {
            if (!enumerator.MoveNext())
            {
                throw new ArgumentException($"Missing value for {name}");
            }

            return enumerator.Current ?? string.Empty;
        }
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
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    private static class StageNames
    {
        public const string DecodeImage = "decode_image_ms";
        public const string PreparePage = "prepare_page_ms";
        public const string CropTables = "crop_tables_ms";
        public const string Tensorize = "tensorize_ms";
        public const string ModelInference = "model_inference_ms";
        public const string SequenceDecode = "sequence_decode_ms";
        public const string CellMatch = "cell_match_ms";
        public const string PostProcess = "postprocess_ms";
        public const string Assemble = "assemble_ms";
    }

    public static async Task<int> RunAsync(BenchmarkOptions options)
    {
        if (!options.DatasetDirectory.Exists)
        {
            Console.Error.WriteLine($"Dataset directory not found: {options.DatasetDirectory.FullName}");
            return 1;
        }

        options.RunsDirectory.Create();
        options.OutputPath.Directory?.Create();
        options.MarkdownReportPath.Directory?.Create();

        var imageFiles = options.DatasetDirectory
            .EnumerateFiles("*.png", SearchOption.TopDirectoryOnly)
            .OrderBy(file => file.Name, StringComparer.Ordinal)
            .ToList();

        if (imageFiles.Count == 0)
        {
            Console.Error.WriteLine($"No PNG files found in {options.DatasetDirectory.FullName}");
            return 1;
        }

        using var bootstrapper = new TableFormerArtifactBootstrapper(options.ArtifactCacheDirectory);
        var bootstrapResult = await bootstrapper.EnsureArtifactsAsync().ConfigureAwait(false);
        var initializationSnapshot = await bootstrapResult.InitializePredictorAsync().ConfigureAwait(false);

        using var neuralModel = new TableFormerTorchSharpSdk.Model.TableFormerNeuralModel(
            bootstrapResult.ConfigSnapshot,
            initializationSnapshot,
            bootstrapResult.ModelDirectory);

        var actualTorchThreads = TorchSharp.torch.get_num_threads();
        var actualTorchInteropThreads = TorchSharp.torch.get_num_interop_threads();

        if (actualTorchThreads != options.RequestedThreads)
        {
            Console.WriteLine(
                $"TorchSharp configured {actualTorchThreads} thread(s) despite request for {options.RequestedThreads}. " +
                "Use environment variables to control TorchSharp threading if needed.");
        }

        var decoder = new TableFormerSequenceDecoder(initializationSnapshot);
        var cellMatcher = new TableFormerCellMatcher(bootstrapResult.ConfigSnapshot);
        var cropper = new TableFormerTableCropper();
        var tensorizer = TableFormerImageTensorizer.FromConfig(bootstrapResult.ConfigSnapshot);
        var preparer = new TableFormerPageInputPreparer();
        var postProcessor = new TableFormerMatchingPostProcessor();
        var assembler = new TableFormerDoclingResponseAssembler();

        var gitCommit = ResolveGitCommit(options.RepositoryRoot);
        var dotnetVersion = Environment.Version.ToString();
        var torchSharpVersion = typeof(TorchSharp.torch).Assembly.GetName().Version?.ToString() ?? "unknown";

        BenchmarkReport? lastReport = null;
        BenchmarkReport? bestReport = null;
        FileInfo? bestRunFile = null;
        double bestAverage = double.MaxValue;

        for (var iteration = 1; iteration <= options.Iterations; iteration++)
        {
            var iterationTimestamp = DateTimeOffset.UtcNow;
            var iterationLabel = options.Iterations > 1
                ? $"{options.Label}_iter{iteration:D2}"
                : options.Label;

            var report = RunIteration(
                imageFiles,
                neuralModel,
                decoder,
                cellMatcher,
                cropper,
                tensorizer,
                preparer,
                postProcessor,
                assembler,
                options.DatasetDirectory.FullName,
                iterationLabel,
                iteration,
                iterationTimestamp,
                options.RequestedThreads,
                actualTorchThreads,
                actualTorchInteropThreads,
                dotnetVersion,
                torchSharpVersion,
                gitCommit);

            var runFile = new FileInfo(Path.Combine(
                options.RunsDirectory.FullName,
                $"{iterationTimestamp:yyyyMMddHHmmssfff}_{iterationLabel}.json"));

            await report.SaveAsync(runFile, JsonOptions).ConfigureAwait(false);

            Console.WriteLine(
                $"[{iterationLabel}] total {report.Summary.TotalMs:F2} ms, " +
                $"average {report.Summary.AverageMs:F2} ms over {report.Summary.NumDocuments} documents.");
            Console.WriteLine($"Saved iteration report to {runFile.FullName}");

            lastReport = report;

            if (report.Summary.AverageMs < bestAverage)
            {
                bestAverage = report.Summary.AverageMs;
                bestReport = report;
                bestRunFile = runFile;
            }
        }

        if (bestReport is null)
        {
            Console.Error.WriteLine("No benchmark iterations executed.");
            return 1;
        }

        var selectedReport = bestReport;
        await WritePredictionsAsync(selectedReport, options.OutputPath).ConfigureAwait(false);
        Console.WriteLine($"Selected iteration '{selectedReport.Metadata.Label}' copied to {options.OutputPath.FullName}");
        if (bestRunFile is not null)
        {
            Console.WriteLine($"Best iteration report: {bestRunFile.FullName}");
        }

        if (!options.SkipReferenceCheck && options.ReferencePath.Exists)
        {
            var matches = await VerifyReferenceAsync(selectedReport, options.ReferencePath).ConfigureAwait(false);
            if (!matches)
            {
                Console.Error.WriteLine(
                    "TorchSharp benchmark output diverges from canonical reference. " +
                    "Use --skip-reference-check to collect timings anyway.");
                return 2;
            }
        }

        if (options.BaselinePath is not null && options.BaselinePath.Exists)
        {
            var baselineReport = await BenchmarkReport.LoadAsync(options.BaselinePath, JsonOptions).ConfigureAwait(false);
            var comparison = BenchmarkComparison.Create(baselineReport, selectedReport);
            await File.WriteAllTextAsync(options.MarkdownReportPath.FullName, comparison.ToMarkdown()).ConfigureAwait(false);
            Console.WriteLine($"Performance comparison written to {options.MarkdownReportPath.FullName}");
        }

        return 0;
    }

    private static BenchmarkReport RunIteration(
        IReadOnlyList<FileInfo> imageFiles,
        TableFormerTorchSharpSdk.Model.TableFormerNeuralModel neuralModel,
        TableFormerSequenceDecoder decoder,
        TableFormerCellMatcher cellMatcher,
        TableFormerTableCropper cropper,
        TableFormerImageTensorizer tensorizer,
        TableFormerPageInputPreparer preparer,
        TableFormerMatchingPostProcessor postProcessor,
        TableFormerDoclingResponseAssembler assembler,
        string datasetPath,
        string label,
        int iteration,
        DateTimeOffset timestamp,
        int requestedThreads,
        int actualTorchThreads,
        int actualTorchInteropThreads,
        string dotnetVersion,
        string torchSharpVersion,
        string gitCommit)
    {
        var documents = new Dictionary<string, BenchmarkDocumentResult>(StringComparer.Ordinal);
        var stageSummary = new Dictionary<string, double>(StringComparer.Ordinal);

        foreach (var imageFile in imageFiles)
        {
            var docStopwatch = Stopwatch.StartNew();
            var stageCollector = new StageTimingCollector();

            stageCollector.Restart();
            var decodedImage = TableFormerDecodedPageImage.Decode(imageFile);
            stageCollector.StopAndRecord(StageNames.DecodeImage);

            stageCollector.Restart();
            var pageSnapshot = preparer.PreparePageInput(decodedImage);
            stageCollector.StopAndRecord(StageNames.PreparePage);

            stageCollector.Restart();
            var cropSnapshot = cropper.PrepareTableCrops(decodedImage, pageSnapshot.TableBoundingBoxes);
            stageCollector.StopAndRecord(StageNames.CropTables);

            var tables = new List<Dictionary<string, object?>>(cropSnapshot.TableCrops.Count);

            for (var tableIndex = 0; tableIndex < cropSnapshot.TableCrops.Count; tableIndex++)
            {
                var crop = cropSnapshot.TableCrops[tableIndex];

                stageCollector.Restart();
                using var tensorSnapshot = tensorizer.CreateTensor(crop);
                stageCollector.StopAndRecord(StageNames.Tensorize);

                stageCollector.Restart();
                using var prediction = neuralModel.Predict(tensorSnapshot.Tensor);
                stageCollector.StopAndRecord(StageNames.ModelInference);

                stageCollector.Restart();
                var decoded = decoder.Decode(prediction);
                stageCollector.StopAndRecord(StageNames.SequenceDecode);

                stageCollector.Restart();
                var matchingResult = cellMatcher.MatchCells(pageSnapshot, crop, decoded);
                var matchingDetails = matchingResult.ToMatchingDetails();
                stageCollector.StopAndRecord(StageNames.CellMatch);

                stageCollector.Restart();
                var processed = pageSnapshot.Tokens.Count > 0
                    ? postProcessor.Process(matchingDetails.ToMutable(), correctOverlappingCells: false)
                    : matchingDetails;
                stageCollector.StopAndRecord(StageNames.PostProcess);

                stageCollector.Restart();
                var assembled = assembler.Assemble(processed, decoded, sortRowColIndexes: true);
                stageCollector.StopAndRecord(StageNames.Assemble);

                tables.Add(assembled.ToDictionary());
            }

            docStopwatch.Stop();

            var predictionPayload = new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["num_tables"] = tables.Count,
                ["tables"] = tables,
            };

            var documentResult = new BenchmarkDocumentResult
            {
                TimingMs = docStopwatch.Elapsed.TotalMilliseconds,
                StageTimingsMs = stageCollector.ToDictionary(),
                Prediction = predictionPayload,
            };

            documents[imageFile.Name] = documentResult;

            foreach (var stage in documentResult.StageTimingsMs)
            {
                if (stageSummary.TryGetValue(stage.Key, out var total))
                {
                    stageSummary[stage.Key] = total + stage.Value;
                }
                else
                {
                    stageSummary[stage.Key] = stage.Value;
                }
            }
        }

        var totalMs = documents.Values.Sum(doc => doc.TimingMs);
        var averageMs = totalMs / Math.Max(1, documents.Count);

        var metadata = new BenchmarkMetadata
        {
            Dataset = datasetPath,
            Label = label,
            Iteration = iteration,
            Timestamp = timestamp,
            RequestedThreadCount = requestedThreads,
            TorchSharpThreadCount = actualTorchThreads,
            TorchSharpInteropThreadCount = actualTorchInteropThreads,
            DotnetVersion = dotnetVersion,
            TorchSharpVersion = torchSharpVersion,
            GitCommit = gitCommit,
        };

        var summary = new BenchmarkSummary
        {
            NumDocuments = documents.Count,
            TotalMs = totalMs,
            AverageMs = averageMs,
        };

        return new BenchmarkReport
        {
            Metadata = metadata,
            Documents = documents,
            Summary = summary,
            StageSummaryMs = stageSummary,
        };
    }

    private static async Task<bool> VerifyReferenceAsync(BenchmarkReport report, FileInfo referencePath)
    {
        var referenceJson = await File.ReadAllTextAsync(referencePath.FullName).ConfigureAwait(false);
        using var referenceDocument = JsonDocument.Parse(referenceJson);
        var referenceTables = ExtractCanonicalTables(referenceDocument.RootElement);
        var actualTables = ExtractCanonicalTables(report);

        if (referenceTables.Count != actualTables.Count)
        {
            return false;
        }

        foreach (var pair in referenceTables)
        {
            if (!actualTables.TryGetValue(pair.Key, out var actualValue))
            {
                return false;
            }

            if (!string.Equals(pair.Value, actualValue, StringComparison.Ordinal))
            {
                return false;
            }
        }

        return true;
    }

    private static string ResolveGitCommit(string repositoryRoot)
    {
        try
        {
            var startInfo = new ProcessStartInfo("git", "rev-parse HEAD")
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
                WorkingDirectory = repositoryRoot,
            };

            using var process = Process.Start(startInfo);
            if (process is null)
            {
                return "unknown";
            }

            var output = process.StandardOutput.ReadToEnd().Trim();
            process.WaitForExit();
            return string.IsNullOrWhiteSpace(output) ? "unknown" : output;
        }
        catch
        {
            return "unknown";
        }
    }

    private sealed class StageTimingCollector
    {
        private readonly Dictionary<string, double> _timings = new(StringComparer.Ordinal);
        private readonly Stopwatch _stopwatch = new();

        public void Restart() => _stopwatch.Restart();

        public void StopAndRecord(string stageName)
        {
            _stopwatch.Stop();
            var elapsed = _stopwatch.Elapsed.TotalMilliseconds;

            if (_timings.TryGetValue(stageName, out var total))
            {
                _timings[stageName] = total + elapsed;
            }
            else
            {
                _timings[stageName] = elapsed;
            }
        }

        public Dictionary<string, double> ToDictionary() => new(_timings, StringComparer.Ordinal);
    }

    private static Dictionary<(string ImageName, int TableIndex), string> ExtractCanonicalTables(JsonElement root)
    {
        var result = new Dictionary<(string, int), string>();

        foreach (var property in root.EnumerateObject())
        {
            var imageName = property.Name;
            if (!property.Value.TryGetProperty("tables", out var tablesElement))
            {
                continue;
            }

            var index = 0;
            foreach (var tableElement in tablesElement.EnumerateArray())
            {
                string canonical = tableElement.ValueKind == JsonValueKind.String
                    ? tableElement.GetString() ?? string.Empty
                    : JsonCanonicalizer.GetCanonicalJson(tableElement);

                result[(imageName, index)] = canonical;
                index++;
            }
        }

        return result;
    }

    private static Dictionary<(string ImageName, int TableIndex), string> ExtractCanonicalTables(BenchmarkReport report)
    {
        var result = new Dictionary<(string, int), string>();

        foreach (var document in report.Documents)
        {
            if (!document.Value.Prediction.TryGetValue("tables", out var tablesObj))
            {
                continue;
            }

            var tablesEnumerable = tablesObj as IEnumerable<object?> ?? Array.Empty<object?>();
            var index = 0;

            foreach (var table in tablesEnumerable)
            {
                string canonical = table switch
                {
                    null => string.Empty,
                    string s => s,
                    JsonElement element => JsonCanonicalizer.GetCanonicalJson(element),
                    IDictionary<string, object?> dictionary => CanonicalizeDictionary(dictionary),
                    _ => CanonicalizeUnknown(table),
                };

                result[(document.Key, index)] = canonical;
                index++;
            }
        }

        return result;
    }

    private static string CanonicalizeDictionary(IDictionary<string, object?> dictionary)
    {
        var json = JsonSerializer.Serialize(dictionary, JsonOptions);
        using var document = JsonDocument.Parse(json);
        return JsonCanonicalizer.GetCanonicalJson(document.RootElement);
    }

    private static string CanonicalizeUnknown(object value)
    {
        var json = JsonSerializer.Serialize(value, JsonOptions);
        using var document = JsonDocument.Parse(json);
        return JsonCanonicalizer.GetCanonicalJson(document.RootElement);
    }

    private static async Task WritePredictionsAsync(BenchmarkReport report, FileInfo outputPath)
    {
        var predictions = report.ExtractPredictions();
        var json = JsonSerializer.Serialize(predictions, JsonOptions);
        outputPath.Directory?.Create();
        await File.WriteAllTextAsync(outputPath.FullName, json).ConfigureAwait(false);
    }
}

internal sealed class BenchmarkReport
{
    public BenchmarkMetadata Metadata { get; set; } = new();

    public Dictionary<string, BenchmarkDocumentResult> Documents { get; set; } = new(StringComparer.Ordinal);

    public BenchmarkSummary Summary { get; set; } = new();

    public Dictionary<string, double> StageSummaryMs { get; set; } = new(StringComparer.Ordinal);

    public Dictionary<string, Dictionary<string, object?>> ExtractPredictions()
    {
        return Documents.ToDictionary(
            pair => pair.Key,
            pair => new Dictionary<string, object?>(pair.Value.Prediction, StringComparer.Ordinal),
            StringComparer.Ordinal);
    }

    public async Task SaveAsync(FileInfo file, JsonSerializerOptions options)
    {
        file.Directory?.Create();
        var json = JsonSerializer.Serialize(this, options);
        await File.WriteAllTextAsync(file.FullName, json).ConfigureAwait(false);
    }

    public static async Task<BenchmarkReport> LoadAsync(FileInfo file, JsonSerializerOptions options)
    {
        var json = await File.ReadAllTextAsync(file.FullName).ConfigureAwait(false);
        using var document = JsonDocument.Parse(json);

        if (document.RootElement.TryGetProperty("Documents", out _))
        {
            var report = JsonSerializer.Deserialize<BenchmarkReport>(json, options)
                ?? throw new InvalidDataException($"File '{file.FullName}' does not contain a benchmark report.");

            report.Normalize();
            return report;
        }

        if (document.RootElement.TryGetProperty("predictions", out var predictionsElement))
        {
            var legacy = LoadLegacyBenchmark(file, document.RootElement, predictionsElement);
            legacy.Normalize();
            return legacy;
        }

        throw new InvalidDataException($"File '{file.FullName}' does not contain a recognized benchmark report format.");
    }

    private void Normalize()
    {
        Documents = Documents.ToDictionary(
            pair => pair.Key,
            pair =>
            {
                pair.Value.StageTimingsMs = new Dictionary<string, double>(pair.Value.StageTimingsMs, StringComparer.Ordinal);
                pair.Value.Prediction = new Dictionary<string, object?>(pair.Value.Prediction, StringComparer.Ordinal);
                return pair.Value;
            },
            StringComparer.Ordinal);

        StageSummaryMs = new Dictionary<string, double>(StageSummaryMs, StringComparer.Ordinal);
    }

    private static BenchmarkReport LoadLegacyBenchmark(FileInfo file, JsonElement root, JsonElement predictionsElement)
    {
        var documents = new Dictionary<string, BenchmarkDocumentResult>(StringComparer.Ordinal);

        foreach (var predictionProperty in predictionsElement.EnumerateObject())
        {
            var imageName = predictionProperty.Name;
            var prediction = predictionProperty.Value;

            var tables = new List<object?>(prediction.GetProperty("tables").GetArrayLength());
            foreach (var tableElement in prediction.GetProperty("tables").EnumerateArray())
            {
                if (tableElement.ValueKind == JsonValueKind.String)
                {
                    tables.Add(tableElement.GetString());
                }
                else
                {
                    tables.Add(tableElement.Clone());
                }
            }

            var predictionDictionary = new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["num_tables"] = prediction.GetProperty("num_tables").GetInt32(),
                ["tables"] = tables,
            };

            double timingMs = 0.0;
            if (root.TryGetProperty("timings_ms", out var timingsElement) &&
                timingsElement.TryGetProperty(imageName, out var timingValue))
            {
                timingMs = timingValue.GetDouble();
            }

            documents[imageName] = new BenchmarkDocumentResult
            {
                TimingMs = timingMs,
                StageTimingsMs = new Dictionary<string, double>(StringComparer.Ordinal),
                Prediction = predictionDictionary,
            };
        }

        double totalMs = documents.Values.Sum(doc => doc.TimingMs);
        double averageMs = documents.Count > 0 ? totalMs / documents.Count : 0.0;

        if (root.TryGetProperty("summary", out var summaryElement))
        {
            if (summaryElement.TryGetProperty("total_ms", out var totalMsElement))
            {
                totalMs = totalMsElement.GetDouble();
            }

            if (summaryElement.TryGetProperty("average_ms", out var averageElement))
            {
                averageMs = averageElement.GetDouble();
            }
        }

        var metadata = new BenchmarkMetadata
        {
            Dataset = string.Empty,
            Label = Path.GetFileNameWithoutExtension(file.Name),
            Iteration = 0,
            Timestamp = DateTimeOffset.MinValue,
            RequestedThreadCount = 0,
            TorchSharpThreadCount = 0,
            TorchSharpInteropThreadCount = 0,
            DotnetVersion = string.Empty,
            TorchSharpVersion = string.Empty,
            GitCommit = string.Empty,
        };

        var summary = new BenchmarkSummary
        {
            NumDocuments = documents.Count,
            TotalMs = totalMs,
            AverageMs = averageMs,
        };

        return new BenchmarkReport
        {
            Metadata = metadata,
            Documents = documents,
            Summary = summary,
            StageSummaryMs = new Dictionary<string, double>(StringComparer.Ordinal),
        };
    }
}

internal sealed class BenchmarkMetadata
{
    public string Dataset { get; set; } = string.Empty;

    public string Label { get; set; } = string.Empty;

    public int Iteration { get; set; }

    public DateTimeOffset Timestamp { get; set; }

    public int RequestedThreadCount { get; set; }

    public int TorchSharpThreadCount { get; set; }

    public int TorchSharpInteropThreadCount { get; set; }

    public string DotnetVersion { get; set; } = string.Empty;

    public string TorchSharpVersion { get; set; } = string.Empty;

    public string GitCommit { get; set; } = string.Empty;
}

internal sealed class BenchmarkDocumentResult
{
    public double TimingMs { get; set; }

    public Dictionary<string, double> StageTimingsMs { get; set; } = new(StringComparer.Ordinal);

    public Dictionary<string, object?> Prediction { get; set; } = new(StringComparer.Ordinal);
}

internal sealed class BenchmarkSummary
{
    public int NumDocuments { get; set; }

    public double TotalMs { get; set; }

    public double AverageMs { get; set; }
}

internal sealed class BenchmarkComparison
{
    private BenchmarkComparison(BenchmarkReport baseline, BenchmarkReport current)
    {
        Baseline = baseline;
        Current = current;
    }

    public BenchmarkReport Baseline { get; }

    public BenchmarkReport Current { get; }

    public static BenchmarkComparison Create(BenchmarkReport baseline, BenchmarkReport current)
        => new(baseline, current);

    public string ToMarkdown()
    {
        var builder = new StringBuilder();
        builder.AppendLine("# TableFormer TorchSharp Performance Report");
        builder.AppendLine();
        builder.AppendLine("## Iterations");
        builder.AppendLine();
        builder.AppendLine($"- Baseline: `{Baseline.Metadata.Label}` (iteration {Baseline.Metadata.Iteration}) captured {Baseline.Metadata.Timestamp:O}");
        builder.AppendLine($"- Current: `{Current.Metadata.Label}` (iteration {Current.Metadata.Iteration}) captured {Current.Metadata.Timestamp:O}");
        builder.AppendLine($"- Dataset: `{Current.Metadata.Dataset}`");
        builder.AppendLine();

        builder.AppendLine("## Summary");
        builder.AppendLine();
        builder.AppendLine("| Metric | Baseline (ms) | Current (ms) | Delta (ms) | Delta (%) |");
        builder.AppendLine("| --- | ---: | ---: | ---: | ---: |");
        builder.AppendLine(FormatSummaryRow("Total", Baseline.Summary.TotalMs, Current.Summary.TotalMs));
        builder.AppendLine(FormatSummaryRow("Average per document", Baseline.Summary.AverageMs, Current.Summary.AverageMs));
        builder.AppendLine();

        builder.AppendLine("## Stage Breakdown");
        builder.AppendLine();
        builder.AppendLine("| Stage | Baseline (ms) | Current (ms) | Delta (ms) | Delta (%) | Baseline Share (%) | Current Share (%) |");
        builder.AppendLine("| --- | ---: | ---: | ---: | ---: | ---: | ---: |");

        var totalBaselineStage = Math.Max(1e-9, Baseline.StageSummaryMs.Values.Sum());
        var totalCurrentStage = Math.Max(1e-9, Current.StageSummaryMs.Values.Sum());
        foreach (var stage in Baseline.StageSummaryMs.Keys
                     .Union(Current.StageSummaryMs.Keys)
                     .OrderBy(stage => stage, StringComparer.Ordinal))
        {
            var baselineValue = Baseline.StageSummaryMs.TryGetValue(stage, out var b) ? b : 0.0;
            var currentValue = Current.StageSummaryMs.TryGetValue(stage, out var c) ? c : 0.0;
            var baselineShare = baselineValue / totalBaselineStage * 100.0;
            var currentShare = currentValue / totalCurrentStage * 100.0;
            builder.AppendLine(FormatStageRow(stage, baselineValue, currentValue, baselineShare, currentShare));
        }

        builder.AppendLine();
        builder.AppendLine("## Per-document Timings");
        builder.AppendLine();
        builder.AppendLine("| Document | Baseline (ms) | Current (ms) | Delta (ms) | Delta (%) |");
        builder.AppendLine("| --- | ---: | ---: | ---: | ---: |");

        foreach (var document in Baseline.Documents.Keys
                     .Union(Current.Documents.Keys)
                     .OrderBy(doc => doc, StringComparer.Ordinal))
        {
            var baselineValue = Baseline.Documents.TryGetValue(document, out var b) ? b.TimingMs : 0.0;
            var currentValue = Current.Documents.TryGetValue(document, out var c) ? c.TimingMs : 0.0;
            builder.AppendLine(FormatSummaryRow(document, baselineValue, currentValue));
        }

        builder.AppendLine();
        builder.AppendLine("## Environment");
        builder.AppendLine();
        builder.AppendLine("| Setting | Baseline | Current |");
        builder.AppendLine("| --- | --- | --- |");
        builder.AppendLine($"| Requested threads | {Baseline.Metadata.RequestedThreadCount} | {Current.Metadata.RequestedThreadCount} |");
        builder.AppendLine($"| TorchSharp threads | {Baseline.Metadata.TorchSharpThreadCount} | {Current.Metadata.TorchSharpThreadCount} |");
        builder.AppendLine($"| TorchSharp interop threads | {Baseline.Metadata.TorchSharpInteropThreadCount} | {Current.Metadata.TorchSharpInteropThreadCount} |");
        builder.AppendLine($"| .NET version | {Baseline.Metadata.DotnetVersion} | {Current.Metadata.DotnetVersion} |");
        builder.AppendLine($"| TorchSharp version | {Baseline.Metadata.TorchSharpVersion} | {Current.Metadata.TorchSharpVersion} |");
        builder.AppendLine($"| Git commit | {Baseline.Metadata.GitCommit} | {Current.Metadata.GitCommit} |");

        return builder.ToString();
    }

    private static string FormatSummaryRow(string label, double baseline, double current)
    {
        var delta = current - baseline;
        var percent = PercentChange(baseline, current);
        return FormattableString.Invariant($"| {label} | {baseline:F2} | {current:F2} | {FormatSigned(delta)} | {FormatSigned(percent)} |");
    }

    private static string FormatStageRow(string stage, double baseline, double current, double baselineShare, double currentShare)
    {
        var delta = current - baseline;
        var percent = PercentChange(baseline, current);
        return FormattableString.Invariant(
            $"| {stage} | {baseline:F2} | {current:F2} | {FormatSigned(delta)} | {FormatSigned(percent)} | {baselineShare:F2} | {currentShare:F2} |");
    }

    private static string FormatSigned(double value)
    {
        if (Math.Abs(value) < 1e-6)
        {
            return "0.00";
        }

        var formatted = value.ToString("F2", CultureInfo.InvariantCulture);
        return value >= 0 ? $"+{formatted}" : formatted;
    }

    private static double PercentChange(double baseline, double current)
    {
        if (Math.Abs(baseline) < 1e-9)
        {
            return 0.0;
        }

        return (current - baseline) / baseline * 100.0;
    }
}
