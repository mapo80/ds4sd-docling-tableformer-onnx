using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;

using TableFormerTorchSharpSdk.Artifacts;
using TableFormerTorchSharpSdk.Assets;
using TableFormerTorchSharpSdk.Configuration;
using TableFormerTorchSharpSdk.Decoding;
using TableFormerTorchSharpSdk.Matching;
using TableFormerTorchSharpSdk.Model;
using TableFormerTorchSharpSdk.PagePreparation;
using TableFormerTorchSharpSdk.Results;
using TableFormerTorchSharpSdk.Tensorization;
using TableFormerTorchSharpSdk.Utilities;

return await TableFormerCli.RunAsync(args).ConfigureAwait(false);

internal static class TableFormerCli
{
    public static async Task<int> RunAsync(string[] args)
    {
        try
        {
            var options = CliOptions.Parse(args);
            if (!Directory.Exists(options.DatasetDirectory))
            {
                Console.Error.WriteLine($"Dataset directory not found: {options.DatasetDirectory}");
                return 1;
            }

            Directory.CreateDirectory(options.CacheDirectory);

            var repoRoot = options.RepositoryRoot;
            var resultsDirectory = Path.Combine(repoRoot, "results");
            Directory.CreateDirectory(resultsDirectory);

            var variantMetadatas = options.ResolveVariants();
            var failures = new List<string>();

            TableFormerNeuralModel.ConfigureThreading(Environment.ProcessorCount);

            foreach (var metadata in variantMetadatas)
            {
                Console.WriteLine($"=== Variant: {metadata.DisplayName} ===");
                try
                {
                    await VerifyVariantAsync(
                        metadata,
                        options,
                        resultsDirectory).ConfigureAwait(false);

                    Console.WriteLine($"✓ {metadata.DisplayName} successful");
                }
                catch (Exception ex)
                {
                    failures.Add($"{metadata.DisplayName}: {ex.Message}");
                    Console.Error.WriteLine($"✗ {metadata.DisplayName} failed: {ex.Message}");
                }
            }

            if (failures.Count > 0)
            {
                Console.Error.WriteLine("Failures:");
                foreach (var failure in failures)
                {
                    Console.Error.WriteLine($" - {failure}");
                }

                return 1;
            }

            return 0;
        }
        catch (ArgumentException ex)
        {
            Console.Error.WriteLine($"Argument error: {ex.Message}");
            return 1;
        }
    }

    private static async Task VerifyVariantAsync(
        VariantMetadata metadata,
        CliOptions options,
        string resultsDirectory)
    {
        var variantCacheDirectory = Path.Combine(options.CacheDirectory, metadata.CacheSubdirectory);
        using var bootstrapper = new TableFormerArtifactBootstrapper(
            new DirectoryInfo(variantCacheDirectory),
            metadata.Variant);

        var bootstrapResult = await bootstrapper.EnsureArtifactsAsync().ConfigureAwait(false);

        var configReferencePath = Path.Combine(resultsDirectory, metadata.ConfigReferenceFile);
        if (options.UpdateReference)
        {
            WriteConfigReference(
                configReferencePath,
                metadata,
                bootstrapResult);
        }
        else
        {
            if (!File.Exists(configReferencePath))
            {
                throw new FileNotFoundException(
                    $"Missing config reference for {metadata.DisplayName}. Run with --update-reference to create it.",
                    configReferencePath);
            }

            var reference = await TableFormerConfigReference.LoadAsync(configReferencePath).ConfigureAwait(false);
            bootstrapResult.ConfigSnapshot.EnsureMatches(reference);
        }

        var datasetDirectory = new DirectoryInfo(options.DatasetDirectory);
        var datasetImages = datasetDirectory
            .EnumerateFiles("*.png", SearchOption.TopDirectoryOnly)
            .OrderBy(file => file.Name, StringComparer.Ordinal)
            .ToList();

        if (datasetImages.Count == 0)
        {
            throw new InvalidOperationException($"No PNG files found under '{datasetDirectory.FullName}'.");
        }

        var initializationSnapshot = await bootstrapResult.InitializePredictorAsync().ConfigureAwait(false);
        using var neuralModel = new TableFormerNeuralModel(
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

        var pageResults = new List<PageVerificationResult>();
        foreach (var imageFile in datasetImages)
        {
            var pageSnapshot = preparer.PreparePageInput(imageFile);
            var cropSnapshot = cropper.PrepareTableCrops(imageFile, pageSnapshot.TableBoundingBoxes);

            var tableResults = new List<TableVerificationResult>();
            for (var tableIndex = 0; tableIndex < cropSnapshot.TableCrops.Count; tableIndex++)
            {
                var crop = cropSnapshot.TableCrops[tableIndex];
                using var tensorSnapshot = tensorizer.CreateTensor(crop);
                using var prediction = neuralModel.Predict(tensorSnapshot.Tensor);

                var decoded = decoder.Decode(prediction);
                var matching = cellMatcher.MatchCells(pageSnapshot, crop, decoded);
                var matchingDetails = matching.ToMatchingDetails();
                var processed = pageSnapshot.Tokens.Count > 0
                    ? postProcessor.Process(matchingDetails.ToMutable(), correctOverlappingCells: false)
                    : matchingDetails;

                var assembled = assembler.Assemble(processed, decoded, sortRowColIndexes: true);
                var canonical = Canonicalize(assembled.ToDictionary());

                tableResults.Add(new TableVerificationResult(
                    tableIndex,
                    assembled,
                    canonical));
            }

            pageResults.Add(new PageVerificationResult(imageFile.Name, tableResults));
        }

        var docReferencePath = Path.Combine(resultsDirectory, metadata.DoclingReferenceFile);
        if (options.UpdateReference)
        {
            WriteDoclingReference(docReferencePath, pageResults);
            Console.WriteLine($"Updated reference: {docReferencePath}");
        }
        else
        {
            if (!File.Exists(docReferencePath))
            {
                throw new FileNotFoundException(
                    $"Missing Docling reference for {metadata.DisplayName}. Run with --update-reference to create it.",
                    docReferencePath);
            }

            var reference = DoclingReference.Load(docReferencePath);
            CompareAgainstReference(reference, pageResults, metadata.DisplayName);
        }
    }

    private static void CompareAgainstReference(
        DoclingReference reference,
        IReadOnlyList<PageVerificationResult> actualPages,
        string displayName)
    {
        var failures = new List<string>();

        foreach (var page in actualPages)
        {
            if (!reference.Pages.TryGetValue(page.ImageName, out var referencePage))
            {
                failures.Add($"Unexpected page '{page.ImageName}' produced by {displayName}.");
                continue;
            }

            if (referencePage.NumTables != page.Tables.Count)
            {
                failures.Add(
                    $"Table count mismatch for '{page.ImageName}': expected {referencePage.NumTables}, actual {page.Tables.Count}.");
            }

            for (var index = 0; index < page.Tables.Count; index++)
            {
                var table = page.Tables[index];
                if (!reference.Tables.TryGetValue((page.ImageName, table.Index), out var expectedCanonical))
                {
                    failures.Add($"Missing reference for '{page.ImageName}' table #{table.Index}.");
                    continue;
                }

                if (!string.Equals(expectedCanonical, table.CanonicalJson, StringComparison.Ordinal))
                {
                    failures.Add($"Mismatch for '{page.ImageName}' table #{table.Index}.");
                }
            }
        }

        foreach (var referencePage in reference.Pages.Values)
        {
            if (actualPages.All(page => !string.Equals(page.ImageName, referencePage.ImageName, StringComparison.Ordinal)))
            {
                failures.Add($"Reference page '{referencePage.ImageName}' was not produced by {displayName}.");
            }
        }

        if (failures.Count > 0)
        {
            throw new InvalidOperationException(string.Join(Environment.NewLine, failures));
        }
    }

    private static void WriteConfigReference(
        string referencePath,
        VariantMetadata metadata,
        TableFormerBootstrapResult bootstrapResult)
    {
        var configJson = new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["repo_id"] = "ds4sd/docling-models",
            ["revision"] = "main",
            ["variant"] = metadata.VariantName,
            ["canonical_json"] = bootstrapResult.ConfigSnapshot.NormalizedJson,
            ["sha256"] = bootstrapResult.ConfigSnapshot.Sha256Hash,
        };

        var options = new JsonSerializerOptions
        {
            WriteIndented = true
        };

        File.WriteAllText(referencePath, JsonSerializer.Serialize(configJson, options));
        Console.WriteLine($"Updated config reference: {referencePath}");
    }

    private static void WriteDoclingReference(
        string referencePath,
        IReadOnlyList<PageVerificationResult> pages)
    {
        var root = new Dictionary<string, object?>(StringComparer.Ordinal);
        foreach (var page in pages)
        {
            var tables = page.Tables.Select(table => table.Assembled.ToDictionary()).ToList();
            root[page.ImageName] = new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["num_tables"] = page.Tables.Count,
                ["tables"] = tables
            };
        }

        var options = new JsonSerializerOptions
        {
            WriteIndented = true
        };

        var payload = JsonSerializer.Serialize(root, options) + Environment.NewLine;
        File.WriteAllText(referencePath, payload);
    }

    private static string Canonicalize(Dictionary<string, object?> data)
    {
        var json = JsonSerializer.Serialize(data);
        using var document = JsonDocument.Parse(json);
        return JsonCanonicalizer.GetCanonicalJson(document.RootElement);
    }

    private sealed record VariantMetadata(
        TableFormerModelVariant Variant,
        string VariantName,
        string DisplayName,
        string ConfigReferenceFile,
        string DoclingReferenceFile,
        string CacheSubdirectory);

    private sealed record PageVerificationResult(
        string ImageName,
        IReadOnlyList<TableVerificationResult> Tables);

    private sealed record TableVerificationResult(
        int Index,
        TableFormerDoclingTablePrediction Assembled,
        string CanonicalJson);

    private sealed class DoclingReference
    {
        private DoclingReference(
            IReadOnlyDictionary<string, DoclingPage> pages,
            IReadOnlyDictionary<(string ImageName, int TableIndex), string> tables)
        {
            Pages = pages;
            Tables = tables;
        }

        public IReadOnlyDictionary<string, DoclingPage> Pages { get; }

        public IReadOnlyDictionary<(string ImageName, int TableIndex), string> Tables { get; }

        public static DoclingReference Load(string path)
        {
            using var stream = File.OpenRead(path);
            using var document = JsonDocument.Parse(stream);

            var pages = new Dictionary<string, DoclingPage>(StringComparer.Ordinal);
            var tables = new Dictionary<(string ImageName, int TableIndex), string>();

            foreach (var property in document.RootElement.EnumerateObject())
            {
                var imageName = property.Name;
                var pageObject = property.Value;

                var numTables = pageObject.GetProperty("num_tables").GetInt32();
                var tableEntries = new List<string>(numTables);

                var tableArray = pageObject.GetProperty("tables");
                var tableIndex = 0;

                foreach (var tableElement in tableArray.EnumerateArray())
                {
                    string canonical = tableElement.ValueKind switch
                    {
                        JsonValueKind.String => tableElement.GetString() ?? string.Empty,
                        _ => JsonCanonicalizer.GetCanonicalJson(tableElement),
                    };

                    tableEntries.Add(canonical);
                    tables[(imageName, tableIndex)] = canonical;
                    tableIndex += 1;
                }

                pages[imageName] = new DoclingPage(imageName, numTables, tableEntries);
            }

            return new DoclingReference(pages, tables);
        }
    }

    private sealed record DoclingPage(string ImageName, int NumTables, IReadOnlyList<string> Tables);

    private sealed class CliOptions
    {
        private CliOptions(
            IReadOnlyList<TableFormerModelVariant> variants,
            string datasetDirectory,
            string cacheDirectory,
            bool updateReference,
            string repositoryRoot)
        {
            Variants = variants;
            DatasetDirectory = datasetDirectory;
            CacheDirectory = cacheDirectory;
            UpdateReference = updateReference;
            RepositoryRoot = repositoryRoot;
        }

        public IReadOnlyList<TableFormerModelVariant> Variants { get; }

        public string DatasetDirectory { get; }

        public string CacheDirectory { get; }

        public bool UpdateReference { get; }

        public string RepositoryRoot { get; }

        public static CliOptions Parse(string[] args)
        {
            var repoRoot = ResolveRepositoryRoot();
            var variants = new HashSet<TableFormerModelVariant> { TableFormerModelVariant.Fast, TableFormerModelVariant.Accurate };
            var datasetDirectory = Path.Combine(repoRoot, "dataset", "FinTabNet", "benchmark");
            var cacheDirectory = Path.Combine(repoRoot, "dotnet", "artifacts_cli_cache");
            var updateReference = false;

            for (var i = 0; i < args.Length; i++)
            {
                var current = args[i];
                switch (current)
                {
                    case "--variant":
                        i = ExpectValue(args, i);
                        variants.Clear();
                        foreach (var token in args[i].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
                        {
                            variants.Add(ParseVariant(token));
                        }
                        break;
                    case "--dataset":
                        i = ExpectValue(args, i);
                        datasetDirectory = Path.GetFullPath(args[i]);
                        break;
                    case "--cache-dir":
                        i = ExpectValue(args, i);
                        cacheDirectory = Path.GetFullPath(args[i]);
                        break;
                    case "--update-reference":
                        updateReference = true;
                        break;
                    case "--help":
                    case "-h":
                        PrintHelp(repoRoot);
                        Environment.Exit(0);
                        break;
                    default:
                        throw new ArgumentException($"Unknown argument '{current}'. Use --help for usage information.");
                }
            }

            if (variants.Count == 0)
            {
                variants.Add(TableFormerModelVariant.Fast);
                variants.Add(TableFormerModelVariant.Accurate);
            }

            return new CliOptions(
                variants.ToList(),
                datasetDirectory,
                cacheDirectory,
                updateReference,
                repoRoot);
        }

        public IReadOnlyList<VariantMetadata> ResolveVariants()
        {
            var list = new List<VariantMetadata>();
            foreach (var variant in Variants)
            {
                list.Add(variant switch
                {
                    TableFormerModelVariant.Fast => new VariantMetadata(
                        TableFormerModelVariant.Fast,
                        "fast",
                        "TableFormer Fast",
                        ConfigReferenceFile: "tableformer_config_fast_hash.json",
                        DoclingReferenceFile: "tableformer_docling_fintabnet.json",
                        CacheSubdirectory: "fast"),
                    TableFormerModelVariant.Accurate => new VariantMetadata(
                        TableFormerModelVariant.Accurate,
                        "accurate",
                        "TableFormer Accurate",
                        ConfigReferenceFile: "tableformer_config_accurate_hash.json",
                        DoclingReferenceFile: "tableformer_docling_fintabnet_accurate.json",
                        CacheSubdirectory: "accurate"),
                    _ => throw new ArgumentOutOfRangeException(nameof(variant), variant, "Unsupported variant.")
                });
            }

            return list;
        }

        private static int ExpectValue(string[] args, int index)
        {
            if (index + 1 >= args.Length)
            {
                throw new ArgumentException($"Argument '{args[index]}' expects a value.");
            }

            return index + 1;
        }

        private static TableFormerModelVariant ParseVariant(string token)
        {
            return token.ToLowerInvariant() switch
            {
                "fast" => TableFormerModelVariant.Fast,
                "accurate" => TableFormerModelVariant.Accurate,
                _ => throw new ArgumentException($"Unknown variant '{token}'. Allowed values are 'fast', 'accurate'.")
            };
        }

        private static void PrintHelp(string repoRoot)
        {
            Console.WriteLine("TableFormerTorchSharpSdk CLI");
            Console.WriteLine();
            Console.WriteLine("Usage:");
            Console.WriteLine("  dotnet run --project dotnet/TableFormerTorchSharpSdk.Cli -- [options]");
            Console.WriteLine();
            Console.WriteLine("Options:");
            Console.WriteLine("  --variant <fast|accurate|both>   Select the model variant(s) to verify (default: both).");
            Console.WriteLine("  --dataset <path>                 Path to the dataset folder (default: dataset/FinTabNet/benchmark).");
            Console.WriteLine("  --cache-dir <path>               Directory used to cache downloaded models (default: dotnet/artifacts_cli_cache).");
            Console.WriteLine("  --update-reference               Regenerate reference JSON files instead of verifying.");
            Console.WriteLine("  --help                           Display this help message.");
            Console.WriteLine();
            Console.WriteLine($"Repository root: {repoRoot}");
        }

        private static string ResolveRepositoryRoot()
        {
            var current = new DirectoryInfo(AppContext.BaseDirectory);
            while (current is not null)
            {
                if (File.Exists(Path.Combine(current.FullName, "TableFormerSdk.sln")))
                {
                    return current.FullName;
                }

                current = current.Parent;
            }

            throw new InvalidOperationException("Unable to locate repository root containing TableFormerSdk.sln.");
        }
    }
}
