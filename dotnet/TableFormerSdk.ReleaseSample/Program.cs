using System;
using System.IO;
using TableFormerSdk.Configuration;
using TableFormerSdk.Enums;
using TableFormerSdk.Models;

var options = new TableFormerSdkOptions();
if (options.ModelCatalog is not ReleaseModelCatalog catalog)
{
    Console.WriteLine("The sample requires ReleaseModelCatalog to access packaged artifacts.");
    return;
}

Console.WriteLine("Docling TableFormer SDK packaged models");
Console.WriteLine($"Base directory: {AppContext.BaseDirectory}");
Console.WriteLine();

foreach (var runtime in options.AvailableRuntimes)
{
    Console.WriteLine($"Runtime: {runtime}");
    foreach (TableFormerModelVariant variant in Enum.GetValues<TableFormerModelVariant>())
    {
        if (!catalog.SupportsVariant(runtime, variant))
        {
            Console.WriteLine($"  - {variant}: not available");
            continue;
        }

        var artifact = catalog.GetArtifact(runtime, variant);
        var sizeBytes = new FileInfo(artifact.ModelPath).Length;
        Console.WriteLine($"  - {variant}: {Path.GetFileName(artifact.ModelPath)} ({sizeBytes / (1024 * 1024.0):F1} MB)");
        if (!string.IsNullOrWhiteSpace(artifact.WeightsPath))
        {
            var weightsSize = new FileInfo(artifact.WeightsPath!).Length;
            Console.WriteLine($"      weights: {Path.GetFileName(artifact.WeightsPath!)} ({weightsSize / (1024 * 1024.0):F1} MB)");
        }
    }

    Console.WriteLine();
}

Console.WriteLine("All models are available locally without additional downloads.");
