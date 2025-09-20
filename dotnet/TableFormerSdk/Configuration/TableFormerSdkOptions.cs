using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using TableFormerSdk.Constants;
using TableFormerSdk.Enums;
using TableFormerSdk.Models;
using TableFormerSdk.Performance;

namespace TableFormerSdk.Configuration;

public sealed class TableFormerSdkOptions
{
    private readonly IReadOnlyCollection<TableFormerLanguage> _supportedLanguages;

    public TableFormerSdkOptions(
        ITableFormerModelCatalog? modelCatalog = null,
        TableFormerLanguage defaultLanguage = TableFormerLanguage.English,
        IEnumerable<TableFormerLanguage>? supportedLanguages = null,
        TableVisualizationOptions? visualizationOptions = null,
        TableFormerPerformanceOptions? performanceOptions = null)
    {
        ModelCatalog = modelCatalog ?? ReleaseModelCatalog.CreateDefault();

        _supportedLanguages = BuildSupportedLanguages(defaultLanguage, supportedLanguages);
        DefaultLanguage = defaultLanguage;
        Visualization = visualizationOptions ?? TableVisualizationOptions.CreateDefault();
        Performance = performanceOptions ?? TableFormerPerformanceOptions.CreateDefault();
        AvailableRuntimes = BuildAvailableRuntimes();
    }

    public ITableFormerModelCatalog ModelCatalog { get; }

    public TableFormerLanguage DefaultLanguage { get; }

    public IReadOnlyCollection<TableFormerLanguage> SupportedLanguages => _supportedLanguages;

    public TableVisualizationOptions Visualization { get; }

    public TableFormerPerformanceOptions Performance { get; }

    public IReadOnlyList<TableFormerRuntime> AvailableRuntimes { get; }

    public void EnsureLanguageIsSupported(TableFormerLanguage language)
    {
        if (!_supportedLanguages.Contains(language))
        {
            throw new NotSupportedException(string.Format(TableFormerConstants.UnsupportedLanguageMessage, language));
        }
    }

    private static IReadOnlyCollection<TableFormerLanguage> BuildSupportedLanguages(TableFormerLanguage defaultLanguage, IEnumerable<TableFormerLanguage>? supportedLanguages)
    {
        var languages = supportedLanguages?.Distinct().ToList() ?? new List<TableFormerLanguage> { defaultLanguage };
        if (!languages.Contains(defaultLanguage))
        {
            languages.Insert(0, defaultLanguage);
        }

        return new ReadOnlyCollection<TableFormerLanguage>(languages);
    }

    private IReadOnlyList<TableFormerRuntime> BuildAvailableRuntimes()
    {
        var runtimes = Enum.GetValues<TableFormerRuntime>()
            .Where(runtime => runtime != TableFormerRuntime.Auto && ModelCatalog.SupportsRuntime(runtime))
            .ToList();

        if (runtimes.Count == 0)
        {
            throw new InvalidOperationException("Nessun runtime disponibile: verificare che i modelli siano presenti nel catalogo configurato");
        }

        return new ReadOnlyCollection<TableFormerRuntime>(runtimes);
    }
}
