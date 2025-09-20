using TableFormerSdk.Enums;

namespace TableFormerSdk.Models;

public interface ITableFormerModelCatalog
{
    bool SupportsRuntime(TableFormerRuntime runtime);

    bool SupportsVariant(TableFormerRuntime runtime, TableFormerModelVariant variant);

    TableFormerModelArtifact GetArtifact(TableFormerRuntime runtime, TableFormerModelVariant variant);
}
