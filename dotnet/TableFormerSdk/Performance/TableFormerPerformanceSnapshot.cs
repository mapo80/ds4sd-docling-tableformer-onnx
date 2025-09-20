using System;
using System.Globalization;
using TableFormerSdk.Enums;

namespace TableFormerSdk.Performance;

public sealed record TableFormerPerformanceSnapshot(
    TableFormerRuntime Runtime,
    TableFormerModelVariant Variant,
    int WindowSampleCount,
    int TotalSampleCount,
    double AverageLatencyMilliseconds,
    double BestLatencyMilliseconds,
    double LastLatencyMilliseconds)
{
    public override string ToString() => string.Format(
        CultureInfo.InvariantCulture,
        "{0}/{1}: n={2}, avg={3:F3} ms, best={4:F3} ms",
        Runtime,
        Variant,
        WindowSampleCount,
        AverageLatencyMilliseconds,
        BestLatencyMilliseconds);

    public TimeSpan AverageLatency => TimeSpan.FromMilliseconds(AverageLatencyMilliseconds);

    public TimeSpan BestLatency => TimeSpan.FromMilliseconds(BestLatencyMilliseconds);

    public TimeSpan LastLatency => TimeSpan.FromMilliseconds(LastLatencyMilliseconds);
}
