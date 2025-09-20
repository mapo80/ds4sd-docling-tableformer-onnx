using System;
using TableFormerSdk.Backends;
using TableFormerSdk.Enums;
using TableFormerSdk.Performance;
using Xunit;

namespace TableFormerSdk.Tests;

public class PerformanceTests
{
    [Fact]
    public void TableFormerPerformanceOptions_ValidateArguments()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new TableFormerPerformanceOptions(minimumSamples: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TableFormerPerformanceOptions(slidingWindowSize: 0));

        var options = new TableFormerPerformanceOptions(runtimePriority: new[]
        {
            TableFormerRuntime.Auto,
            TableFormerRuntime.OpenVino,
            TableFormerRuntime.OpenVino,
            TableFormerRuntime.Onnx
        });

        Assert.True(options.EnableAdaptiveRuntimeSelection);
        Assert.Equal(3, options.MinimumSamples);
        Assert.Equal(20, options.SlidingWindowSize);
        Assert.Equal(TableFormerRuntime.Onnx, options.DefaultRuntime);
        Assert.Equal(new[] { TableFormerRuntime.OpenVino, TableFormerRuntime.Onnx }, options.RuntimePriority);
    }

    [Fact]
    public void TimeWindowStatistics_ComputesSnapshots()
    {
        var key = new BackendKey(TableFormerRuntime.Onnx, TableFormerModelVariant.Fast);
        var stats = new TimeWindowStatistics(3);

        var empty = stats.Snapshot(key);
        Assert.Equal(double.PositiveInfinity, empty.AverageLatencyMilliseconds);
        Assert.Equal(0, empty.TotalSampleCount);

        stats.Register(key, TimeSpan.FromMilliseconds(10));
        stats.Register(key, TimeSpan.FromMilliseconds(20));
        stats.Register(key, TimeSpan.FromMilliseconds(30));
        stats.Register(key, TimeSpan.FromMilliseconds(40));

        Assert.Equal(3, stats.WindowSampleCount);
        Assert.Equal(4, stats.TotalSampleCount);

        var snapshot = stats.Snapshot(key);
        Assert.Equal(3, snapshot.WindowSampleCount);
        Assert.Equal(4, snapshot.TotalSampleCount);
        Assert.Equal(30, snapshot.AverageLatencyMilliseconds, precision: 1);
        Assert.Equal(20, snapshot.BestLatencyMilliseconds, precision: 1);
        Assert.Equal(40, snapshot.LastLatencyMilliseconds, precision: 1);
    }

    [Fact]
    public void PerformanceMonitor_TracksSnapshotsAndSelection()
    {
        var monitor = new PerformanceMonitor(windowSize: 3);
        var fastOnnx = new BackendKey(TableFormerRuntime.Onnx, TableFormerModelVariant.Fast);
        var fastOpenVino = new BackendKey(TableFormerRuntime.OpenVino, TableFormerModelVariant.Fast);

        monitor.Record(fastOpenVino, TimeSpan.FromMilliseconds(15));
        monitor.Record(fastOpenVino, TimeSpan.FromMilliseconds(5));
        monitor.Record(fastOnnx, TimeSpan.FromMilliseconds(30));

        var snapshots = monitor.GetSnapshots(TableFormerModelVariant.Fast);
        Assert.Equal(2, snapshots.Count);
        Assert.Equal(TableFormerRuntime.OpenVino, snapshots[0].Runtime);
        Assert.True(snapshots[0].AverageLatencyMilliseconds <= snapshots[1].AverageLatencyMilliseconds);

        var pending = monitor.FindRuntimeNeedingSamples(TableFormerModelVariant.Fast, new[] { TableFormerRuntime.Onnx, TableFormerRuntime.OpenVino }, minimumSamples: 2);
        Assert.Equal(TableFormerRuntime.Onnx, pending?.Runtime);

        var best = monitor.GetBestSnapshot(TableFormerModelVariant.Fast, new[] { TableFormerRuntime.Onnx, TableFormerRuntime.OpenVino }, minimumSamples: 2);
        Assert.Equal(TableFormerRuntime.OpenVino, best?.Runtime);

        var openVinoSnapshot = monitor.TryGetSnapshot(fastOpenVino);
        Assert.NotNull(openVinoSnapshot);
        Assert.Equal(2, openVinoSnapshot.WindowSampleCount);
    }

    [Fact]
    public void TableFormerPerformanceAdvisor_AdaptiveSelection()
    {
        var options = new TableFormerPerformanceOptions(
            enableAdaptiveRuntimeSelection: true,
            minimumSamples: 1,
            slidingWindowSize: 5,
            defaultRuntime: TableFormerRuntime.Onnx,
            runtimePriority: new[] { TableFormerRuntime.OpenVino, TableFormerRuntime.Onnx });

        var advisor = new TableFormerPerformanceAdvisor(options, new[] { TableFormerRuntime.Onnx, TableFormerRuntime.OpenVino });
        var variant = TableFormerModelVariant.Fast;

        var durations = new Dictionary<TableFormerRuntime, TimeSpan>
        {
            [TableFormerRuntime.Onnx] = TimeSpan.FromMilliseconds(12),
            [TableFormerRuntime.OpenVino] = TimeSpan.FromMilliseconds(5)
        };

        var first = advisor.ResolveBackend(variant, TableFormerRuntime.Auto);
        advisor.Record(first, durations[first.Runtime]);

        var second = advisor.ResolveBackend(variant, TableFormerRuntime.Auto);
        Assert.NotEqual(first.Runtime, second.Runtime);
        advisor.Record(second, durations[second.Runtime]);

        var third = advisor.ResolveBackend(variant, TableFormerRuntime.Auto);
        var expectedBest = durations[second.Runtime] < durations[first.Runtime] ? second.Runtime : first.Runtime;
        Assert.Equal(expectedBest, third.Runtime);

        var latest = advisor.TryGetSnapshot(expectedBest, variant);
        Assert.NotNull(latest);
        Assert.Equal(1, latest.WindowSampleCount);

        var snapshots = advisor.GetSnapshots(variant);
        Assert.Equal(2, snapshots.Count);
    }

    [Fact]
    public void TableFormerPerformanceAdvisor_DisabledAdaptiveSelection_UsesRequestedRuntime()
    {
        var options = new TableFormerPerformanceOptions(enableAdaptiveRuntimeSelection: false, defaultRuntime: TableFormerRuntime.OpenVino);
        var advisor = new TableFormerPerformanceAdvisor(options, Array.Empty<TableFormerRuntime>());

        var requested = advisor.ResolveBackend(TableFormerModelVariant.Fast, TableFormerRuntime.Onnx);
        Assert.Equal(TableFormerRuntime.Onnx, requested.Runtime);

        var fallback = advisor.ResolveBackend(TableFormerModelVariant.Fast, TableFormerRuntime.Auto);
        Assert.Equal(TableFormerRuntime.OpenVino, fallback.Runtime);
    }

    [Fact]
    public void ValueStopwatch_MeasuresElapsedTime()
    {
        var stopwatch = ValueStopwatch.StartNew();
        System.Threading.Thread.Sleep(10);
        var elapsed = stopwatch.GetElapsedTime();

        Assert.True(elapsed.TotalMilliseconds >= 0);
    }

    [Fact]
    public void TableFormerPerformanceSnapshot_ToString_IncludesMetrics()
    {
        var snapshot = new TableFormerPerformanceSnapshot(
            TableFormerRuntime.OpenVino,
            TableFormerModelVariant.Accurate,
            5,
            6,
            12.345,
            10.5,
            13.7);

        var text = snapshot.ToString();

        Assert.Contains("OpenVino", text);
        Assert.Contains("Accurate", text);
        Assert.Contains("n=5", text);
        Assert.Contains("avg=12.345", text);
        Assert.Equal(TimeSpan.FromMilliseconds(12.345), snapshot.AverageLatency);
        Assert.Equal(TimeSpan.FromMilliseconds(10.5), snapshot.BestLatency);
        Assert.Equal(TimeSpan.FromMilliseconds(13.7), snapshot.LastLatency);
    }
}
