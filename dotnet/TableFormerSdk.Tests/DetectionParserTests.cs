using System;
using TableFormerSdk.Backends;
using Xunit;

namespace TableFormerSdk.Tests;

public class DetectionParserTests
{
    [Fact]
    public void Parse_InvalidShapes_ReturnsEmpty()
    {
        var result = TableFormerDetectionParser.Parse(Array.Empty<float>(), new long[] { 0 }, Array.Empty<float>(), new long[] { 0 }, 10, 10);
        Assert.Empty(result);
    }

    [Fact]
    public void Parse_UnsupportedBatch_Throws()
    {
        var logits = new float[6];
        var boxes = new float[8];
        Assert.Throws<NotSupportedException>(() => TableFormerDetectionParser.Parse(logits, new long[] { 2, 1, 3 }, boxes, new long[] { 1, 2, 4 }, 10, 10));
    }

    [Fact]
    public void Parse_SkipsLowConfidenceDetections()
    {
        var logits = new[] { -10f, -9f, -8f };
        var boxes = new[] { 0.5f, 0.5f, 0.2f, 0.2f };
        var result = TableFormerDetectionParser.Parse(logits, new long[] { 1, 1, 3 }, boxes, new long[] { 1, 1, 4 }, 10, 10);
        Assert.Empty(result);
    }

    [Fact]
    public void Parse_StopsWhenBoxesAreIncomplete()
    {
        var logits = new[] { 5f, -5f };
        var boxes = new[] { 0.5f, 0.5f, 0.5f };
        var result = TableFormerDetectionParser.Parse(logits, new long[] { 1, 1, 2 }, boxes, new long[] { 1, 1, 3 }, 10, 10);
        Assert.Empty(result);
    }

    [Fact]
    public void Parse_ReturnsClampedRegions()
    {
        var logits = new[] { -5f, 5f };
        var boxes = new[] { 1.5f, 1.5f, 2f, 2f };
        var result = TableFormerDetectionParser.Parse(logits, new long[] { 1, 1, 2 }, boxes, new long[] { 1, 1, 4 }, 10, 20);
        var region = Assert.Single(result);
        Assert.Equal(5, region.X);
        Assert.Equal(10, region.Y);
        Assert.Equal(5, region.Width);
        Assert.Equal(10, region.Height);
        Assert.Equal("class_1", region.Label);
    }
}
