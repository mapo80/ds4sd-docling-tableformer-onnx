using System;
using System.IO;
using TableFormerSdk.Constants;
using TableFormerSdk.Enums;

namespace TableFormerSdk.Configuration;

public sealed class PipelineModelPaths
{
    public PipelineModelPaths(string encoderPath, string bboxDecoderPath, string decoderPath)
    {
        EncoderPath = ValidateModelPath(encoderPath, nameof(encoderPath));
        BboxDecoderPath = ValidateModelPath(bboxDecoderPath, nameof(bboxDecoderPath));
        DecoderPath = ValidateModelPath(decoderPath, nameof(decoderPath));
    }

    public string EncoderPath { get; }

    public string BboxDecoderPath { get; }

    public string DecoderPath { get; }

    public (string Encoder, string BboxDecoder, string Decoder) ModelPaths =>
        (EncoderPath, BboxDecoderPath, DecoderPath);

    private static string ValidateModelPath(string path, string argumentName)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            throw new ArgumentException("Model path is empty", argumentName);
        }

        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Model file not found: {path}", path);
        }

        return path;
    }
}