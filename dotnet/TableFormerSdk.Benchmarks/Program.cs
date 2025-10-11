using Microsoft.Extensions.Logging.Abstractions;
using SkiaSharp;
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using TableFormerSdk;
using TableFormerSdk.Configuration;
using TableFormerSdk.Enums;

public class TableFormerPerformanceAnalyzer
{
    public static void Main(string[] args)
    {
        Console.WriteLine("🚀 TABLEFORMER PIPELINE PERFORMANCE ANALYZER");
        Console.WriteLine("===========================================\n");

        var analyzer = new TableFormerPerformanceAnalyzer();
        analyzer.RunAnalysis();
    }

    public void RunAnalysis()
    {
        // Carica immagine di test
        var testImagePath = FindTestImage();
        if (testImagePath == null)
        {
            Console.WriteLine("❌ Test image not found");
            return;
        }

        Console.WriteLine($"📷 Test Image: {testImagePath}");
        using var testImage = SKBitmap.Decode(testImagePath);
        if (testImage == null)
        {
            Console.WriteLine("❌ Failed to decode test image");
            return;
        }

        Console.WriteLine($"📐 Image Size: {testImage.Width}x{testImage.Height}\n");

        // Benchmark 1: Pipeline completa
        var pipelineTime = BenchmarkPipeline(testImagePath);
        Console.WriteLine($"⏱️  Pipeline completa: {pipelineTime:F2}ms");

        // Benchmark 2: Solo preprocessing
        var preprocessingTime = BenchmarkPreprocessing(testImage);
        Console.WriteLine($"⏱️  Solo preprocessing: {preprocessingTime:F2}ms");

        // Benchmark 3: Solo encoder
        var encoderTime = BenchmarkEncoder(testImage);
        Console.WriteLine($"⏱️  Solo encoder: {encoderTime:F2}ms");

        // Benchmark 4: Encoder + BboxDecoder
        var pipelineStepTime = BenchmarkPipelineStep(testImage);
        Console.WriteLine($"⏱️  Encoder + BboxDecoder: {pipelineStepTime:F2}ms");

        // Analisi risultati
        AnalyzeResults(pipelineTime, preprocessingTime, encoderTime, pipelineStepTime);
    }

    private string? FindTestImage()
    {
        var possiblePaths = new[]
        {
            Path.Combine("..", "..", "..", "..", "..", "dataset", "golden", "v0.12.0", "2305.03393v1-pg9", "source", "2305.03393v1-pg9-img.png"),
            Path.Combine("dataset", "golden", "v0.12.0", "2305.03393v1-pg9", "source", "2305.03393v1-pg9-img.png"),
            "dataset/2305.03393v1-pg9-img.png"
        };

        return possiblePaths.FirstOrDefault(File.Exists);
    }

    private double BenchmarkPipeline(string imagePath)
    {
        var encoderPath = Path.Combine("..", "..", "..", "models", "encoder.onnx");
        var bboxDecoderPath = Path.Combine("..", "..", "..", "models", "bbox_decoder.onnx");
        var decoderPath = Path.Combine("..", "..", "..", "models", "decoder.onnx");

        var options = new TableFormerSdkOptions(
            onnx: new TableFormerModelPaths(encoderPath, null),
            pipeline: new PipelineModelPaths(encoderPath, bboxDecoderPath, decoderPath)
        );

        using var sdk = new TableFormerSdk(options);

        var stopwatch = Stopwatch.StartNew();
        var result = sdk.Process(
            imagePath: imagePath,
            overlay: false,
            runtime: TableFormerRuntime.Pipeline,
            variant: TableFormerModelVariant.Fast
        );
        stopwatch.Stop();

        return stopwatch.Elapsed.TotalMilliseconds;
    }

    private double BenchmarkPreprocessing(SKBitmap image)
    {
        var stopwatch = Stopwatch.StartNew();

        // Resize to 448x448 as expected by encoder
        using var resized = image.Resize(new SKImageInfo(448, 448), new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.Linear));

        var data = new float[1, 3, 448, 448];
        for (int y = 0; y < 448; y++)
        {
            for (int x = 0; x < 448; x++)
            {
                var color = resized.GetPixel(x, y);
                data[0, 0, y, x] = color.Red / 255f;
                data[0, 1, y, x] = color.Green / 255f;
                data[0, 2, y, x] = color.Blue / 255f;
            }
        }

        stopwatch.Stop();
        return stopwatch.Elapsed.TotalMilliseconds;
    }

    private double BenchmarkEncoder(SKBitmap image)
    {
        var stopwatch = Stopwatch.StartNew();

        var encoderPath = Path.Combine("..", "..", "..", "models", "encoder.onnx");
        using var encoderSession = new Microsoft.ML.OnnxRuntime.InferenceSession(encoderPath);

        // Preprocessing
        using var resized = image.Resize(new SKImageInfo(448, 448), new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.Linear));
        var data = new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(new[] { 1, 3, 448, 448 });

        for (int y = 0; y < 448; y++)
        {
            for (int x = 0; x < 448; x++)
            {
                var color = resized.GetPixel(x, y);
                data[0, 0, y, x] = color.Red / 255f;
                data[0, 1, y, x] = color.Green / 255f;
                data[0, 2, y, x] = color.Blue / 255f;
            }
        }

        // Inference
        var input = Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor("images", data);
        using var results = encoderSession.Run(new[] { input });

        stopwatch.Stop();
        return stopwatch.Elapsed.TotalMilliseconds;
    }

    private double BenchmarkPipelineStep(SKBitmap image)
    {
        var stopwatch = Stopwatch.StartNew();

        var encoderPath = Path.Combine("..", "..", "..", "models", "encoder.onnx");
        var bboxDecoderPath = Path.Combine("..", "..", "..", "models", "bbox_decoder.onnx");

        using var encoderSession = new Microsoft.ML.OnnxRuntime.InferenceSession(encoderPath);
        using var bboxSession = new Microsoft.ML.OnnxRuntime.InferenceSession(bboxDecoderPath);

        // Preprocessing
        using var resized = image.Resize(new SKImageInfo(448, 448), new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.Linear));
        var data = new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(new[] { 1, 3, 448, 448 });

        for (int y = 0; y < 448; y++)
        {
            for (int x = 0; x < 448; x++)
            {
                var color = resized.GetPixel(x, y);
                data[0, 0, y, x] = color.Red / 255f;
                data[0, 1, y, x] = color.Green / 255f;
                data[0, 2, y, x] = color.Blue / 255f;
            }
        }

        // Encoder inference
        var input = Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor("images", data);
        using var encoderResults = encoderSession.Run(new[] { input });
        var encoderOut = encoderResults[0].AsTensor<float>();

        // BboxDecoder inference
        var bboxInput = Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor("encoder_out",
            new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(encoderOut.ToArray(), encoderOut.Dimensions.ToArray()));
        using var bboxResults = bboxSession.Run(new[] { bboxInput });

        stopwatch.Stop();
        return stopwatch.Elapsed.TotalMilliseconds;
    }

    private void AnalyzeResults(double pipelineTime, double preprocessingTime, double encoderTime, double pipelineStepTime)
    {
        Console.WriteLine("\n📊 PERFORMANCE ANALYSIS:");
        Console.WriteLine("========================");

        var inferenceTime = pipelineStepTime - preprocessingTime;
        var overhead = pipelineTime - pipelineStepTime;

        Console.WriteLine($"Preprocessing: {preprocessingTime:F2}ms ({preprocessingTime/pipelineTime*100:F1}%)");
        Console.WriteLine($"Inference:     {inferenceTime:F2}ms ({inferenceTime/pipelineTime*100:F1}%)");
        Console.WriteLine($"Overhead:      {overhead:F2}ms ({overhead/pipelineTime*100:F1}%)");
        Console.WriteLine($"Total:         {pipelineTime:F2}ms");

        Console.WriteLine("\n🎯 PERFORMANCE TARGETS:");
        Console.WriteLine("=======================");
        Console.WriteLine("Target Python:     ≤ 800ms");
        Console.WriteLine($"Current .NET:      {pipelineTime:F2}ms");
        Console.WriteLine($"Gap to target:     {pipelineTime - 800:F2}ms");

        if (pipelineTime <= 800)
        {
            Console.WriteLine("✅ PERFORMANCE GOAL: RAGGIUNTO!");
        }
        else
        {
            Console.WriteLine("⚠️  PERFORMANCE GOAL: NON RAGGIUNTO");
            Console.WriteLine("\n🔧 OTTIMIZZAZIONI RICHIESTE:");
            if (preprocessingTime > 100)
                Console.WriteLine("  • Ottimizzare preprocessing immagini");
            if (inferenceTime > 600)
                Console.WriteLine("  • Ottimizzare inference ONNX");
            if (overhead > 100)
                Console.WriteLine("  • Ridurre overhead pipeline");
        }

        Console.WriteLine("\n📈 RACCOMANDAZIONI:");
        Console.WriteLine("===================");
        Console.WriteLine("1. Parallel processing per modelli pipeline");
        Console.WriteLine("2. Tensor reuse tra inference");
        Console.WriteLine("3. Memory pooling per ottimizzazione GC");
        Console.WriteLine("4. Native preprocessing con Span<T>");
        Console.WriteLine("5. Session reuse per modelli ONNX");
    }
}
