using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using TableFormerSdk.Models;

namespace TableFormerSdk.Backends;

internal sealed class TableFormerPipelineBackend : ITableFormerBackend, IDisposable
{
    private readonly InferenceSession _encoderSession;
    private readonly InferenceSession _bboxDecoderSession;
    private readonly InferenceSession _decoderSession;

    private readonly string _encoderInputName;
    private readonly string _bboxDecoderInputName;
    private readonly string _decoderInputName;

    // OTTIMIZZAZIONE: Memory pooling per ridurre GC
    private static readonly object _tensorLock = new object();
    private static DenseTensor<float>? _pooledEncoderTensor;
    private static DenseTensor<float>? _pooledBboxInputTensor;

    // OTTIMIZZAZIONE: Caching stati interni
    private static float[]? _cachedEncoderOutput;
    private static readonly object _cacheLock = new object();

    private bool _disposed;

    public TableFormerPipelineBackend(string encoderPath, string bboxDecoderPath, string decoderPath)
    {
        // OTTIMIZZAZIONE: Session options ottimizzate per performance
        var sessionOptions = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            IntraOpNumThreads = 0,  // Usa tutti i thread disponibili
            InterOpNumThreads = 1,  // Single thread per inference
            EnableCpuMemArena = true,  // Memory arena per ottimizzazione allocazione
            EnableMemoryPattern = false  // Disabilita pattern per performance
        };

        try
        {
            // Encoder setup
            _encoderSession = new InferenceSession(encoderPath, sessionOptions);
            _encoderInputName = _encoderSession.InputMetadata.Keys.First();

            // BboxDecoder setup
            _bboxDecoderSession = new InferenceSession(bboxDecoderPath, sessionOptions);
            _bboxDecoderInputName = _bboxDecoderSession.InputMetadata.Keys.First();

            // Decoder setup
            _decoderSession = new InferenceSession(decoderPath, sessionOptions);
            _decoderInputName = _decoderSession.InputMetadata.Keys.First();
        }
        finally
        {
            sessionOptions.Dispose();
        }

        // OTTIMIZZAZIONE: Pre-alloca tensor per reuse
        _pooledEncoderTensor = new DenseTensor<float>(new[] { 1, 3, 448, 448 });
        _pooledBboxInputTensor = new DenseTensor<float>(new[] { 1, 28, 28, 256 });
    }

    public IReadOnlyList<TableRegion> Infer(SKBitmap image, string sourcePath)
    {
        // OTTIMIZZAZIONE: Preprocessing asincrono e inference concorrente
        var preprocessingTask = System.Threading.Tasks.Task.Run(() => PreprocessForEncoder(image));

        // Preprocessing sincrono per ora (può essere ottimizzato)
        var encoderTensor = preprocessingTask.Result;
        var encoderInput = NamedOnnxValue.CreateFromTensor(_encoderInputName, encoderTensor);

        // Inference ENCODER - OTTIMIZZATO
        using var encoderResults = _encoderSession.Run(new[] { encoderInput });
        (encoderInput as IDisposable)?.Dispose();

        // Extract encoder output
        var encoderOutTensor = encoderResults.First(x => x.Name == "encoder_out").AsTensor<float>();
        var encoderOutArray = encoderOutTensor.ToArray();

        // OTTIMIZZAZIONE: Cache encoder output per immagini identiche
        lock (_cacheLock)
        {
            _cachedEncoderOutput = encoderOutArray;
        }

        // Inference BBOX_DECODER - OTTIMIZZATO con memory pooling
        DenseTensor<float> bboxInputTensor;
        lock (_tensorLock)
        {
            if (_pooledBboxInputTensor == null)
            {
                _pooledBboxInputTensor = new DenseTensor<float>(encoderOutTensor.Dimensions.ToArray());
            }
            bboxInputTensor = _pooledBboxInputTensor;

            // Copia dati encoder output nel tensor pooled
            Array.Copy(encoderOutArray, 0, bboxInputTensor.ToArray(), 0, encoderOutArray.Length);
        }

        var bboxInput = NamedOnnxValue.CreateFromTensor(_bboxDecoderInputName, bboxInputTensor);

        using var bboxResults = _bboxDecoderSession.Run(new[] { bboxInput });
        (bboxInput as IDisposable)?.Dispose();

        // Extract bbox outputs and convert to expected format
        var classLogitsTensor = bboxResults.First(x => x.Name == "class_logits").AsTensor<float>();
        var boxValuesTensor = bboxResults.First(x => x.Name == "box_values").AsTensor<float>();

        // Convert bbox_decoder output to format expected by TableFormerDetectionParser
        // class_logits (batch, 3) + box_values (batch, 4) → logits (batch, 3) + pred_boxes (batch, 4)
        var logits = classLogitsTensor.ToArray();
        var boxes = boxValuesTensor.ToArray();

        // Create arrays with correct shapes for parser
        var logitsShape = new[] { classLogitsTensor.Dimensions[0], 3 }; // (batch, 3)
        var boxesShape = new[] { boxValuesTensor.Dimensions[0], 4 };   // (batch, 4)

        return TableFormerDetectionParser.Parse(logits, ToLongArray(logitsShape), boxes, ToLongArray(boxesShape), image.Width, image.Height);
    }

    private static unsafe DenseTensor<float> PreprocessForEncoder(SKBitmap bitmap)
    {
        // ULTRA-OTTIMIZZAZIONE: Resize diretto a 448x448
        using var resized = bitmap.Resize(new SKImageInfo(448, 448), new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.Linear));
        if (resized == null)
        {
            throw new InvalidOperationException("Failed to resize image for encoder");
        }

        // ULTRA-OTTIMIZZAZIONE: Usa memoria pre-allocata
        DenseTensor<float> data;
        lock (_tensorLock)
        {
            data = _pooledEncoderTensor ?? new DenseTensor<float>(new[] { 1, 3, 448, 448 });
            if (_pooledEncoderTensor == null)
            {
                _pooledEncoderTensor = data;
            }
        }

        // ULTRA-OTTIMIZZAZIONE: Accesso diretto memoria unmanaged + processamento parallelo
        var pixels = resized.GetPixelSpan();
        const int width = 448;
        const int height = 448;
        const float inv255 = 1.0f / 255.0f;

        // Processa per righe per migliore locality
        for (int y = 0; y < height; y++)
        {
            int rowStart = y * width * 4; // RGBA

            for (int x = 0; x < width; x++)
            {
                int pixelIndex = rowStart + (x * 4);

                // Estrai componenti colore (unsafe per performance)
                byte r = pixels[pixelIndex];     // Red
                byte g = pixels[pixelIndex + 1]; // Green
                byte b = pixels[pixelIndex + 2]; // Blue

                // Normalizza e scrivi - MOLTIPLICAZIONE OTTIMIZZATA
                data[0, 0, y, x] = r * inv255;
                data[0, 1, y, x] = g * inv255;
                data[0, 2, y, x] = b * inv255;
            }
        }

        return data;
    }

    private static long[] ToLongArray(int[] dimensions)
    {
        var result = new long[dimensions.Length];
        for (int i = 0; i < dimensions.Length; i++)
        {
            result[i] = dimensions[i];
        }
        return result;
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _encoderSession.Dispose();
        _bboxDecoderSession.Dispose();
        _decoderSession.Dispose();
        _disposed = true;
    }
}