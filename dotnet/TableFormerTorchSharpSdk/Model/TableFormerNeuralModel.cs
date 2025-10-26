using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json.Nodes;

using TorchSharp;
using torch = TorchSharp.torch;
using TorchTensor = TorchSharp.torch.Tensor;

using TableFormerTorchSharpSdk.Configuration;
using TableFormerTorchSharpSdk.Initialization;
using TableFormerTorchSharpSdk.Safetensors;

namespace TableFormerTorchSharpSdk.Model;

internal sealed class TableFormerNeuralModel : IDisposable
{
    private static readonly object TorchInitLock = new();
    private static bool TorchThreadsConfigured;
    private static int? RequestedComputeThreads;
    private static int? RequestedInteropThreads;

    private const string TorchThreadsEnvironmentVariable = "TABLEFORMER_TORCH_THREADS";
    private const string TorchInteropThreadsEnvironmentVariable = "TABLEFORMER_TORCH_INTEROP_THREADS";

    private readonly TableModel04RsModule _model;
    private readonly IReadOnlyDictionary<string, int> _wordMapTag;
    private readonly IReadOnlyDictionary<int, string> _reverseWordMapTag;
    private readonly int _maxSteps;
    private readonly int _startTag;
    private readonly int _endTag;
    private readonly int _lcelTag;
    private readonly int _xcelTag;
    private readonly int _cachedFcel;
    private readonly int _cachedEcel;
    private readonly int _cachedChed;
    private readonly int _cachedRhed;
    private readonly int _cachedSrow;
    private readonly int _cachedNl;
    private readonly int _cachedUcel;

    private TorchTensor? _encoderMaskCache;
    private int _encoderMaskPositions;

    private TorchTensor? _keyPaddingMaskCache;
    private int _keyPaddingMaskBatchSize;
    private int _keyPaddingMaskPositions;

    public TableFormerNeuralModel(
        TableFormerConfigSnapshot configSnapshot,
        TableFormerInitializationSnapshot initializationSnapshot,
        DirectoryInfo modelDirectory)
    {
        ArgumentNullException.ThrowIfNull(configSnapshot);
        ArgumentNullException.ThrowIfNull(initializationSnapshot);
        ArgumentNullException.ThrowIfNull(modelDirectory);

        lock (TorchInitLock)
        {
            if (!TorchThreadsConfigured)
            {
                var threadConfig = ResolveTorchThreadConfiguration();
                torch.set_num_threads(threadConfig.ComputeThreads);
                torch.set_num_interop_threads(threadConfig.InteropThreads);
                TorchThreadsConfigured = true;
            }

            torch.manual_seed(0);
        }

        var modelConfig = GetSection(configSnapshot.Config, "model");
        var predictConfig = GetSection(configSnapshot.Config, "predict");

        var encImageSize = GetInt(modelConfig, "enc_image_size");
        var hiddenDim = GetInt(modelConfig, "hidden_dim");
        var tagAttentionDim = GetInt(modelConfig, "tag_attention_dim");
        var tagDecoderDim = GetInt(modelConfig, "tag_decoder_dim");
        var bboxAttentionDim = GetInt(modelConfig, "bbox_attention_dim");
        var bboxEmbedDim = GetInt(modelConfig, "bbox_embed_dim");
        var encLayers = GetInt(modelConfig, "enc_layers");
        var decLayers = GetInt(modelConfig, "dec_layers");
        var nHeads = GetInt(modelConfig, "nheads");
        var numClasses = GetInt(modelConfig, "bbox_classes");
        _maxSteps = GetInt(predictConfig, "max_steps");

        _wordMapTag = initializationSnapshot.WordMap.TagMap;
        _reverseWordMapTag = initializationSnapshot.WordMap.ReverseTagMap;
        _startTag = GetTagIndex("<start>");
        _endTag = GetTagIndex("<end>");
        _lcelTag = GetTagIndex("lcel");
        _xcelTag = GetTagIndex("xcel");
        _cachedFcel = GetTagIndex("fcel");
        _cachedEcel = GetTagIndex("ecel");
        _cachedChed = GetTagIndex("ched");
        _cachedRhed = GetTagIndex("rhed");
        _cachedSrow = GetTagIndex("srow");
        _cachedNl = GetTagIndex("nl");
        _cachedUcel = GetTagIndex("ucel");

        var vocabSize = _wordMapTag.Count;
        var tagTransformer = new TagTransformerModule(
            vocabSize,
            hiddenDim,
            encLayers,
            decLayers,
            encImageSize,
            nHeads,
            tagAttentionDim * 4);

        var bboxDecoder = new BBoxDecoderModule(
            bboxAttentionDim,
            bboxEmbedDim,
            tagDecoderDim,
            hiddenDim,
            numClasses,
            hiddenDim);

        _model = new TableModel04RsModule(encImageSize, hiddenDim, tagTransformer, bboxDecoder);

        LoadWeights(modelDirectory, _model);
        _model.eval();
    }

    public static void ConfigureThreading(int? computeThreads, int? interopThreads = null)
    {
        lock (TorchInitLock)
        {
            RequestedComputeThreads = SanitizeThreadCount(computeThreads);
            RequestedInteropThreads = SanitizeThreadCount(interopThreads);
            TorchThreadsConfigured = false;
        }
    }

    public TableFormerNeuralPrediction Predict(TorchTensor tensor)
    {
        using var inferenceGuard = torch.inference_mode();
        using var encoded = _model.EncodeImage(tensor);
        using var encoderChannelsFirst = encoded.permute(0, 3, 1, 2);
        using var filteredForTransformer = _model._tag_transformer._input_filter.call(encoderChannelsFirst);
        using var encoderOut = filteredForTransformer.permute(0, 2, 3, 1);
        using var filteredForBounding = _model._bbox_decoder.PrepareEncoderFeatures(encoderChannelsFirst);

        var batchSize = (int)encoderOut.shape[0];
        var encoderDim = (int)encoderOut.shape[^1];

        using var encoderView = encoderOut.view(batchSize, -1, encoderDim);
        using var encInputs = encoderView.permute(1, 0, 2);
        var positions = (int)encInputs.shape[0];

        var encoderMask = GetEncoderMask(positions);
        encoderMask.zero_();
        var keyPaddingMask = GetKeyPaddingMask(batchSize, positions);
        keyPaddingMask.zero_();
        using var transformerEncoderOut = _model._tag_transformer._encoder.call(encInputs, encoderMask);
        using var decodedTagsBuffer = torch.empty(new long[] { _maxSteps + 1, 1 }, dtype: torch.ScalarType.Int64);
        var decodedTagsAccessor = decodedTagsBuffer.data<long>();
        decodedTagsAccessor[0, 0] = _startTag;
        var sequenceLength = 1;

        var outputTags = new List<int>();
        TorchTensor? cache = null;
        var tagHiddenStates = new List<TorchTensor>();

        var bboxesToMerge = new Dictionary<int, int>();
        var skipNextTag = true;
        var prevTagUcel = false;
        var firstLcel = true;
        var curBboxIndex = -1;
        var bboxIndex = 0;

        try
        {
            while (outputTags.Count < _maxSteps)
            {
                using var decodedTagsView = decodedTagsBuffer.narrow(0, 0, sequenceLength);
                using var decodedEmbedding = _model._tag_transformer._embedding.call(decodedTagsView);
                using var positioned = _model._tag_transformer._positional_encoding.forward(decodedEmbedding);

                var (decodedTensor, newCache) = _model._tag_transformer._decoder.Decode(
                    positioned,
                    transformerEncoderOut,
                    cache,
                    keyPaddingMask);

                cache?.Dispose();
                cache = newCache;

                using var decoded = decodedTensor;
                using var lastStep = decoded[decoded.shape[0] - 1];
                using var logits = _model._tag_transformer._fc.call(lastStep);
                using var argmax = logits.argmax(dim: 1);
                using var argmaxCpu = argmax.to(torch.CPU);
                var newTag = (int)argmaxCpu.data<long>()[0];

                if (outputTags.Count == 0 && newTag == _xcelTag)
                {
                    newTag = _lcelTag;
                }

                if (prevTagUcel && newTag == _lcelTag)
                {
                    newTag = _cachedFcel;
                }

                if (newTag == _endTag)
                {
                    outputTags.Add(newTag);
                    decodedTagsAccessor[sequenceLength, 0] = newTag;
                    sequenceLength += 1;
                    break;
                }

                outputTags.Add(newTag);

                if (!skipNextTag && IsBoundingTag(newTag))
                {
                    tagHiddenStates.Add(lastStep.clone());
                    if (!firstLcel)
                    {
                        bboxesToMerge[curBboxIndex] = bboxIndex;
                    }

                    bboxIndex += 1;
                }

                if (newTag != _lcelTag)
                {
                    firstLcel = true;
                }
                else if (firstLcel)
                {
                    tagHiddenStates.Add(lastStep.clone());
                    firstLcel = false;
                    curBboxIndex = bboxIndex;
                    bboxesToMerge[curBboxIndex] = -1;
                    bboxIndex += 1;
                }

                skipNextTag = newTag == _cachedNl || newTag == _cachedUcel || newTag == _xcelTag;
                prevTagUcel = newTag == _cachedUcel;

                decodedTagsAccessor[sequenceLength, 0] = newTag;
                sequenceLength += 1;
            }
        }
        finally
        {
            cache?.Dispose();
        }

        var sequence = new int[sequenceLength];
        sequence[0] = _startTag;
        for (var i = 0; i < outputTags.Count; i++)
        {
            sequence[i + 1] = outputTags[i];
        }

        var (classTensor, coordTensor) = _model._bbox_decoder.DecodeBoundingBoxes(
            filteredForBounding,
            tagHiddenStates);
        foreach (var hidden in tagHiddenStates)
        {
            hidden.Dispose();
        }

        var (mergedClasses, mergedCoords) = MergeBoundingBoxes(classTensor, coordTensor, bboxesToMerge);

        return new TableFormerNeuralPrediction(sequence, mergedClasses, mergedCoords);
    }

    public void Dispose()
    {
        _model.Dispose();
        _encoderMaskCache?.Dispose();
        _encoderMaskCache = null;
        _keyPaddingMaskCache?.Dispose();
        _keyPaddingMaskCache = null;
    }

    private static TorchThreadConfiguration ResolveTorchThreadConfiguration()
    {
        var requestedThreads = RequestedComputeThreads
            ?? ParseEnvironmentOverride(TorchThreadsEnvironmentVariable)
            ?? Math.Max(1, Environment.ProcessorCount);

        var requestedInterop = RequestedInteropThreads
            ?? ParseEnvironmentOverride(TorchInteropThreadsEnvironmentVariable)
            ?? Math.Max(1, Math.Min(requestedThreads, Environment.ProcessorCount));

        return new TorchThreadConfiguration(requestedThreads, requestedInterop);
    }

    private static int? ParseEnvironmentOverride(string variableName)
    {
        var value = Environment.GetEnvironmentVariable(variableName);
        if (string.IsNullOrWhiteSpace(value))
        {
            return null;
        }

        if (int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed) && parsed > 0)
        {
            return parsed;
        }

        return null;
    }

    private static int? SanitizeThreadCount(int? requested)
    {
        if (requested is null)
        {
            return null;
        }

        if (requested <= 0)
        {
            return null;
        }

        return requested;
    }

    private TorchTensor GetEncoderMask(int positions)
    {
        if (_encoderMaskCache is null || _encoderMaskPositions != positions)
        {
            _encoderMaskCache?.Dispose();
            _encoderMaskCache = torch.zeros(new long[] { positions, positions }, dtype: torch.ScalarType.Bool);
            _encoderMaskPositions = positions;
        }

        return _encoderMaskCache!;
    }

    private TorchTensor GetKeyPaddingMask(int batchSize, int positions)
    {
        if (_keyPaddingMaskCache is null ||
            _keyPaddingMaskBatchSize != batchSize ||
            _keyPaddingMaskPositions != positions)
        {
            _keyPaddingMaskCache?.Dispose();
            _keyPaddingMaskCache = torch.zeros(new long[] { batchSize, positions }, dtype: torch.ScalarType.Bool);
            _keyPaddingMaskBatchSize = batchSize;
            _keyPaddingMaskPositions = positions;
        }

        return _keyPaddingMaskCache!;
    }

    private static void LoadWeights(DirectoryInfo modelDirectory, TableModel04RsModule model)
    {
        var safetensors = modelDirectory.GetFiles("*.safetensors", SearchOption.TopDirectoryOnly);
        if (safetensors.Length == 0)
        {
            throw new FileNotFoundException($"No safetensors weights found in '{modelDirectory.FullName}'.");
        }

        var loader = SafeTensorLoader.LoadFromFiles(safetensors);
        using var noGrad = torch.no_grad();
        var parameters = model.named_parameters(recurse: true).ToDictionary(p => p.name, p => p.parameter, StringComparer.Ordinal);
        var buffers = model.named_buffers(recurse: true)
            .ToDictionary(b => b.name, b => b.buffer, StringComparer.Ordinal);

        foreach (var entry in loader.Entries)
        {
            using var tensor = entry.CreateTensor();
            if (parameters.TryGetValue(entry.Name, out var parameter))
            {
                parameter.copy_(tensor);
            }
            else if (buffers.TryGetValue(entry.Name, out var buffer))
            {
                buffer.copy_(tensor);
            }
            else
            {
                throw new InvalidDataException($"Unexpected tensor '{entry.Name}' encountered while loading weights.");
            }
        }
    }

    private (TorchTensor classes, TorchTensor coords) MergeBoundingBoxes(
        TorchTensor classes,
        TorchTensor coords,
        IReadOnlyDictionary<int, int> bboxesToMerge)
    {
        if (coords.numel() == 0)
        {
            return (classes, coords);
        }

        var mergedCoords = new List<TorchTensor>();
        var mergedClasses = new List<TorchTensor>();
        var skip = new HashSet<int>();

        for (var i = 0; i < coords.shape[0]; i++)
        {
            if (skip.Contains(i))
            {
                continue;
            }

            using var box = coords[i];
            using var cls = classes[i];

            if (bboxesToMerge.TryGetValue(i, out var mergeIndex) && mergeIndex >= 0)
            {
                using var other = coords[mergeIndex];
                skip.Add(mergeIndex);
                mergedCoords.Add(MergeBoxes(box, other));
                mergedClasses.Add(cls.clone());
            }
            else
            {
                mergedCoords.Add(box.clone());
                mergedClasses.Add(cls.clone());
            }
        }

        TorchTensor coordTensor;
        TorchTensor classTensor;

        if (mergedCoords.Count > 0)
        {
            coordTensor = torch.stack(mergedCoords.ToArray());
            classTensor = torch.stack(mergedClasses.ToArray());
        }
        else
        {
            coordTensor = torch.empty(new long[] { 0 }, dtype: torch.ScalarType.Float32);
            classTensor = torch.empty(new long[] { 0 }, dtype: torch.ScalarType.Float32);
        }

        foreach (var tensor in mergedCoords)
        {
            tensor.Dispose();
        }

        foreach (var tensor in mergedClasses)
        {
            tensor.Dispose();
        }

        classes.Dispose();
        coords.Dispose();

        return (classTensor, coordTensor);
    }

    private static TorchTensor MergeBoxes(TorchTensor bbox1, TorchTensor bbox2)
    {
        using var bbox1Float = bbox1.to_type(torch.ScalarType.Float32);
        using var bbox1Cpu = bbox1Float.to(torch.CPU);
        var bbox1Values = bbox1Cpu.data<float>().ToArray();
        using var bbox2Float = bbox2.to_type(torch.ScalarType.Float32);
        using var bbox2Cpu = bbox2Float.to(torch.CPU);
        var bbox2Values = bbox2Cpu.data<float>().ToArray();

        var bbox1Cx = bbox1Values[0];
        var bbox1Cy = bbox1Values[1];
        var bbox1W = bbox1Values[2];
        var bbox1H = bbox1Values[3];

        var bbox2Cx = bbox2Values[0];
        var bbox2Cy = bbox2Values[1];
        var bbox2W = bbox2Values[2];
        var bbox2H = bbox2Values[3];

        var newW = (bbox2Cx + bbox2W / 2f) - (bbox1Cx - bbox1W / 2f);
        var newH = (bbox2Cy + bbox2H / 2f) - (bbox1Cy - bbox1H / 2f);

        var newLeft = bbox1Cx - bbox1W / 2f;
        var newTop = MathF.Min(bbox2Cy - bbox2H / 2f, bbox1Cy - bbox1H / 2f);
        var newCx = newLeft + newW / 2f;
        var newCy = newTop + newH / 2f;

        return torch.tensor(new float[] { newCx, newCy, newW, newH }, dtype: torch.ScalarType.Float32);
    }

    private bool IsBoundingTag(int tag)
    {
        return tag == _cachedFcel || tag == _cachedEcel || tag == _cachedChed || tag == _cachedRhed ||
               tag == _cachedSrow || tag == _cachedNl || tag == _cachedUcel;
    }

    private static JsonObject GetSection(JsonObject root, string name)
    {
        if (root[name] is JsonObject section)
        {
            return section;
        }

        throw new InvalidDataException($"Configuration missing '{name}' section.");
    }

    private static int GetInt(JsonObject obj, string name)
    {
        var node = obj[name] ?? throw new InvalidDataException($"Configuration missing '{name}'.");
        return (int)Math.Round(node.GetValue<double>());
    }

    private int GetTagIndex(string name)
    {
        if (_wordMapTag.TryGetValue(name, out var value))
        {
            return value;
        }

        throw new InvalidDataException($"Word map is missing required tag '{name}'.");
    }
}

internal readonly record struct TorchThreadConfiguration(int ComputeThreads, int InteropThreads);

internal sealed class TableFormerNeuralPrediction : IDisposable
{
    public TableFormerNeuralPrediction(IReadOnlyList<int> sequence, TorchTensor classes, TorchTensor coordinates)
    {
        Sequence = sequence;
        Classes = classes;
        Coordinates = coordinates;
    }

    public IReadOnlyList<int> Sequence { get; }

    public TorchTensor Classes { get; }

    public TorchTensor Coordinates { get; }

    public void Dispose()
    {
        Classes.Dispose();
        Coordinates.Dispose();
    }
}
