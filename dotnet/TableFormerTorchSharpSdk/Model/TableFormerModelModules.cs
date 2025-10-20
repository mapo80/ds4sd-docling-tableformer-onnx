using System;
using System.Collections.Generic;

using TorchSharp;
using TorchSharp.Modules;
using torch = TorchSharp.torch;
using TorchTensor = TorchSharp.torch.Tensor;

namespace TableFormerTorchSharpSdk.Model;

internal sealed class TableModel04RsModule : torch.nn.Module<TorchTensor, TorchTensor>
{
    internal readonly Encoder04Module _encoder;
    internal readonly TagTransformerModule _tag_transformer;
    internal readonly BBoxDecoderModule _bbox_decoder;
    private readonly AdaptiveAvgPool2d _adaptivePool;

    private readonly int _encImageSize;
    private readonly int _hiddenDim;

    public TableModel04RsModule(
        int encImageSize,
        int hiddenDim,
        TagTransformerModule tagTransformer,
        BBoxDecoderModule bboxDecoder)
        : base(nameof(TableModel04RsModule))
    {
        _encImageSize = encImageSize;
        _hiddenDim = hiddenDim;
        _encoder = new Encoder04Module(encImageSize, hiddenDim);
        _tag_transformer = tagTransformer;
        _bbox_decoder = bboxDecoder;
        _adaptivePool = torch.nn.AdaptiveAvgPool2d((encImageSize, encImageSize));

        register_module(nameof(_encoder), _encoder);
        register_module(nameof(_tag_transformer), _tag_transformer);
        register_module(nameof(_bbox_decoder), _bbox_decoder);
        register_module("_adaptive_pool", _adaptivePool);
    }

    public override TorchTensor forward(TorchTensor input)
    {
        throw new NotSupportedException("Use Predict instead of forward().");
    }

    public TorchTensor EncodeImage(TorchTensor images)
    {
        var encoded = _encoder.forward(images);
        var pooled = _adaptivePool.call(encoded);
        return pooled.permute(0, 2, 3, 1);
    }
}

internal sealed class Encoder04Module : torch.nn.Module<TorchTensor, TorchTensor>
{
    private readonly Sequential _resnet;

    public Encoder04Module(int encImageSize, int hiddenDim)
        : base(nameof(Encoder04Module))
    {
        _resnet = BuildResNetTrunk();
        register_module("_resnet", _resnet);
    }

    public override TorchTensor forward(TorchTensor input)
    {
        return _resnet.call(input);
    }

    private static Sequential BuildResNetTrunk()
    {
        var conv1 = torch.nn.Conv2d(3, 64, (7, 7), stride: (2, 2), padding: (3, 3), bias: false);
        var bn1 = torch.nn.BatchNorm2d(64);
        var relu = torch.nn.ReLU(inplace: true);
        var maxPool = torch.nn.MaxPool2d((3, 3), stride: (2, 2), padding: (1, 1));

        var layer1 = BuildLayer(64, 64, stride: 1);
        var layer2 = BuildLayer(64, 128, stride: 2);
        var layer3 = BuildLayer(128, 256, stride: 2);

        return torch.nn.Sequential(conv1, bn1, relu, maxPool, layer1, layer2, layer3);
    }

    private static Sequential BuildLayer(int inChannels, int outChannels, int stride)
    {
        var blocks = new List<torch.nn.Module<TorchTensor, TorchTensor>>
        {
            new BasicBlockModule(inChannels, outChannels, stride),
            new BasicBlockModule(outChannels, outChannels, 1),
        };

        return torch.nn.Sequential(blocks.ToArray());
    }
}

internal sealed class BasicBlockModule : torch.nn.Module<TorchTensor, TorchTensor>
{
    private readonly Conv2d _conv1;
    private readonly BatchNorm2d _bn1;
    private readonly ReLU _relu;
    private readonly Conv2d _conv2;
    private readonly BatchNorm2d _bn2;
    private readonly Sequential? _downsample;

    public BasicBlockModule(int inChannels, int outChannels, int stride)
        : base(nameof(BasicBlockModule))
    {
        _conv1 = torch.nn.Conv2d(inChannels, outChannels, (3, 3), stride: (stride, stride), padding: (1, 1), bias: false);
        _bn1 = torch.nn.BatchNorm2d(outChannels);
        _relu = torch.nn.ReLU(inplace: true);
        _conv2 = torch.nn.Conv2d(outChannels, outChannels, (3, 3), stride: (1, 1), padding: (1, 1), bias: false);
        _bn2 = torch.nn.BatchNorm2d(outChannels);

        if (stride != 1 || inChannels != outChannels)
        {
            var downsample = torch.nn.Sequential(
                torch.nn.Conv2d(inChannels, outChannels, (1, 1), stride: (stride, stride), bias: false),
                torch.nn.BatchNorm2d(outChannels));
            _downsample = downsample;
            register_module("downsample", downsample);
        }

        register_module("conv1", _conv1);
        register_module("bn1", _bn1);
        register_module("relu", _relu);
        register_module("conv2", _conv2);
        register_module("bn2", _bn2);
    }

    public override TorchTensor forward(TorchTensor input)
    {
        var identity = input;

        var outTensor = _conv1.call(input);
        outTensor = _bn1.call(outTensor);
        outTensor = _relu.call(outTensor);

        outTensor = _conv2.call(outTensor);
        outTensor = _bn2.call(outTensor);

        if (_downsample is not null)
        {
            identity = _downsample.call(input);
        }

        outTensor = torch.add(outTensor, identity);
        outTensor = _relu.call(outTensor);
        return outTensor;
    }
}

internal sealed class TagTransformerModule : torch.nn.Module<TorchTensor, TorchTensor>
{
    internal readonly Embedding _embedding;
    internal readonly PositionalEncodingModule _positional_encoding;
    internal readonly TransformerEncoder _encoder;
    internal readonly TMTransformerDecoderModule _decoder;
    internal readonly Sequential _input_filter;
    internal readonly Linear _fc;
    internal readonly int _n_heads;

    public TagTransformerModule(
        int vocabSize,
        int embedDim,
        int encoderLayers,
        int decoderLayers,
        int encImageSize,
        int nHeads,
        int dimFeedForward)
        : base(nameof(TagTransformerModule))
    {
        _n_heads = nHeads;
        _embedding = torch.nn.Embedding(vocabSize, embedDim);
        _positional_encoding = new PositionalEncodingModule(embedDim, 0.1, 1024);
        _encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model: embedDim, nhead: nHeads, dim_feedforward: dimFeedForward, dropout: 0.1),
            encoderLayers);
        _decoder = new TMTransformerDecoderModule(embedDim, nHeads, dimFeedForward, decoderLayers);
        _input_filter = BuildInputFilter();
        _fc = torch.nn.Linear(embedDim, vocabSize);

        register_module("_embedding", _embedding);
        register_module("_positional_encoding", _positional_encoding);
        register_module("_encoder", _encoder);
        register_module("_decoder", _decoder);
        register_module("_input_filter", _input_filter);
        register_module("_fc", _fc);
    }

    internal static Sequential BuildInputFilter()
    {
        var block1 = new BasicBlockModule(256, 512, 1);
        var block2 = new BasicBlockModule(512, 512, 1);
        return torch.nn.Sequential(block1, block2);
    }

    public override TorchTensor forward(TorchTensor input)
    {
        throw new NotSupportedException("Use explicit transformer helpers instead.");
    }
}

internal sealed class PositionalEncodingModule : torch.nn.Module<TorchTensor, TorchTensor>
{
    private readonly int _dModel;
    private readonly Dropout _dropout;
    private TorchTensor _pe;

    public PositionalEncodingModule(int dModel, double dropout, int maxLen)
        : base(nameof(PositionalEncodingModule))
    {
        _dModel = dModel;
        _dropout = torch.nn.Dropout(dropout);
        _pe = CreateEncoding(dModel, maxLen);
        register_buffer("pe", _pe);
        register_module("dropout", _dropout);
    }

    public override TorchTensor forward(TorchTensor input)
    {
        var length = input.shape[0];
        var slice = _pe.narrow(0, 0, length);
        var added = torch.add(input, slice);
        return _dropout.call(added);
    }

    private static TorchTensor CreateEncoding(int dModel, int maxLen)
    {
        var values = new float[maxLen * dModel];
        for (var pos = 0; pos < maxLen; pos++)
        {
            for (var i = 0; i < dModel; i += 2)
            {
                var angle = (float)(pos / Math.Pow(10000.0, (double)i / dModel));
                values[pos * dModel + i] = MathF.Sin(angle);
                if (i + 1 < dModel)
                {
                    values[pos * dModel + i + 1] = MathF.Cos(angle);
                }
            }
        }

        var tensor = torch.tensor(values, new long[] { maxLen, dModel }, dtype: torch.ScalarType.Float32);
        return tensor.unsqueeze(1);
    }
}

internal sealed class TMTransformerDecoderModule : torch.nn.Module<TorchTensor, TorchTensor>
{
    private readonly ModuleList<TMTransformerDecoderLayerModule> _layers;

    public TMTransformerDecoderModule(int embedDim, int nHeads, int dimFeedForward, int layerCount)
        : base(nameof(TMTransformerDecoderModule))
    {
        var layers = new List<TMTransformerDecoderLayerModule>(layerCount);
        for (var i = 0; i < layerCount; i++)
        {
            layers.Add(new TMTransformerDecoderLayerModule(embedDim, nHeads, dimFeedForward));
        }

        _layers = torch.nn.ModuleList(layers.ToArray());
        register_module("layers", _layers);
    }

    public override TorchTensor forward(TorchTensor input)
    {
        throw new NotSupportedException("Use Decode method instead.");
    }

    public (TorchTensor output, TorchTensor cache) Decode(
        TorchTensor tgt,
        TorchTensor memory,
        TorchTensor? cache,
        TorchTensor? memoryKeyPaddingMask)
    {
        var output = tgt;
        var layerOutputs = new List<TorchTensor>(_layers.Count);

        for (var i = 0; i < _layers.Count; i++)
        {
            var layer = _layers[i];
            var layerOutput = layer.Process(output, memory, memoryKeyPaddingMask);
            layerOutputs.Add(layerOutput);

            if (cache is not null)
            {
                var cachedLayer = cache[i];
                output = torch.cat(new[] { cachedLayer, layerOutput }, dim: 0);
            }
            else
            {
                output = layerOutput;
            }
        }

        var stacked = torch.stack(layerOutputs.ToArray(), dim: 0);
        if (cache is not null)
        {
            var updatedCache = torch.cat(new[] { cache, stacked }, dim: 1);
            return (output, updatedCache);
        }

        return (output, stacked);
    }
}

internal sealed class TMTransformerDecoderLayerModule : torch.nn.Module<TorchTensor, TorchTensor>
{
    private readonly MultiheadAttention _self_attn;
    private readonly MultiheadAttention _multihead_attn;
    private readonly Linear _linear1;
    private readonly Linear _linear2;
    private readonly LayerNorm _norm1;
    private readonly LayerNorm _norm2;
    private readonly LayerNorm _norm3;
    private readonly Dropout _dropout;
    private readonly Dropout _dropout1;
    private readonly Dropout _dropout2;
    private readonly Dropout _dropout3;

    public TMTransformerDecoderLayerModule(int embedDim, int nHeads, int dimFeedForward)
        : base(nameof(TMTransformerDecoderLayerModule))
    {
        _self_attn = torch.nn.MultiheadAttention(embedDim, nHeads, dropout: 0.0);
        _multihead_attn = torch.nn.MultiheadAttention(embedDim, nHeads, dropout: 0.0);
        _linear1 = torch.nn.Linear(embedDim, dimFeedForward);
        _linear2 = torch.nn.Linear(dimFeedForward, embedDim);
        _norm1 = torch.nn.LayerNorm(embedDim);
        _norm2 = torch.nn.LayerNorm(embedDim);
        _norm3 = torch.nn.LayerNorm(embedDim);
        _dropout = torch.nn.Dropout(0.1);
        _dropout1 = torch.nn.Dropout(0.1);
        _dropout2 = torch.nn.Dropout(0.1);
        _dropout3 = torch.nn.Dropout(0.1);

        register_module("self_attn", _self_attn);
        register_module("multihead_attn", _multihead_attn);
        register_module("linear1", _linear1);
        register_module("linear2", _linear2);
        register_module("norm1", _norm1);
        register_module("norm2", _norm2);
        register_module("norm3", _norm3);
        register_module("dropout", _dropout);
        register_module("dropout1", _dropout1);
        register_module("dropout2", _dropout2);
        register_module("dropout3", _dropout3);
    }

    public override TorchTensor forward(TorchTensor input)
    {
        throw new NotSupportedException("Use Process instead.");
    }

    public TorchTensor Process(TorchTensor tgt, TorchTensor memory, TorchTensor? memoryKeyPaddingMask)
    {
        var lastIndex = tgt.shape[0] - 1;
        using var index = torch.tensor(new long[] { lastIndex }, dtype: torch.int64);
        var tgtLast = tgt.index_select(0, index);

        var (selfAttn, _) = _self_attn.call(tgtLast, tgt, tgt, key_padding_mask: null, need_weights: false, attn_mask: null);
        selfAttn = _dropout1.call(selfAttn);
        var residual = torch.add(tgtLast, selfAttn);
        residual = _norm1.call(residual);

        TorchTensor attnResidual = residual;
        if (memory is not null)
        {
            var (attnOutput, _) = _multihead_attn.call(
                residual,
                memory,
                memory,
                key_padding_mask: memoryKeyPaddingMask,
                need_weights: false,
                attn_mask: null);
            attnOutput = _dropout2.call(attnOutput);
            attnResidual = _norm2.call(torch.add(residual, attnOutput));
        }

        var linearOut = _linear2.call(_dropout.call(torch.nn.functional.relu(_linear1.call(attnResidual))));
        linearOut = _dropout3.call(linearOut);
        var output = _norm3.call(torch.add(attnResidual, linearOut));
        return output;
    }
}

internal sealed class CellAttentionModule : torch.nn.Module<TorchTensor, TorchTensor>
{
    private readonly Linear _encoder_att;
    private readonly Linear _tag_decoder_att;
    private readonly Linear _language_att;
    private readonly Linear _full_att;
    private readonly ReLU _relu;
    private readonly Softmax _softmax;

    public CellAttentionModule(int encoderDim, int tagDecoderDim, int languageDim, int attentionDim)
        : base(nameof(CellAttentionModule))
    {
        _encoder_att = torch.nn.Linear(encoderDim, attentionDim);
        _tag_decoder_att = torch.nn.Linear(tagDecoderDim, attentionDim);
        _language_att = torch.nn.Linear(languageDim, attentionDim);
        _full_att = torch.nn.Linear(attentionDim, 1);
        _relu = torch.nn.ReLU();
        _softmax = torch.nn.Softmax(dim: 1);

        register_module("_encoder_att", _encoder_att);
        register_module("_tag_decoder_att", _tag_decoder_att);
        register_module("_language_att", _language_att);
        register_module("_full_att", _full_att);
        register_module("_relu", _relu);
        register_module("_softmax", _softmax);
    }

    public override TorchTensor forward(TorchTensor input)
    {
        throw new NotSupportedException("Use Apply instead.");
    }

    public (TorchTensor weightedEncoding, TorchTensor weights) Apply(
        TorchTensor encoderOut,
        TorchTensor decoderHidden,
        TorchTensor languageOut)
    {
        using var projection = ProjectLanguage(languageOut);
        return ApplyWithLanguageProjection(encoderOut, decoderHidden, projection);
    }

    public TorchTensor ProjectLanguage(TorchTensor languageOut)
    {
        ArgumentNullException.ThrowIfNull(languageOut);
        return _language_att.call(languageOut);
    }

    public (TorchTensor weightedEncoding, TorchTensor weights) ApplyWithLanguageProjection(
        TorchTensor encoderOut,
        TorchTensor decoderHidden,
        TorchTensor languageProjection)
    {
        ArgumentNullException.ThrowIfNull(languageProjection);

        using var att1 = _encoder_att.call(encoderOut);
        using var att2 = _tag_decoder_att.call(decoderHidden).unsqueeze(1);
        using var att3 = languageProjection.unsqueeze(1);
        using var preActivation = att1 + att2 + att3;
        using var relu = _relu.call(preActivation);
        using var logits = _full_att.call(relu).squeeze(2);
        var alpha = _softmax.call(logits);
        using var alphaExpanded = alpha.unsqueeze(2);
        var weighted = (encoderOut * alphaExpanded).sum(1);
        return (weighted, alpha);
    }
}

internal sealed class MlpModule : torch.nn.Module<TorchTensor, TorchTensor>
{
    private readonly ModuleList<Linear> _layers;
    private readonly int _layerCount;

    public MlpModule(int inputDim, int hiddenDim, int outputDim, int layerCount)
        : base(nameof(MlpModule))
    {
        _layerCount = layerCount;
        var layers = new List<Linear>(layerCount);
        var dims = new List<int> { inputDim };
        for (var i = 0; i < layerCount - 1; i++)
        {
            dims.Add(hiddenDim);
        }
        dims.Add(outputDim);

        for (var i = 0; i < layerCount; i++)
        {
            var layer = torch.nn.Linear(dims[i], dims[i + 1]);
            layers.Add(layer);
        }

        _layers = torch.nn.ModuleList(layers.ToArray());
        register_module("layers", _layers);
    }

    public override TorchTensor forward(TorchTensor input)
    {
        var x = input;
        for (var i = 0; i < _layerCount; i++)
        {
            var layer = (Linear)_layers[i];
            x = layer.call(x);
            if (i < _layerCount - 1)
            {
                x = torch.nn.functional.relu(x);
            }
        }

        return x;
    }
}

internal sealed class BBoxDecoderModule : torch.nn.Module<TorchTensor, TorchTensor>
{
    private readonly CellAttentionModule _attention;
    private readonly Linear _init_h;
    private readonly Linear _f_beta;
    private readonly Linear _class_embed;
    private readonly MlpModule _bbox_embed;
    private readonly Sigmoid _sigmoid;
    private readonly Sequential _input_filter;

    public BBoxDecoderModule(
        int attentionDim,
        int embedDim,
        int tagDecoderDim,
        int decoderDim,
        int numClasses,
        int encoderDim)
        : base(nameof(BBoxDecoderModule))
    {
        _attention = new CellAttentionModule(encoderDim, tagDecoderDim, decoderDim, attentionDim);
        _init_h = torch.nn.Linear(encoderDim, decoderDim);
        _f_beta = torch.nn.Linear(decoderDim, encoderDim);
        _class_embed = torch.nn.Linear(512, numClasses + 1);
        _bbox_embed = new MlpModule(512, 256, 4, 3);
        _sigmoid = torch.nn.Sigmoid();
        _input_filter = TagTransformerModule.BuildInputFilter();

        register_module("_attention", _attention);
        register_module("_init_h", _init_h);
        register_module("_f_beta", _f_beta);
        register_module("_class_embed", _class_embed);
        register_module("_bbox_embed", _bbox_embed);
        register_module("_sigmoid", _sigmoid);
        register_module("_input_filter", _input_filter);
    }

    public override TorchTensor forward(TorchTensor input)
    {
        throw new NotSupportedException("Use DecodeBoundingBoxes instead.");
    }

    public TorchTensor PrepareEncoderFeatures(TorchTensor encoderChannelsFirst)
    {
        ArgumentNullException.ThrowIfNull(encoderChannelsFirst);
        return _input_filter.call(encoderChannelsFirst).permute(0, 2, 3, 1);
    }

    public (TorchTensor classes, TorchTensor coordinates) DecodeBoundingBoxes(
        TorchTensor filteredEncoderOut,
        IReadOnlyList<TorchTensor> tagHiddenStates)
    {
        if (tagHiddenStates.Count == 0)
        {
            return (
                torch.empty(new long[] { 0 }, dtype: torch.ScalarType.Float32),
                torch.empty(new long[] { 0 }, dtype: torch.ScalarType.Float32));
        }

        var encoderDim = filteredEncoderOut.shape[3];
        using var flattened = filteredEncoderOut.view(new long[] { 1, -1, encoderDim });
        using var baseHidden = InitializeHidden(flattened, 1);
        using var gateBase = _sigmoid.call(_f_beta.call(baseHidden));
        using var languageProjection = _attention.ProjectLanguage(baseHidden);

        var classOutputs = new List<TorchTensor>(tagHiddenStates.Count);
        var bboxOutputs = new List<TorchTensor>(tagHiddenStates.Count);

        foreach (var tagHidden in tagHiddenStates)
        {
            var (attentionWeighted, attentionWeights) = _attention.ApplyWithLanguageProjection(
                flattened,
                tagHidden,
                languageProjection);
            using (attentionWeights)
            using (attentionWeighted)
            {
                using var gated = gateBase * attentionWeighted;
                using var contextual = gated * baseHidden;

                using var bboxFull = _bbox_embed.forward(contextual).sigmoid();
                var bboxSlice = bboxFull[0].clone();
                bboxOutputs.Add(bboxSlice);

                using var classFull = _class_embed.call(contextual);
                var classSlice = classFull[0].clone();
                classOutputs.Add(classSlice);
            }
        }

        var bboxTensor = bboxOutputs.Count > 0
            ? torch.stack(bboxOutputs.ToArray())
            : torch.empty(new long[] { 0 }, dtype: torch.ScalarType.Float32);
        var classTensor = classOutputs.Count > 0
            ? torch.stack(classOutputs.ToArray())
            : torch.empty(new long[] { 0 }, dtype: torch.ScalarType.Float32);

        foreach (var tensor in bboxOutputs)
        {
            tensor.Dispose();
        }

        foreach (var tensor in classOutputs)
        {
            tensor.Dispose();
        }

        return (classTensor, bboxTensor);
    }

    private TorchTensor InitializeHidden(TorchTensor encoderOut, int batchSize)
    {
        var mean = encoderOut.mean(new long[] { 1 }, keepdim: false);
        return _init_h.call(mean).expand(batchSize, -1);
    }
}

