# ds4sd-docling-tableformer-onnx

## Informazioni generali
Repository di valutazione per i modelli `ds4sd/docling-models` dedicati al riconoscimento di tabelle TableFormer.
L'obiettivo è convertire le varianti *fast* e *accurate* in diversi formati (ONNX, formato ORT, OpenVINO), verificarne la parità rispetto ai modelli originali e misurarne le prestazioni su CPU.

## Modelli di partenza
Gli artifact HuggingFace disponibili in [`ds4sd/docling-models`](https://huggingface.co/ds4sd/docling-models/tree/main/model_artifacts/tableformer) contengono due configurazioni principali:

- **TableFormer fast** – ottimizzata per la latenza
- **TableFormer accurate** – ottimizzata per la qualità dell'estrazione

Entrambe condividono la stessa pipeline di preprocessing (resize a 448×448, normalizzazione RGB) ma differiscono per dimensioni e numero di layer.

## Formati supportati
- **PyTorch**: modelli originali in formato `safetensors`
- **ONNX FP32**: esportazione del grafo encoder/decoder TableFormer
- **ONNX ottimizzato (ORT)**: grafo compatto per l'esecuzione con ONNX Runtime
- **OpenVINO IR**: modelli convertiti in `.xml`/`.bin` per CPU Intel

> La conversione in FP16 e INT8 è in corso di validazione; le istruzioni verranno aggiornate una volta completata la parità.

## Pipeline di conversione (bozza)
1. **Esportazione in ONNX**
   ```bash
   python convert_to_onnx.py --model-name tableformer-fast --output models/tableformer-fast.onnx --dataset dataset
   python convert_to_onnx.py --model-name tableformer-accurate --output models/tableformer-accurate.onnx --dataset dataset
   ```
2. **Ottimizzazione**
   ```bash
   python optimize_onnx.py --input models/tableformer-fast.onnx --output models/tableformer-fast-optimized.onnx
   python optimize_onnx.py --input models/tableformer-accurate.onnx --output models/tableformer-accurate-optimized.onnx
   ```
3. **Formato ORT**
   ```bash
   python scripts/convert_to_ort.py --input models/tableformer-fast-optimized.onnx --output models/tableformer-fast.ort
   python scripts/convert_to_ort.py --input models/tableformer-accurate-optimized.onnx --output models/tableformer-accurate.ort
   ```
4. **Conversione in OpenVINO**
   ```bash
   python scripts/convert_to_openvino.py --input models/tableformer-fast.onnx --output-dir models/ov-ir-fast
   python scripts/convert_to_openvino.py --input models/tableformer-accurate.onnx --output-dir models/ov-ir-accurate
   ```

## Benchmark
Gli script di benchmark generano cartelle `results/<variant>/run-YYYYMMDD-HHMMSS/` con:
- `timings.csv` – latenza per immagine
- `summary.json` – statistiche aggregate (media, mediana, p95)
- `model_info.json` – percorso, dimensione, precisione del modello
- `env.json`, `config.json`, `manifest.json`, `logs.txt` – contesto di esecuzione

Esempio di benchmark Python per la variante fast:
```bash
python scripts/bench_python.py --model models/tableformer-fast-optimized.onnx \
  --images ./dataset --variant-name onnx-fast-fp32-cpu \
  --sequential --threads-intra 0 --threads-inter 1 --target-h 448 --target-w 448
```

## SDK .NET 9
La libreria `TableFormerSdk` fornisce un'API modulare per orchestrare i diversi runtime supportati:

- **ONNX Runtime** (`.onnx`)
- **ONNX Runtime** con grafi ottimizzati (`.ort`)
- **OpenVINO CSharp API** (`.xml`/`.bin`)

La soluzione è stata migrata a .NET 9 per sfruttare i miglioramenti del runtime e delle librerie di base. Tutti i progetti sono ora allineati con le ultime versioni stabili dei pacchetti NuGet (ONNX Runtime 1.22.1, binding OpenVINO per ORT 1.22.0, SkiaSharp 3.119.0, suite xUnit 2.9.x) per garantire compatibilità e patch di sicurezza aggiornate.

### Architettura della libreria
Il refactoring suddivide le responsabilità principali in componenti dedicate:

- `TableFormerSdk` coordina la pipeline di inferenza, applica la validazione degli input e restituisce `TableStructureResult`.
- `TableFormerSdkOptions` gestisce lingua, visualizzazione e telemetria, delegando il reperimento dei modelli all'interfaccia `ITableFormerModelCatalog` (per default `ReleaseModelCatalog`).
- `BackendRegistry` gestisce il ciclo di vita dei backend concreti, istanziati tramite `DefaultBackendFactory` a partire dalle opzioni fornite.
- Le classi in `TableFormerSdk.Constants` centralizzano messaggi e valori di default (colore/ampiezza degli overlay).
- `OverlayRenderer` incapsula la generazione della bitmap con le bounding box.

### Enumerazioni ufficiali
- `TableFormerRuntime` – runtime di esecuzione (`Auto`, ONNX, ORT, OpenVINO).
- `TableFormerModelVariant` – variante del modello (*Fast* / *Accurate*).
- `TableFormerLanguage` – lingua dell'input gestita dalla configurazione (English, Italian, French, German, Spanish, Portuguese, Chinese, Japanese).

### Ottimizzazione basata sui tempi di esecuzione
La release introduce un sottosistema di ottimizzazione che misura automaticamente le latenze di inferenza di ogni backend e sceglie la configurazione più veloce per ciascuna variante del modello. Il driver è il nuovo `TableFormerPerformanceAdvisor`, alimentato da una finestra scorrevole di metriche (`TableFormerPerformanceOptions.SlidingWindowSize`) e da soglie configurabili (`MinimumSamples`).

- La modalità `TableFormerRuntime.Auto` effettua dapprima un'esplorazione dei runtime disponibili (rispettando l'ordine preferenziale definito in `TableFormerPerformanceOptions.RuntimePriority`) per raccogliere un numero minimo di campioni.
- Una volta popolata la finestra di osservazione, l'SDK sceglie sempre il runtime con la latenza media più bassa, aggiornando l'informazione a ogni chiamata.
- I risultati sono esposti tramite `TableStructureResult.PerformanceSnapshot` e dagli helper `TableFormerSdk.GetPerformanceSnapshots` / `GetLatestSnapshot`, utili per esporre dashboard o telemetrie enterprise.

### Configurazione e uso
```csharp
using SkiaSharp;
using TableFormerSdk;
using TableFormerSdk.Configuration;
using TableFormerSdk.Enums;
using TableFormerSdk.Models;
using TableFormerSdk.Performance;

var options = new TableFormerSdkOptions(
    modelCatalog: ReleaseModelCatalog.CreateDefault(),
    supportedLanguages: new[]
    {
        TableFormerLanguage.English,
        TableFormerLanguage.Italian
    },
    visualizationOptions: new TableVisualizationOptions(
        strokeColor: SKColors.DeepSkyBlue,
        strokeWidth: 3f),
    performanceOptions: new TableFormerPerformanceOptions(
        enableAdaptiveRuntimeSelection: true,
        minimumSamples: 2,
        runtimePriority: new[]
        {
            TableFormerRuntime.OpenVino,
            TableFormerRuntime.Ort,
            TableFormerRuntime.Onnx
        }));

using var sdk = new TableFormerSdk(options);

var result = sdk.Process(
    imagePath: "dataset/gazette_de_france.jpg",
    overlay: true,
    runtime: TableFormerRuntime.Auto,
    variant: TableFormerModelVariant.Fast,
    language: TableFormerLanguage.Italian);

Console.WriteLine($"Detected regions: {result.Regions.Count}");
Console.WriteLine($"Language: {result.Language}");
Console.WriteLine($"Runtime: {result.Runtime} Avg={result.PerformanceSnapshot.AverageLatencyMilliseconds:F2}ms");

result.OverlayImage?.Encode(SKEncodedImageFormat.Png, 90)
    .SaveTo(File.OpenWrite("overlay.png"));
```
> Il catalogo (`ReleaseModelCatalog`) risolve automaticamente gli artifact pubblicati nella release (`models/heron-optimized.onnx`, `models/heron-optimized.ort`, `models/ov-ir-fp16/heron-optimized.xml`/`.bin`), quindi l'applicazione .NET deve solo scegliere il runtime da utilizzare.

I pacchetti NuGet di riferimento includono `Microsoft.ML.OnnxRuntime`, `OpenVINO.CSharp.API`, `OpenVINO.runtime.ubuntu.24-x86_64` e `SkiaSharp`.

### Benchmark runtime TableFormer (.NET)
Abbiamo eseguito l'applicazione `TableFormerSdk.Benchmarks` su due immagini del dataset FinTabNet (cartella `dataset/FinTabNet/benchmark`) effettuando un warm-up e cinque inferenze utili per immagine su ciascun backend: ONNX Runtime con grafi FP32, il nuovo backend ORT e l'IR OpenVINO pubblicato nella release.【031f70†L1-L1】【cf0f34†L1-L16】【40b16d†L1-L8】

```bash
# ONNX Runtime (modello ONNX FP32)
dotnet run --project dotnet/TableFormerSdk.Benchmarks/TableFormerSdk.Benchmarks.csproj -- \
  --engine Onnx --variant-name Fast \
  --images dataset/FinTabNet/benchmark --output results/benchmarks \
  --runs-per-image 5 --warmup 1 --target-h 448 --target-w 448

# ONNX Runtime (formato ORT)
dotnet run --project dotnet/TableFormerSdk.Benchmarks/TableFormerSdk.Benchmarks.csproj -- \
  --engine Ort --variant-name Fast \
  --images dataset/FinTabNet/benchmark --output results/benchmarks \
  --runs-per-image 5 --warmup 1 --target-h 448 --target-w 448

# OpenVINO (IR FP16 nativo)
dotnet run --project dotnet/TableFormerSdk.Benchmarks/TableFormerSdk.Benchmarks.csproj -- \
  --engine OpenVino --variant-name Fast \
  --images dataset/FinTabNet/benchmark --output results/benchmarks \
  --runs-per-image 5 --warmup 1 --target-h 640 --target-w 640
```
Se gli artifact della release non si trovano nella cartella `models/` accanto agli eseguibili, passare `--models-root <percorso>` per indicare la directory che li contiene.

Le statistiche aggregate (millisecondi) calcolate sui dieci campioni utili sono sintetizzate nella tabella seguente; le directory `summary.json` e `timings.csv` prodotte dall'applicazione riportano i dettagli completi.

| Runtime | Risoluzione input | Media | Mediana | p95 |
| --- | --- | ---: | ---: | ---: |
| ONNX Runtime (`.onnx`) | 448×448 | 41,02 | 36,37 | 75,38 |
| ONNX Runtime (`.ort`) | 448×448 | 28,86 | 13,98 | 72,38 |
| OpenVINO FP16 (`.xml/.bin`) | 640×640 | 6,18 | 5,83 | 7,85 |

I valori derivano dai file `summary.json` generati in `results/benchmarks/Fast`.【0d42e5†L1-L6】【0c3176†L1-L6】【e6f227†L1-L6】 La console di benchmark registra il runtime effettivo in `timings.csv`, mantenendo allineate le misurazioni con i dati restituiti da `TableStructureResult.InferenceTime`.
### Test e benchmark .NET
- **Test unitari**: `dotnet test TableFormerSdk.sln`
- **Benchmark**: l'applicazione console `TableFormerSdk.Benchmarks` replica gli script Python e salva gli stessi artifact.

```bash
dotnet run --project dotnet/TableFormerSdk.Benchmarks/TableFormerSdk.Benchmarks.csproj -- \
  --engine Onnx --variant-name Fast \
  --images dataset/FinTabNet/benchmark --output results/benchmarks \
  --runs-per-image 5 --warmup 1 --target-h 640 --target-w 640
```

## Requisiti
Installare le dipendenze principali:
```bash
pip install -r requirements.txt
```
I modelli di grandi dimensioni vengono salvati nella cartella `models/` ed esclusi dal versionamento git.

## Pacchetto NuGet con i modelli ufficiali

La libreria `TableFormerSdk` è pubblicata come pacchetto NuGet (`TableFormerSdk` versione `1.0.0`) allegato alla release [v1.0.0](https://github.com/mapo80/ds4sd-docling-tableformer-onnx/releases/tag/v1.0.0). Il `.nupkg` contiene **tutti** gli artifact distribuiti nella release (encoder, decoder e bbox decoder per le varianti *fast* e *accurate*, sia ONNX sia OpenVINO) all'interno di `contentFiles/any/any/models/`.

### Download degli asset della release

```bash
export GITHUB_TOKEN=<token_con_permessi_repo>
python scripts/download_release_assets.py \
  --repo mapo80/ds4sd-docling-tableformer-onnx \
  --tag v1.0.0 \
  --output dotnet/TableFormerSdk/ReleaseModels \
  --skip-existing
```

Lo script verifica dimensione e integrità dei file, salvandoli in `dotnet/TableFormerSdk/ReleaseModels` (cartella ignorata da Git ma confezionata durante il `dotnet pack`).

### Creazione del pacchetto

```bash
dotnet pack dotnet/TableFormerSdk/TableFormerSdk.csproj -c Release
```

L'output viene salvato in `artifacts/nuget/TableFormerSdk.1.0.0.nupkg`.

### Pubblicazione sulla release GitHub

```bash
curl -H "Authorization: token $GITHUB_TOKEN" \
     -H "Content-Type: application/octet-stream" \
     --data-binary @artifacts/nuget/TableFormerSdk.1.0.0.nupkg \
     "https://uploads.github.com/repos/mapo80/ds4sd-docling-tableformer-onnx/releases/248661306/assets?name=TableFormerSdk.1.0.0.nupkg"
```

### Progetto di esempio

Il progetto `dotnet/TableFormerSdk.ReleaseSample` utilizza il pacchetto locale per verificare la disponibilità dei modelli senza download aggiuntivi:

```bash
dotnet add dotnet/TableFormerSdk.ReleaseSample/TableFormerSdk.ReleaseSample.csproj \
  package TableFormerSdk --version 1.0.0 --source artifacts/nuget

dotnet run --project dotnet/TableFormerSdk.ReleaseSample/TableFormerSdk.ReleaseSample.csproj
```

L'output elenca i modelli disponibili per ogni runtime, confermando che i file sono già inclusi nel pacchetto NuGet.
