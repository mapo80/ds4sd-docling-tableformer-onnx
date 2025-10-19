# Stato verifiche TableFormer TorchSharp

Questo documento traccia lo stato di avanzamento della parità Python/.NET per la pipeline di TableFormer. Ogni sezione descrive
un componente, come riprodurre la verifica e l'esito corrente. **Non è consentito procedere alla fase successiva finché la
verifica in corso non produce risultati identici tra Python e .NET.**

## 1. Bootstrap artefatti e configurazione
- **Descrizione**: il bootstrap scarica gli asset Hugging Face, normalizza `tm_config.json` e replica la logica Docling di patch
dei percorsi. L'implementazione .NET corrispondente vive in
`Artifacts/TableFormerArtifactBootstrapper.cs` e `Configuration/TableFormerConfigLoader.cs`.
- **Come verificare**:
  1. Rigenera il riferimento Python:  
     `PYTHONPATH=tableformer-docling python scripts/hash_tableformer_config.py --variant fast --output results/tableformer_config_fast_hash.json`
  2. Esegui il test di parità:  
     `dotnet test TableFormerSdk.sln --filter TableFormerTorchSharpInitializationTests.PredictorInitializationMatchesPythonReference`
  3. Il test confronta l'hash SHA-256 del JSON canonico calcolato da Python con il valore prodotto da .NET.
- **Risultato corrente**: ✅ Hash configurazione combacia (`63b1bab86ccb4d78f475d60674d61c5eafe4c5702c9f1d930422a92f0b4da5bf`).

## 2. Inizializzazione di `TFPredictor`
- **Descrizione**: l'inizializzazione carica la word map, costruisce le mappe inverse e calcola i checksum delle tabelle
`safetensors`. Il codice si trova in `Initialization/TableFormerPredictorInitializer.cs`, supportato dai DTO in
`Initialization/TableFormerInitializationReference.cs` e dal parser in `Safetensors/SafeTensorFile.cs`.
- **Come verificare**:
  1. Esporta il riferimento Python aggiornato:  
     `PYTHONPATH=tableformer-docling python scripts/export_tableformer_init_reference.py --variant fast --output results/tableformer_init_fast_reference.json`
  2. Avvia i test .NET come nello step 1 (lo stesso test ricopre anche questa fase perché confronta word map e tensori).
  3. Il test valida la canonicalizzazione del dizionario, la mappa inversa e la corrispondenza dei digest per ogni tensore.
- **Risultato corrente**: ✅ Parità confermata. Il digest della word map (`6d1147d6fcda9c0434601c5acc79d252bc3644f6025ecbedf25017a801d1ffa2`) e i 1,365
checksum dei tensori coincidono tra Python e .NET (vedi `results/tableformer_init_fast_reference.json`).

## 3. Preparazione input pagina
- **Descrizione**: `TableFormerDocling.predict_page` costruisce la struttura `page_input` caricando l'immagine RGB della pagina,
  impostando larghezza/altezza, inizializzando i token OCR e preparando la lista di bounding box delle tabelle in coordinate pagina.
  La controparte .NET è implementata in `PagePreparation/TableFormerPageInputPreparer.cs` e nei DTO definiti in
  `PagePreparation/TableFormerPageInputSnapshot.cs`.
- **Come verificare**:
  1. Rigenera il dump Python con i byte raw delle immagini:
     `PYTHONPATH=tableformer-docling python scripts/export_tableformer_page_input.py --output results/tableformer_page_input_reference.json`
  2. Esegui il test dedicato:
     `dotnet test TableFormerSdk.sln --filter TableFormerTorchSharpPageInputTests.PageInputMatchesPythonReference`
  3. Il test confronta dimensioni pagina, conteggio token, bounding box (tolleranza 1e-6) e verifica byte-per-byte che il buffer RGB
     e l'hash SHA-256 coincidano con quelli generati in Python.
- **Risultato corrente**: ✅ Parità raggiunta su entrambe le pagine FinTabNet di riferimento (`HAL.2004.page_82.pdf_125315.png` e
  `HAL.2004.page_82.pdf_125317.png`).

Le sezioni successive saranno aggiunte solo dopo che gli step precedenti continueranno a soddisfare i test di parità.

## 4. Ridimensionamento pagina e cropping tabella
- **Descrizione**: questa fase replica `resize_img` e il ritaglio delle ROI in `multi_table_predict`, ridimensionando la pagina
  a 1024 pixel di altezza con interpolazione bilineare e ritagliando ogni bounding box (con arrotondamento ai pixel e clamp ai
  bordi) per produrre le immagini di tabella.
- **Come verificare**:
  1. Rigenera i riferimenti Python:
     `PYTHONPATH=tableformer-docling python scripts/export_tableformer_table_crops.py --output results/tableformer_table_crops_reference.json`
  2. Esegui il test .NET dedicato:
     `dotnet test TableFormerSdk.sln --filter TableFormerTorchSharpTableCropTests.TableCroppingMatchesPythonReference`
  3. Il test confronta fattore di scala, dimensioni ridimensionate, bounding box originali/scalati/arrotondati (tolleranza 1e-6)
     e verifica che gli hash SHA-256, la lunghezza byte, la media e la deviazione standard dei pixel delle ROI coincidano con i
     valori esportati da Python.
- **Risultato corrente**: ✅ Parità confermata. Per entrambe le immagini FinTabNet, gli hash ROI principali sono
  `acd45dadb470d038469a38224c7d39bcbf480e4ddbe9ac1eac774f4426da7c8e` e
  `c9c09c78d421082119871147368e0732741887c917bb8dcb6fdbc05f89500fce`; le statistiche (media 229.170441, deviazione standard
  45.118282) sono identiche tra Python e .NET. Non è consentito procedere allo step successivo finché questo controllo non è
  verde.

## 5. Normalizzazione e conversione a tensore
- **Descrizione**: `_prepare_image` normalizza l'immagine di tabella, la ridimensiona a un quadrato 448×448 e genera un tensore
  Torch `[1, 3, 448, 448]` con canali ordinati come in Docling (C, width, height). L'implementazione .NET corrispondente è in
  `Tensorization/TableFormerImageTensorizer.cs`, che usa SkiaSharp per i resize intermedi e TorchSharp per creare il batch.
- **Come verificare**:
  1. Rigenera i riferimenti Python compressi:
     `PYTHONPATH=tableformer-docling python scripts/export_tableformer_image_tensors.py --output results/tableformer_image_tensors_reference.json`
  2. Esegui il test di parità:
     `dotnet test TableFormerSdk.sln --filter TableFormerTorchSharpImageTensorTests.ImageTensorizationMatchesPythonReference`
  3. Il test confronta forma del tensore, digest SHA-256 dei float32, min/max/media/deviazione standard e ogni elemento del batch
     con tolleranza 1e-6.
- **Risultato corrente**: ✅ Parità raggiunta. Entrambe le tabelle producono SHA-256 `acfe8240b75c07268af1b4bb54b976d9aec030b8b8c9f940d20a0595053e16f6`
  e `b06af0e93cd304ae3a0deae05d0443605a7f47cbf4ef68e5d7fb0d8013900d3d`, con media `-0.2463388` e `-0.2454946` e deviazione
  standard `0.9869987` e `0.9853885` identiche in Python e .NET. Come sempre non si procede allo step successivo se questa parità
  non è rispettata.

## 6. Inferenza neurale (encoder, transformer, decoder bbox)
- **Descrizione**: `TableModel04_rs` combina l'encoder ResNet, il trasformatore di tag e il decodificatore di bounding box per
  produrre le sequenze di tag, le logits di classe e le coordinate normalizzate. La controparte .NET è definita in
  `Model/TableFormerNeuralModel.cs` e nei moduli ausiliari in `Model/TableFormerModelModules.cs`.
- **Come verificare**:
  1. Rigenera il riferimento Python:
     `PYTHONPATH=tableformer-docling python scripts/export_tableformer_neural_outputs.py --output results/tableformer_neural_outputs_reference.json`
  2. Esegui il test di parità neurale:
     `dotnet test TableFormerSdk.sln --filter TableFormerTorchSharpNeuralInferenceTests.NeuralInferenceMatchesPythonReference`
  3. Il test richiede uguaglianza esatta per la sequenza di tag e per le forme dei tensori, verifica le statistiche aggregate e
     confronta elemento per elemento logits e coordinate con `torch.allclose` e una tolleranza massima (delta assoluto) pari a
     `1e-5`. Se anche un solo elemento eccede la soglia la verifica fallisce e non si può procedere agli step successivi.
- **Risultato corrente**: ✅ Parità confermata con deviazione massima osservata `5.72e-06`, ben entro la soglia `1e-5`. Le sequenze
  di tag coincidono e gli argmax riga-per-riga risultano identici tra Python e .NET.
