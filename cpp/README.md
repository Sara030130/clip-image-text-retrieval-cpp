# C++ ONNX Inference

## Build

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/opt/homebrew
cmake --build build -j4
```

## Run image mode

```bash
./build/clip_infer ../checkpoints/clip_image_encoder.onnx path/to/image.jpg output.bin ../data/text_index
```

## Run tensor mode

```bash
./build/clip_infer --tensor ../checkpoints/clip_image_encoder.onnx input.bin output.bin
```
