#!/bin/bash
set -e

# PaddlePaddle GPU (CUDA 12.6 빌드)
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# PaddleOCR + 문서 파서 확장
python -m pip install -U "paddleocr[doc-parser]"

# safetensors (nightly wheel URL 직접 지정)
python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl
