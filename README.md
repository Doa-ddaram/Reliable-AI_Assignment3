# Reliable-AI_Assignment3

## 실행 환경

- OS: Linux
- Python: conda 환경 `marabou`

## 환경 설정

처음 사용하는 경우 conda 환경을 만들고 필수 패키지를 설치합니다.

```bash
conda create -n marabou python=3.11
conda activate marabou
```

```bash
cd Reliable-AI_Assignment3
pip install -r requirement.txt
```

필요 패키지가 누락되었다는 오류가 나오면 아래를 추가로 설치합니다.

```bash
pip install torch torchvision onnx onnxsim tqdm argparse
```

## 실행 절차

아래 순서대로 실행하면 FashionMNIST 2-layer MLP 학습, ONNX 내보내기/단순화, 그리고 Marabou 검증을 수행할 수 있습니다.

### 0) Repository clone

```bash
git clone https://github.com/Doa-ddaram/Reliable-AI_Assignment3.git
```

### 1) conda 환경 활성화

```bash
conda activate marabou
```

### 2) 학습 및 ONNX 내보내기

```bash
cd Reliable-AI_Assignment3
python train_fashion_mnist.py
```

정상 실행 시 다음과 유사한 로그가 출력됩니다.

```plaintext
Training Two layer MLP for 5 Epoch...
Epoch 1 - Loss: 0.5833, Accuracy: 79.72%
Epoch 2 - Loss: 0.4231, Accuracy: 85.19%
Epoch 3 - Loss: 0.3938, Accuracy: 86.10%
Epoch 4 - Loss: 0.3648, Accuracy: 86.93%
Epoch 5 - Loss: 0.3496, Accuracy: 87.42%
Exported to output/custom_fashion_mnist.onnx
Simplified model ready for Marabou saved to output/custom_fashion_mnist_sim.onnx
```

### 3) Marabou 검증 실행

```bash
python test.py --epsilon 0.001
```

예시 출력:

```plaintext
Loading model: onnx/custom_fashion_mnist_sim.onnx
Loaded image index 0. True label: 9
Setting Query -> Epsilon: 0.001, Target Label: 0
Solving targeted robustness against class 0...

unsat

[Result: UNSAT] -> The model IS robust against class 0.
Verified: No inputs within ||x' - x||_inf <= 0.001 are classified as 0.

Verification time (sec): 0.0321
Peak RSS (KB): 713596
```

## 인자 설정 (argparse)

`test.py`는 argparse로 다음 값을 입력받을 수 있습니다.

- `--epsilon`: 검증에 사용할 $\epsilon$
- `--model-path`: ONNX 모델 경로
- `--output-path`: Marabou 결과 저장 경로

예시:

```bash
python test.py --epsilon 0.001 --model-path onnx/custom_fashion_mnist_sim.onnx --output-path output/marabou_output.txt
```

### Experiment Result
실행 결과는 입력값에 따라 `unsat` 또는 `sat`이 출력됩니다.
