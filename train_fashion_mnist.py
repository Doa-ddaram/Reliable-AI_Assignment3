import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import onnx
from onnxsim import simplify

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleMLP().to(device)
    
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    print("Training SimpleMLP for 5 Epoch...")
    model.train()
    for epoch in range(5):
        epoch_loss = 0
        epoch_accuracy = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_accuracy += (output.argmax(dim=1) == target).sum().item()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} - Loss: {epoch_loss/len(train_loader):.4f}, Accuracy: {epoch_accuracy/len(train_loader.dataset)*100:.2f}%")
        
            
    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    onnx_path = 'custom_fashion_mnist.onnx'
    
    # 1. save ONNX format 
    torch.onnx.export(model, dummy_input, onnx_path,
                      export_params=True,
                      opset_version=14,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'])
    print(f"Exported to {onnx_path}")
    
    # 2. Use ONNX Simplifier for Marabou compatibility
    onnx_model = onnx.load(onnx_path)
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    
    sim_path = 'custom_fashion_mnist_sim.onnx'
    onnx.save(model_simp, sim_path)
    print(f"Simplified model ready for Marabou saved to {sim_path}")

if __name__ == '__main__':
    main()
