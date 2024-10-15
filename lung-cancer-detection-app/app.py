import torch
from torchvision import transforms
from PIL import Image
from train import model, reverse_class_dict
import gradio as gr

model.load_state_dict(torch.load('model.pth', weights_only=True))

model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def cancer_detection(image):
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)

    _, predicted = torch.max(output, 1)

    predicted_label = reverse_class_dict[predicted.item()]
    return predicted_label


#  Gradio interface
image = gr.Image(type="pil", label="Upload the CT Scan Image:")

iface = gr.Interface(
    fn=cancer_detection,
    inputs=[image],
    outputs="text",
    title="Lung Cancer Detection"
)

iface.launch()