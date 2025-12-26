import torch
from torchvision import transforms, models
from PIL import Image
import joblib
import numpy as np

_image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_image_model(path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device)
    model = models.resnet18(pretrained=False)
    classes = checkpoint.get('classes', None)
    if classes is None:
        raise ValueError('Checkpoint missing classes list.')
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, classes

def predict_image(model, classes, pil_image, device=None, topk=3):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    img_t = _image_transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    idxs = np.argsort(probs)[::-1][:topk]
    return [(classes[i], float(probs[i])) for i in idxs]

def load_text_model(path):
    return joblib.load(path)

def predict_text(pipe, text, topk=3):
    preds = pipe.predict_proba([text])[0]
    classes = pipe.classes_
    idxs = np.argsort(preds)[::-1][:topk]
    return [(classes[i], float(preds[i])) for i in idxs]
