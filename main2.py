from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError
import torch
from torchvision import transforms
import io
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class Pict_cl_1(torch.nn.Module):
    def __init__(self, inp, out, hidden_units):
        super().__init__()
        self.conv_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=inp, out_channels=hidden_units,
                            kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units,
                            kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(kernel_size=2))
        self.conv_2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=hidden_units,
                            out_channels=hidden_units, kernel_size=3,
                            stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=hidden_units,
                            out_channels=hidden_units, kernel_size=3,
                            stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2))
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=hidden_units*16*16,
                            out_features=out),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.classifier(x)
        return x


def load_model(model_path, model_class, input_channels, num_classes, hidden_units):
    model = model_class(inp=input_channels, out=num_classes, hidden_units=hidden_units)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def process_image(image):
    transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    return image

def predict(image, model, class_names):
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return class_names[predicted.item()]


model_path = 'model.pth'
input_channels = 3
num_classes = 4
hidden_units = 15
model = load_model(model_path, Pict_cl_1, input_channels, num_classes, hidden_units)
class_names = ['Dark', 'Green', 'Light', 'Medium']


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
    except UnidentifiedImageError:
        # Возвращаем пользователя на страницу загрузки с сообщением об ошибке
        return templates.TemplateResponse("upload.html", {
            "request": request,
            "error_message": "Загруженный файл не является изображением. Пожалуйста, загрузите изображение."
        })

    processed_image = process_image(image)

    predicted_class = predict(processed_image, model, class_names)

    upload_dir = "static/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    image_path = os.path.join(upload_dir, file.filename)
    image.save(image_path)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "predicted_class": predicted_class,
        "image_url": f"/static/uploads/{file.filename}"
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.2", port=8000)