import sys
import glob
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
from model.model import LogisticRegression 
from configs.config import Configuration

# take the first argument as the config file path
config_file_path = sys.argv[1]
config = Configuration(config_file_path)
model = LogisticRegression(config.input_dim, config.output_dim, config.hidden_layers)

app = FastAPI()

# Serve the static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# find_latest_model as model_epoch_{max_epoch}.pth
# the model is on basepath of config file
model_files = glob.glob(os.path.join(os.path.dirname(config_file_path), "model_epoch_*.pth"))

latest_model_file = max(model_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
model.load_state_dict(torch.load(latest_model_file))
model.eval()

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("static/index.html") as f:
        return f.read()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        # transform the image to the grayscale and resize to 28x28
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        image = transform(image).unsqueeze(0)

        print(image.shape)
        
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
        
        return {"prediction": predicted.item()}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn

    # have the run automatically reload
    uvicorn.run(app, host="0.0.0.0", port=8000)