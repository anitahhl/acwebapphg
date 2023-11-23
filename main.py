from typing import Optional
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from io import BytesIO
import base64
import style

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
    
    
@app.post("/")
async def transfer(request: Request, style_index: str = Form(...), file: UploadFile = File(...)):
    result = None
    error = None
    
    extension = file.filename.split(".")[-1]
    if extension in ("jpg", "jpeg", "png"):
        try:
            with Image.open(file.file) as img:
                img = ImageOps.exif_transpose(img)
            temp_file = "./static/images/content-images/content.jpg"
            img.save(temp_file)
            
        except Exception as ex:
            error = ex
        finally:
            file.file.close()
            
    elif len(extension)>0:
        error = "Image must be jpg or png format!"
    else:
        temp_file = "./static/images/content-images/mountain.jpg"

    if not error:
        try: 
            result = style.get_result(temp_file, int(style_index))
        except Exception as ex:
            error = ex
        
    return templates.TemplateResponse("index.html", {"request": request, "result": result , "error": error})


if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", port="8000", reload=True)
