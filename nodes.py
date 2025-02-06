from transformers import AutoModelForImageSegmentation  
import torch  
from torchvision import transforms  
import numpy as np  
from PIL import Image  
import torch.nn.functional as F  
import os  
import cv2  

torch.set_float32_matmul_precision(["high", "highest"][0])  

current_path = os.getcwd()  
models_path = os.path.join(current_path, "models", "BiRefNet")  

# 模型配置  
MODEL_VERSIONS = {  
    'BiRefNet': ('BiRefNet', (1024, 1024)),  
    'BiRefNet_HR': ('BiRefNet_HR', (2048, 2048)),  
    'BiRefNet_lite': ('BiRefNet_lite', (1024, 1024)),  
    'BiRefNet_lite-2K': ('BiRefNet_lite-2K', (2560, 1440)),  
    'BiRefNet_512x512': ('BiRefNet_512x512', (512, 512)),  
    'BiRefNet-matting': ('BiRefNet-matting', (1024, 1024)),  
    'BiRefNet-portrait': ('BiRefNet-portrait', (1024, 1024)),  
    'BiRefNet-DIS5K': ('BiRefNet-DIS5K', (1024, 1024)),  
    'BiRefNet-HRSOD': ('BiRefNet-HRSOD', (1024, 1024)),  
    'BiRefNet-COD': ('BiRefNet-COD', (1024, 1024)),  
    'BiRefNet-DIS5K-TR_TEs': ('BiRefNet-DIS5K-TR_TEs', (1024, 1024)),  
    'BiRefNet-legacy': ('BiRefNet-legacy', (1024, 1024))  
}  

def get_device_by_name(device):  
    """  
    根据名称获取设备  
    """  
    if device == 'auto':  
        try:  
            device = "cpu"  
            if torch.cuda.is_available():  
                device = "cuda"  
            elif torch.backends.mps.is_available():  
                device = "mps"  
            elif torch.xpu.is_available():  
                device = "xpu"  
        except:  
            raise AttributeError("What's your device(到底用什么设备跑的)？")  
    print("\033[93mUse Device(使用设备):", device, "\033[0m")  
    return device  
def get_model_path(model_name):  
    return os.path.join(models_path, model_name)  

class ImagePreprocessor:  
    def __init__(self, resolution):  
        self.transform = transforms.Compose([  
            transforms.Resize(resolution),  
            transforms.ToTensor(),  
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  
        ])  

    def __call__(self, image):  
        return self.transform(image)  

# 模型加载节点  
class BiRefNet_Loader:  
    def __init__(self):  
        self.loaded_model = None  
        
    @classmethod  
    def INPUT_TYPES(cls):  
        return {  
            "required": {  
                "model_version": (list(MODEL_VERSIONS.keys()), {"default": "General-HR"}),  
                "device": (["auto", "cuda", "cpu", "mps", "xpu", "meta"], {"default": "auto"})  
            }  
        }  

    RETURN_TYPES = ("BIREFNET_MODEL",)  
    RETURN_NAMES = ("model",)  
    FUNCTION = "load_model"  
    CATEGORY = "BiRefNet🌟"  

    

    def load_model(self, model_version, device):  
        device = get_device_by_name(device)  
        model_name, resolution = MODEL_VERSIONS[model_version]  
        local_model_path = get_model_path(model_name)  
        
        # 首先尝试加载本地模型  
        try:  
            if os.path.exists(local_model_path):  
                print(f"\033[92mLoading local model from: {local_model_path}\033[0m")  
                model = AutoModelForImageSegmentation.from_pretrained(  
                    local_model_path,  
                    trust_remote_code=True  
                )  
            else:  
                print(f"\033[93mLocal model not found, downloading from HuggingFace: {model_name}\033[0m")  
                model = AutoModelForImageSegmentation.from_pretrained(  
                    f"ZhengPeng7/{model_name}",  
                    trust_remote_code=True,  
                    cache_dir=local_model_path  
                )  
        except Exception as e:  
            print(f"\033[91mError loading local model: {str(e)}\033[0m")  
            print("\033[93mFallback to downloading from HuggingFace\033[0m")  
            try:  
                model = AutoModelForImageSegmentation.from_pretrained(  
                    f"ZhengPeng7/{model_name}",  
                    trust_remote_code=True,  
                    cache_dir=local_model_path  
                )  
            except Exception as download_error:  
                raise RuntimeError(f"Failed to load model both locally and from HuggingFace: {str(download_error)}")  

        model.to(device)  
        model.eval()  
        if device == "cuda":  
            model.half()  

        return ({  
            "model": model,  
            "resolution": resolution,  
            "device": device,  
            "half_precision": (device == "cuda")  
        },)  

# 推理节点  
class BiRefNet_Remove_Background:  
    @classmethod  
    def INPUT_TYPES(cls):  
        return {  
            "required": {  
                "image": ("IMAGE",),  
                "model": ("BIREFNET_MODEL",),  
                "background_color": (["transparency"] + ["white", "black", "green", "blue", "red"], {"default": "transparency"}),  
            }  
        }  

    RETURN_TYPES = ("IMAGE", "MASK")  
    RETURN_NAMES = ("image", "mask")  
    FUNCTION = "inference"  
    CATEGORY = "BiRefNet🌟"  

    def inference(self, image, model, background_color):  
        model_data = model  
        model = model_data["model"]  
        device = model_data["device"]  
        use_half_precision = model_data["half_precision"]  
        resolution = model_data["resolution"]  # 直接使用模型的最佳分辨率  

        preprocessor = ImagePreprocessor(resolution)  
        processed_images = []  
        processed_masks = []  

        for img in image:  
            # 转换为PIL图像  
            orig_image = Image.fromarray(np.clip(255. * img.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))  
            w, h = orig_image.size  
            
            # 预处理  
            image_tensor = preprocessor(orig_image.convert('RGB')).unsqueeze(0)  
            if use_half_precision:  
                image_tensor = image_tensor.half()  
            image_tensor = image_tensor.to(device)  

            # 推理  
            with torch.no_grad():  
                pred = model(image_tensor)[-1].sigmoid().cpu()  
            
            # 后处理  
            pred = torch.squeeze(F.interpolate(pred, size=(h, w)))  
            pred = (pred - pred.min()) / (pred.max() - pred.min())  
            mask = Image.fromarray((pred * 255).numpy().astype(np.uint8))  
            
            # 设置背景  
            if background_color == "transparency":  
                result_image = Image.new("RGBA", (w, h), (0, 0, 0, 0))  
            else:  
                result_image = Image.new("RGB", (w, h), background_color)  
            
            result_image.paste(orig_image, mask=mask)  
            
            # 转换回tensor  
            processed_images.append(torch.from_numpy(np.array(result_image).astype(np.float32) / 255.0).unsqueeze(0))  
            processed_masks.append(torch.from_numpy(np.array(mask).astype(np.float32) / 255.0).unsqueeze(0))  

        return torch.cat(processed_images, dim=0), torch.cat(processed_masks, dim=0)  

NODE_CLASS_MAPPINGS = {  
    "BiRefNet_Loader": BiRefNet_Loader,  
    "BiRefNet_Remove_Background": BiRefNet_Remove_Background
}  

NODE_DISPLAY_NAME_MAPPINGS = {  
    "BiRefNet_Loader": "BiRefNet Loader🌟",  
    "BiRefNet_Remove_Background": "BiRefNet Remove Background🌟"  
}