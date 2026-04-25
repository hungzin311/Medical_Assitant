import logging
import cv2 
import segmentation_models_pytorch as smp
import numpy as np
from PIL import Image
import imageio
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms import Resize, InterpolationMode
from torchvision import transforms
from collections import OrderedDict
from scipy import ndimage

class PolypSegmentation: 
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        self.model = self._load_model()

    def _load_model(self):
        """Load the trained DeepLabV3+ model."""
        model = smp.DeepLabV3Plus(
            encoder_name="resnet50",        
            encoder_weights="imagenet",     
            in_channels=3,                  
            classes=3     
        )

        ### Load pretrained model 
        checkpoint = torch.load(self.model_path, map_location = torch.device(self.device))

        new_state_dict = OrderedDict()
        for k, v in checkpoint['model'].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v

        # load params
        model.load_state_dict(new_state_dict)
        model.to(self.device)
        model.eval()
        
        logging.info("Model loaded successfully DeeplabV3Plus Segmentation")
        return model 

    def transform_image(self, image_path):
        """Transform the image to the model input size."""
        transform = transforms.Compose([Resize((512, 512), interpolation=InterpolationMode.BILINEAR),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        image = Image.open(image_path).convert('RGB')
        image = transform(image) 
        return image
    
    def predict(self, image_path,output_path):
        """Predict the polyp segmentation."""
        image = self.transform_image(image_path)
        
        img_tensor = torch.Tensor(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            generated_mask = self.model(img_tensor)
            generated_mask = F.one_hot(torch.argmax(generated_mask[0], 0), num_classes = 3).cpu().float()
        #save mask
        self.save_image(generated_mask, output_path)
        #post processing mask 
        self.process_regions(output_path)

        self.overlay_mask(image_path, output_path, output_path)

    
    def overlay_mask(self, image_path, mask_path, output_path):
        """Overlay only the non-blue (non-background) mask on the image."""

        # Load and resize both images
        image = Image.open(image_path).resize((512, 512)).convert("RGB")
        mask = Image.open(mask_path).resize((512, 512)).convert("RGB")

        # Convert to numpy arrays
        image_np = np.array(image).astype(np.uint8)
        mask_np = np.array(mask).astype(np.uint8)


        blue_threshold = 10  # tolerance for blue matching
        blue_like = (
            (mask_np[:, :, 0] <= blue_threshold) &  # R close to 0
            (mask_np[:, :, 1] <= blue_threshold) &  # G close to 0
            (mask_np[:, :, 2] >= 255 - blue_threshold)  # B close to 255
        )
        mask_np[blue_like] = [0, 0, 255]

        # Create a boolean mask: True where the mask is NOT blue
        not_blue_mask = ~(np.all(mask_np == ([0, 0, 255]), axis=-1))
        
        # Make a copy of the original image
        overlaid = image_np.copy()

        # Blend only non-blue pixels (e.g., red region) with original image
        alpha = 0.5
        overlaid[not_blue_mask] = (
            alpha * image_np[not_blue_mask] + (1 - alpha) * mask_np[not_blue_mask]
        ).astype(np.uint8)

        self.save_image(overlaid, output_path)
        logging.info(f"save image success: {output_path}")

        return True

    
    def save_image(self, image, output_path):
        """Save the image as uint8."""
        if image.dtype == np.float32 or image.dtype == torch.float32:
            image = image.numpy() if isinstance(image, Tensor) else image
            image = (image * 255).clip(0, 255).astype(np.uint8)
        imageio.imwrite(output_path, image)

    def process_regions(self, mask_path):
        image = cv2.imread(mask_path)
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for blue background
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([140, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Create masks for red and green
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Get non-blue regions
        non_blue_mask = cv2.bitwise_not(blue_mask)
        
        # Label connected components in non-blue regions
        labels, num_labels = ndimage.label(non_blue_mask)
        
        # Create result image
        result = np.zeros_like(image)
        result[blue_mask > 0] = ([255, 0, 0])  # Set blue background
        
        # Process each labeled region
        for label in range(1, num_labels + 1):
            region_mask = (labels == label)
            
            # Count red and green pixels in this region
            red_count = cv2.countNonZero(cv2.bitwise_and(red_mask, region_mask.astype(np.uint8)))
            green_count = cv2.countNonZero(cv2.bitwise_and(green_mask, region_mask.astype(np.uint8)))
            
            # If only one color is present, use that color
            if red_count > 0 and green_count == 0:
                result[region_mask] = [0, 0, 255]  # Pure red
            elif green_count > 0 and red_count == 0:
                result[region_mask] = [0, 255, 0]  # Pure green
            # If mixed colors, use the dominant one
            else:
                if red_count > green_count:
                    result[region_mask] = [0, 0, 255]  # Red
                else:
                    result[region_mask] = [0, 255, 0]  # Green
        
        cv2.imwrite(mask_path, result)


# if __name__ == "__main__":
#     polyp_segmentation = PolypSegmentation(model_path = "agents/image_analysis_agent/polyp_seg_tool/models/deeplabv3_resnet50.pth")
#     polyp_segmentation.predict(image_path = "image2.jpeg", output_path = "image_output.png")