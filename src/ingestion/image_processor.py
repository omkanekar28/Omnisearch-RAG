import os
import time
import base64
from abc import ABC, abstractmethod
from io import BytesIO
from PIL import Image, ImageFile
from langchain_core.messages import SystemMessage, HumanMessage
from ..llm.ollama_handler import OllamaHandler
from ..utils.logger_setup import setup_logger

logger = setup_logger("image_processor.py")


class BaseImageProcessor(ABC):
    """Abstract base class for Image processing"""

    def __init__(self, model_handler: OllamaHandler) -> None:
        """Initializes the valid image extensions"""
        self.model_handler = model_handler
        self.valid_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"]

    def validate_file_location(self, filepath: str) -> None:
        """Check if given file exists"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Provided file '{filepath}' does not exist!")
    
    def validate_file_type(self, filepath: str) -> None:
        """Check if given file is a valid image"""
        if not any(filepath.lower().endswith(ext) for ext in self.valid_extensions):
            raise ValueError(f"Provided file '{filepath}' is not a valid image file!")
    
    def validate_image_file(self, filepath: str) -> None:
        """Validates if the given file exists and is an image"""
        self.validate_file_location(filepath)
        self.validate_file_type(filepath)

    def convert_to_base64(self, pil_image: ImageFile.ImageFile) -> str:
        """
        Convert PIL images to Base64 encoded strings

        :param pil_image: PIL image
        :return: Re-sized Base64 string
        """
        if pil_image.mode == "RGBA":
            pil_image = pil_image.convert("RGB")

        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    @abstractmethod
    def process(self, image_filepath: str) -> str:
        """Processes the given image and returns found text content"""
        pass


class OllamaImageProcessor(BaseImageProcessor):
    """Image processor implementation using Ollama vision models"""

    def __init__(
            self,
            ollama_model_handler: OllamaHandler,
            system_prompt: str,
    ) -> None:
        """Initializes the parent and child class parameters"""
        super().__init__(ollama_model_handler)
        self.system_prompt = system_prompt
    
    def process(self, image_filepath: str) -> str:
        """
        Analyzes the given image and returns its textual content.

        Behavior:
            - If the image predominantly contains text, returns the extracted text as-is.
            - Otherwise, returns a descriptive caption summarizing the image content.

        Args:
            image_filepath (str): Path to the input image file.

        Returns:
            str: Extracted text or a generated description of the image.
        """
        self.validate_image_file(image_filepath)
        processing_start_time = time.time()
        try:
            with Image.open(image_filepath) as pil_image:
                image_encoding_base64 = self.convert_to_base64(pil_image=pil_image)

            image_part = {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{image_encoding_base64}",
            }

            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=[image_part])
            ]
            response = self.model_handler.generate_response_chat(messages)
            logger.info(f"Image processed in {time.time() - processing_start_time:.2f} seconds")
            return response
        except Exception as e:
            raise RuntimeError(f"Failed to process image using Ollama model! {e}")


# EXAMPLE USAGE
# if __name__ == "__main__":
#     ollama_handler = OllamaHandler(
#         model_ckpt="moondream:1.8b",
#         reasoning=False,
#         temperature=0.7,
#         num_predict=1024,
#         base_url="http://localhost:11434"
#     )
#     image_processor = OllamaImageProcessor(
#         ollama_model_handler=ollama_handler,
#         system_prompt="You are a helpful assistant that extracts text from images. "
#                       "If the image contains text, return the text as-is. "
#                       "If the image does not contain text, provide a brief description of the image content."
#     )
#     result = image_processor.process(image_filepath="/home/omkanekar28/code/Omnisearch-RAG/data/testing/Screenshot 2023-08-16 150706.png")
#     print(f"Result: {result}")
