import time
import base64
from abc import ABC, abstractmethod
from typing import List
from io import BytesIO
from PIL import Image, ImageFile
from langchain_core.messages import SystemMessage, HumanMessage
from ..llm.ollama_handler import OllamaHandler
from ..utils.logger_setup import setup_logger
from ..utils.common import validate_file_location, validate_file_type

logger = setup_logger(
    logger_name="image_preprocessor.py", 
    filename="image_preprocessor.log"
)


class BaseImageProcessor(ABC):
    """Abstract base class for Image processing"""

    def __init__(self, model_handler: OllamaHandler) -> None:
        """Initializes the valid image extensions"""
        self.model_handler = model_handler
        self.valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}
    
    def validate_image_file(self, filepath: str) -> None:
        """Validates if the given file exists and is an image"""
        validate_file_location(filepath=filepath)
        validate_file_type(
            filepath=filepath,
            valid_extensions=self.valid_extensions
        )

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
    def process(self, image_filepath: str) -> List[str]:
        """Processes the given image and returns found text content"""
        pass


class OllamaImageProcessor(BaseImageProcessor):
    """Image processor implementation using Ollama vision models"""

    def __init__(
            self,
            ollama_model_handler: OllamaHandler = OllamaHandler(
                model_ckpt="moondream:1.8b",
                reasoning=False,
                temperature=0.7,
                num_predict=1024,
                base_url="http://localhost:11434"
            ),
            system_prompt: str = "You are a helpful assistant that extracts text from images. "
                "If the image contains text, return the text as-is. "
                "If the image does not contain text, provide a brief description of the image content.",
    ) -> None:
        """Initializes the parent and child class parameters"""
        super().__init__(ollama_model_handler)
        self.system_prompt = system_prompt
    
    def process(self, image_filepath: str) -> List[str]:
        """
        Analyzes the given image and returns its textual content.

        Behavior:
            - If the image predominantly contains text, returns the extracted text as-is.
            - Otherwise, returns a descriptive caption summarizing the image content.

        Args:
            image_filepath (str): Path to the input image file.

        Returns:
            List[str]: Extracted text or a generated description of the image.
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
            return [response]
        except Exception as e:
            raise RuntimeError(f"Failed to process image using '{self.model_handler.model_ckpt}' "
                               f"model via Ollama! {e}")


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
#     result = image_processor.process(image_filepath="")
#     print(f"Result: {result}")
