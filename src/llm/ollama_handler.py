import time
from abc import ABC, abstractmethod
from typing import List, Union
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from httpx import ConnectError
from ..utils.logger_setup import setup_logger

logger = setup_logger(
    logger_name="ollama_handler.py", 
    filename="ollama_handler.log"
)


class ModelHandler(ABC):
    """Base class for Model Handlers"""

    def validate_messages(
            self, 
            messages: List[Union[
                SystemMessage, 
                HumanMessage, 
                AIMessage
            ]]
    ) -> None:
        """Validates if the given messages are valid"""
        if not messages or not isinstance(messages, list):
            raise ValueError("Messages should be a non-empty list of "
                             "SystemMessage, HumanMessage, or AIMessage objects.")
        for message in messages:
            if not isinstance(message, (SystemMessage, HumanMessage, AIMessage)):
                raise TypeError("Each message should be an instance of "
                                "SystemMessage, HumanMessage, or AIMessage.")

    @abstractmethod
    def initialize_model(self):
        """Initializes the model"""
        pass
    
    @abstractmethod
    def generate_response_chat(
        self, 
        messages: List[Union[
            SystemMessage, 
            HumanMessage, 
            AIMessage
        ]]
    ) -> str:
        """
        Generates response for the given messages

        Example:
            messages = [
                SystemMessage(content="You are a financial analyst expert. 
                                       Extract data accurately from financial documents."),
                HumanMessage(content=[
                    {"type": "text", "text": "What is the dollar based gross retention rate?"},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"}
                ])
            ]
        """
        pass


class OllamaHandler(ModelHandler):
    """Class for handling Ollama models"""

    def __init__(
            self,
            model_ckpt: str,
            reasoning: bool = True,
            temperature: float = 0.7,
            num_predict: int = 1024,
            base_url: str = "http://localhost:11434"     
    ):
        """Initializes the Ollama model"""
        super().__init__()
        self.model_ckpt = model_ckpt
        self.reasoning = reasoning
        self.temperature = temperature
        self.num_predict = num_predict
        self.base_url = base_url
        self.initialize_model()
    
    def initialize_model(self):
        """Initializes the model"""
        logger.info(f"Initializing Ollama model: {self.model_ckpt}...")
        model_initialization_start_time = time.time()
        self.model = ChatOllama(
            model=self.model_ckpt,
            reasoning=self.reasoning,
            temperature=self.temperature,
            num_predict=self.num_predict,
            base_url=self.base_url
        )
        logger.info(f"Model: {self.model_ckpt} initialized in "
                    f"{time.time() - model_initialization_start_time:.2f} seconds")
    
    def generate_response_chat(
            self, 
            messages: List[Union[
                SystemMessage, 
                HumanMessage, 
                AIMessage
            ]]
    ) -> str:
        """Generates response for the given prompt"""
        self.validate_messages(messages)
        logger.info(f"Generating response using model: {self.model_ckpt}...")
        inference_start_time = time.time()
        try:
            response = self.model.invoke(messages)
            logger.debug(f"Model Response: {response}")
            logger.info("Response generated in "
                        f"{time.time() - inference_start_time:.2f} seconds")
            if response and response.content:
                return response.content
            raise ValueError("Model returned empty response")
        except ConnectError:
            logger.error(f"Connection error while connecting to Ollama server! Make sure the server is running and accessible at {self.base_url}.")
            raise
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # import traceback
            # traceback.print_exc()
            raise


# EXAMPLE USAGE
# if __name__ == "__main__":
#     ollama_handler = OllamaHandler(
#         model_ckpt="qwen3:0.6b",
#         reasoning=True,
#         temperature=0.7,
#         num_predict=1024,
#         base_url="http://localhost:11434"
#     )
#     system = "You are an overly cheerful assistant that helps people find information. Always reply in a cheerful tone. Sometimes annoy the user with your cheerfulness and extra details."
#     input_text = "What is the capital of France?"

#     messages = [
#         SystemMessage(content=system),
#         HumanMessage(content=input_text)
#     ]
#     response = ollama_handler.generate_response_chat(messages=messages)
#     print(f"Response: {response}")