import backoff  # for exponential backoff
import openai
import os
import asyncio
from typing import Any

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def chat_completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

async def dispatch_openai_chat_requests(
    messages_list: list[list[dict[str,Any]]],
    model: str,
    max_tokens: int,
) -> list[str]:
    """Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
        stop_words: List of words to stop the model from generating.
    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x, 
            max_tokens=max_tokens,
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)

async def dispatch_openai_prompt_requests(
    messages_list: list[list[dict[str,Any]]],
    model: str,
    max_tokens: int,
) -> list[str]:
    async_responses = [
        openai.Completion.acreate(
            model=model,
            prompt=x,
            max_tokens=max_tokens
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)

class OpenAIModel:
    def __init__(self, API_KEY, model_name, stop_words, max_new_tokens, api_base=None) -> None:
        openai.api_key = API_KEY
        if api_base:
            openai.api_base = api_base
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words
        self.api_base = api_base

    def _log_api_error(self, error: Exception):
        print(f"[OpenAIModel] API error for model {self.model_name}: {error}", flush=True)

    # used for chat-gpt and gpt-4
    def chat_generate(self, input_string, temperature = 0.0):
        try:
            response = chat_completions_with_backoff(
                    model = self.model_name,
                    messages=[
                            {"role": "user", "content": input_string}
                        ],
                    max_tokens = self.max_new_tokens,
            )
            generated_text = response['choices'][0]['message']['content'].strip()
            return generated_text
        except Exception as e:
            self._log_api_error(e)
            return None
    
    # used for text/code-davinci
    def prompt_generate(self, input_string, temperature = 0.0):
        try:
            response = completions_with_backoff(
                model = self.model_name,
                prompt = input_string,
                max_tokens = self.max_new_tokens,
                temperature = temperature
            )
            generated_text = response['choices'][0]['text'].strip()
            return generated_text
        except Exception as e:
            self._log_api_error(e)
            return None

    def generate(self, input_string, temperature = 0.0):
        # default to chat-style models for unknown names (e.g., third-party providers)
        if self.model_name in ['text-davinci-002', 'code-davinci-002', 'text-davinci-003']:
            return self.prompt_generate(input_string, temperature)
        else:
            return self.chat_generate(input_string, temperature)
    
    def batch_chat_generate(self, messages_list, temperature = 0.0):
        open_ai_messages_list = []
        for message in messages_list:
            open_ai_messages_list.append(
                [{"role": "user", "content": message}]
            )
        try:
            predictions = asyncio.run(
                dispatch_openai_chat_requests(
                        open_ai_messages_list, self.model_name, self.max_new_tokens
                )
            )
            return [x['choices'][0]['message']['content'].strip() for x in predictions]
        except Exception as e:
            self._log_api_error(e)
            return [None] * len(messages_list)
    
    def batch_prompt_generate(self, prompt_list, temperature = 0.0):
        try:
            predictions = asyncio.run(
                dispatch_openai_prompt_requests(
                        prompt_list, self.model_name, self.max_new_tokens
                )
            )
            return [x['choices'][0]['text'].strip() for x in predictions]
        except Exception as e:
            self._log_api_error(e)
            return [None] * len(prompt_list)

    def batch_generate(self, messages_list, temperature = 0.0):
        # default to chat-style batching for unknown model names
        if self.model_name in ['text-davinci-002', 'code-davinci-002', 'text-davinci-003']:
            return self.batch_prompt_generate(messages_list, temperature)
        else:
            return self.batch_chat_generate(messages_list, temperature)

    def generate_insertion(self, input_string, suffix, temperature = 0.0):
        try:
            response = completions_with_backoff(
                model = self.model_name,
                prompt = input_string,
                suffix= suffix,
                max_tokens = self.max_new_tokens,
                temperature = temperature
            )
            generated_text = response['choices'][0]['text'].strip()
            return generated_text
        except Exception as e:
            self._log_api_error(e)
            return None
