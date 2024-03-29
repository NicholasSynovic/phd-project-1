from pprint import pprint as print
from typing import List

import click
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion


@click.command()
@click.option(
    "systemPrompt",
    "-s",
    "--system-prompt",
    help="The system prompt to pass to the GPT model",
    required=False,
    default="You are an insightful assistant, skilled in reading various academic papers and aiding in empirical software engineering research",
    type=str,
)
@click.option(
    "gptModel",
    "-g",
    "--gpt-model",
    help="The specific GPT model that you want to use from OpenAI",
    required=False,
    default="gpt-3.5-turbo",
    type=str,
)
@click.option(
    "userPrompt",
    "-u",
    "--user-prompt",
    help="The user prompt to pass to the GPT model",
    required=True,
    type=str,
)
@click.option(
    "apiKey",
    "-k",
    "--api-key",
    help="Your OpenAI GPT API key",
    required=True,
    type=str,
)
def main(systemPrompt: str, gptModel: str, userPrompt: str, apiKey: str) -> None:
    client: OpenAI = OpenAI(api_key=apiKey)

    headers: List[dict[str, str]] = [
        {"role": "system", "content": systemPrompt},
        {"role": "user", "content": userPrompt},
    ]

    response: ChatCompletion = client.chat.completions.create(
        model=gptModel,
        messages=headers,
    )

    print(response.choices[0].model_dump_json(indent=4))


if __name__ == "__main__":
    main()
