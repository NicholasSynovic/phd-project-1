from pathlib import Path
from typing import List, Literal

import click
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from pandas import DataFrame

from phd_project_1.utils import *


def readTextFile(textFilePath: Path) -> str:
    textFileAbsPath: Path | Literal[False] = identifyAbsolutePath(
        path=textFilePath,
        suffix=".txt",
        checkFileExistence=True,
    )

    if textFileAbsPath == False:
        print("Invalid input file. Please use a valid TXT (.txt) file")
        quit(1)

    with open(textFileAbsPath, "r") as file:
        data: str = file.read()
        file.close()

    return data


def getEncodedTextLength(
    systemPrompt: str,
    userPrompt: str,
    gptModel: str,
) -> Tuple[int, int, int]:
    """
    Return format is:
        0: Length of encoded system prompt (x)
        1: Length of encoded user prompt (y)
        2: x + y
    """

    encodedSystemPromptLength: int = encodeTextForGPTModel(
        text=systemPrompt,
        gptModel=gptModel,
    )[1]

    encodedUserPromptLength: int = encodeTextForGPTModel(
        text=userPrompt,
        gptModel=gptModel,
    )[1]

    return (
        encodedSystemPromptLength,
        encodedUserPromptLength,
        encodedSystemPromptLength + encodedUserPromptLength,
    )


@click.command()
@click.option(
    "systemPrompt",
    "-s",
    "--system-prompt",
    help="The system prompt to pass to the GPT model",
    required=False,
    default="You are an insightful assistant, skilled in reading various academic papers and aiding in empirical software engineering research. Return results in JSON format",
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
    "apiKey",
    "-k",
    "--api-key",
    help="Your OpenAI GPT API key",
    required=True,
    type=str,
)
@click.option(
    "textFile",
    "-i",
    "--text-file-input",
    help="Path to a text file to read from as input",
    required=True,
    type=Path,
)
def main(
    systemPrompt: str,
    gptModel: str,
    apiKey: str,
    textFile: Path,
) -> None:
    userPrompt: str = readTextFile(textFilePath=textFile)

    # TODO: Set this to read from a user parameter w.r.t data file location
    try:
        allowedInputTokenCount: int = getGPTModelTokenLimits(gptModel=gptModel)[
            "Input Tokens"
        ].to_list()[0]
    except IndexError:
        print("Invalid GPT model")
        quit(2)

    tokenLengths: Tuple[int, int, int] = getEncodedTextLength(
        systemPrompt=systemPrompt,
        userPrompt=userPrompt,
        gptModel=gptModel,
    )

    if tokenLengths[2] > allowedInputTokenCount:
        print(
            f"""
Your encoded system and user prompts are to long for {gptModel}
System prompt + User prompt > Input token limit
{tokenLengths[0]} + {tokenLengths[1]} = {tokenLengths[2]} > {allowedInputTokenCount}
"""
        )

    quit(3)

    client: OpenAI = OpenAI(api_key=apiKey)

    headers: List[dict[str, str]] = [
        {"role": "system", "content": systemPrompt},
        {"role": "user", "content": userPrompt},
    ]

    response: ChatCompletion = client.chat.completions.create(
        messages=headers,
        model=gptModel,
        logprobs=True,
        max_tokens=100,
        n=1,
        temperature=0.2,
    )

    print(response.choices[0].model_dump_json(indent=4))


if __name__ == "__main__":
    main()
