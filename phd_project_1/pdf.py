from pathlib import Path
from typing import Literal

import click
from progress.bar import Bar
from pypdf import PageObject, PdfReader

from phd_project_1.utils import encodeTextForGPTModel, identifyAbsolutePath


@click.command(context_settings={"show_default": True})
@click.option(
    "pdfPath",
    "-p",
    "--pdf",
    help="Path to PDF file",
    required=True,
    type=Path,
)
@click.option(
    "outputFilePath",
    "-o",
    "--output",
    help="Path to output text file",
    required=True,
    type=Path,
)
@click.option(
    "tokenLengthToConsole",
    "-l",
    "--token-length",
    help="Print token length to console",
    required=False,
    default=False,
    type=bool,
    is_flag=True,
)
@click.option(
    "textToConsole",
    "-t",
    "--text",
    help="Print PDF text to console",
    required=False,
    default=False,
    type=bool,
    is_flag=True,
)
@click.option(
    "modelName",
    "-m",
    "--model",
    help="Used only in conjuncture with -l / --token-length. Defines a GPT model to encode the text with",
    required=False,
    type=str,
    default="gpt-3.5-turbo",
)
def main(
    pdfPath: Path,
    outputFilePath: Path,
    tokenLengthToConsole: bool,
    textToConsole: bool,
    modelName: str,
) -> None:
    """
    Simple function to extract all text from a PDF and print it to the console.
    """

    pdfAbsPath: Path | Literal[False] = identifyAbsolutePath(
        path=pdfPath,
        suffix=".pdf",
        checkFileExistence=True,
    )

    if pdfAbsPath == False:
        print("Invalid input. Please use a valid path to a PDF (.pdf) file.")
        exit(1)

    outputAbsPath: Path | Literal[False] = identifyAbsolutePath(
        path=outputFilePath,
        suffix=outputFilePath.suffix,
        checkFileExistence=False,
    )

    text: str = ""

    pdfReader: PdfReader = PdfReader(stream=pdfAbsPath, strict=False)

    pageCount: int = len(pdfReader.pages)

    with Bar("Extracting text from PDF document...", max=pageCount) as bar:
        page: PageObject
        for page in pdfReader.pages:
            rawContent: str = page.extract_text()
            content: str = rawContent.strip().replace("\n", " ").lower()
            text = text + content
            bar.next()

    if textToConsole:
        print(text)

    if tokenLengthToConsole:
        encodedTextLength: int = encodeTextForGPTModel(
            text=text,
            gptModel=modelName,
        )[1]
        print(encodedTextLength)

    with open(outputAbsPath, "w") as file:
        file.write(text)
        file.close()

    print(type(text))


if __name__ == "__main__":
    main()
