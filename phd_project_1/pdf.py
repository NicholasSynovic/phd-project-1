from os.path import abspath, isfile
from pathlib import Path
from pprint import pprint as print
from typing import List, Literal

import click
import tiktoken
from progress.bar import Bar
from pypdf import PageObject, PdfReader
from tiktoken.core import Encoding


def identifyAbsolutePath(path: Path) -> Path | Literal[False]:
    absPath: Path = Path(abspath(path))

    if isfile(path=absPath) and absPath.suffix == ".pdf":
        return absPath

    else:
        return False


def encodeText(text: str) -> List[int]:
    encoder: Encoding = tiktoken.get_encoding("cl100k_base")
    return encoder.encode(text=text)


@click.command()
@click.option(
    "pdf",
    "-p",
    "--pdf",
    help="Path to PDF file",
    required=True,
    type=Path,
)
def main(pdf: Path) -> None:
    """
    Simple function to extract all text from a PDF and print it to the console.
    """

    pdfAbsPath: Path | Literal[False] = identifyAbsolutePath(path=pdf)

    if pdfAbsPath == False:
        print("Invalid input. Please use a valid path to a PDF file.")
        exit(1)

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

    encodedText: List[int] = encodeText(text=text)

    print(text)


if __name__ == "__main__":
    main()
