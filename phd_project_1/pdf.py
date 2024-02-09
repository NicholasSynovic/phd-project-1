from os.path import abspath, isfile
from pathlib import Path
from typing import List, Literal

import click
import tiktoken
from progress.bar import Bar
from pypdf import PageObject, PdfReader
from tiktoken.core import Encoding

from phd_project_1.utils import encodeTextForGPTModel, identifyAbsolutePath


def identifyPDFAbsolutePath(path: Path) -> Path | Literal[False]:
    pdfAbsolutePath: Path | Literal[False] = identifyAbsolutePath(
        path=path,
        checkFileExistence=True,
    )

    if pdfAbsolutePath == False or pdfAbsolutePath.suffix != ".pdf":
        return False

    return pdfAbsolutePath


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
@click.option(
    "printTokenLength",
    "-l",
    "--token-length",
    help="Rather than printing the text of a PDF file, print its token length",
    required=False,
    default=False,
    type=bool,
    is_flag=True,
)
def main(pdf: Path, printTokenLength: bool) -> None:
    """
    Simple function to extract all text from a PDF and print it to the console.
    """

    pdfAbsPath: Path | Literal[False] = identifyPDFAbsolutePath(path=pdf)

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

    if printTokenLength:
        encodedTextLength: int = encodeTextForGPTModel(text=text)[1]
        print(encodedTextLength)
    else:
        print(text)


if __name__ == "__main__":
    main()
