from os.path import abspath
from pathlib import Path

import tiktoken
from pypdf import PageObject, PdfReader


def main() -> None:
    text: str = ""

    pdfFile: Path = Path(
        abspath(
            path="/home/nicholas/Zotero/storage/84HG6PHM/Kwon et al. - 2022 - A Fast Post-Training Pruning Framework for Transfo.pdf"
        )
    )

    pdfReader: PdfReader = PdfReader(stream=pdfFile, strict=False)

    page: PageObject
    for page in pdfReader.pages:
        rawContent: str = page.extract_text()
        content: str = rawContent.strip().replace("\n", " ").lower()
        text = text + content

    enc = tiktoken.get_encoding("cl100k_base")
    test = enc.encode(text=text)

    print(test)
    print(len(test))


if __name__ == "__main__":
    main()
