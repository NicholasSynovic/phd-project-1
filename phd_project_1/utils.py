from os.path import abspath, isfile
from pathlib import Path
from typing import List, Literal, Tuple

import nltk
import pandas
import tiktoken
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pandas import DataFrame
from tiktoken.core import Encoding


def downloadNLTKStopwords() -> None:
    nltk.download(info_or_id="stopwords", quiet=True)
    nltk.download(info_or_id="punkt", quiet=True)


def encodeTextForGPTModel(
    text: str,
    gptModel: str = "gpt-3.5-turbo",
    removeStopWords: bool = False,
) -> Tuple[List[int], int]:
    """
    Return format is:
        0: The encoded text as a List of ints
        1: The length of the encoded text List
    """

    rawText: str = text

    if removeStopWords:
        downloadNLTKStopwords()
        nltkTokens: List[str] = word_tokenize(text=text)
        filteredText: List[str] = [
            token
            for token in nltkTokens
            if token.lower() not in stopwords.words("english")
        ]
        rawText: str = " ".join(filteredText)

    encoding: Encoding = tiktoken.encoding_for_model(model_name=gptModel)
    encodedText: List[int] = encoding.encode(text=rawText)
    encodedTextLength: int = len(encodedText)
    return (encodedText, encodedTextLength)


def getGPTModelTokenLimits(
    gptModel: str = "gpt-3.5-turbo",
    datafile: Path = Path("../data/gptModelTokenLengths.csv"),
) -> DataFrame:
    df: DataFrame = pandas.read_csv(filepath_or_buffer=datafile)
    return df[df["Model Name"] == gptModel].reset_index(drop=True)


def identifyAbsolutePath(
    path: Path,
    suffix: str,
    checkFileExistence: bool = False,
) -> Path | Literal[False]:
    absolutePath: Path = Path(abspath(path=path))

    if absolutePath == False or absolutePath.suffix != suffix:
        return False

    if checkFileExistence:
        if isfile(path=absolutePath):
            return absolutePath
        else:
            return False

    return absolutePath
