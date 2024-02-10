from os.path import abspath, isfile
from pathlib import Path
from typing import List, Literal, Tuple

import pandas
import tiktoken
from pandas import DataFrame
from tiktoken.core import Encoding


def encodeTextForGPTModel(
    text: str,
    gptModel: str = "gpt-3.5-turbo",
) -> Tuple[List[int], int]:
    encoding: Encoding = tiktoken.encoding_for_model(model_name=gptModel)
    encodedText: List[int] = encoding.encode(text=text)
    encodedTextLength: int = len(encodedText)
    return (encodedText, encodedTextLength)


def getGPTModelTokenLimits(
    gptModel: str = "gpt-3.5-turbo",
    datafile: Path = Path("../data/gptModelTokenLengths.csv"),
) -> DataFrame:
    df: DataFrame = pandas.read_csv(filepath_or_buffer=datafile)
    return df[df["Model Name"] == gptModel].reset_index(inplace=True)


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
