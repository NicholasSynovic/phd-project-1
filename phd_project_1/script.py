import argparse
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load env variables
load_dotenv(".env")


def main(
    user_content: str,
    system_content: Optional[
        str
    ] = "You an insightful assistant, skilled in reading various academic papers and aiding in empirical software engineering research.",
) -> None:
    # init openai client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # create completion
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
    )

    # print returned mssg
    print(completion.choices[0].message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A cli tool to interact with OpenAI's api"
    )
    parser.add_argument(
        "--user_content", type=str, help="The user content to prompt the ai with"
    )
    parser.add_argument(
        "--system_content",
        type=str,
        default="You an insightful assistant, skilled in reading various academic papers and aiding in empirical software engineering research.",
        help="The system content for the ai - who they are (optional)",
    )

    args = parser.parse_args()

    main(args.user_content, args.system_content)
