from string import Template
from urllib.parse import quote
from webbrowser import open_new_tab

import click

URL: Template = Template(
    template="https://www.nature.com/search?q=${searchQuery}&article_type=research&order=relevance&date_range=${startYear}-${endYear}"
)


@click.command()
@click.option(
    "searchQuery",
    "-s",
    "--search-query",
    help="Search query",
    required=True,
    type=str,
)
@click.option(
    "startYear",
    "--start-year",
    help="Start year",
    required=True,
    type=int,
)
@click.option(
    "endYear",
    "--end-year",
    help="End year",
    required=True,
    type=int,
)
def main(searchQuery: str, startYear: int, endYear: int) -> None:
    encodedQuery: str = quote(string=searchQuery)

    url: str = URL.substitute(
        searchQuery=encodedQuery,
        startYear=startYear,
        endYear=endYear,
    )

    open_new_tab(url=url)


if __name__ == "__main__":
    main()
