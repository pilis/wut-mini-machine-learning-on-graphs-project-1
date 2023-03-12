import logging

import click

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
def main():
    print("Hello, World!")


if __name__ == "__main__":
    main()
