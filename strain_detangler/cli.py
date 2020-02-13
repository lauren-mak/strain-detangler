
import click

from .api import (
    build_pangenome,
    find_strains,
)


@click.group()
def main():
    pass


@main.command('pangenome')
def cli_pangenome():
    """Do something pangenome related."""
    build_pangenome(logger=lambda x: click.echo(x, err=True))


@main.command('find-strains')
def cli_find_strains():
    """Do something related to finding strains."""
    find_strains(logger=lambda x: click.echo(x, err=True))


if __name__ == '__main__':
    main()
