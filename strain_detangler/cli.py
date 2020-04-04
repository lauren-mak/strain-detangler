
import click
import pandas as pd

from Bio import SeqIO
from csv import writer
from os.path import join
from seaborn import clustermap

from .api import (
    build_pangenome,
    entropy_reduce_position_matrix,
    prefix,
    check_directory,
    parse_pileup_bases,
    write_to_gene_file,
    concat_matrices,
    remove_uninformative_snps,
    lda_gensim,
)


@click.group()
def main():
    pass


@main.command('pangenome')
def cli_pangenome():
    """Do something pangenome related."""
    build_pangenome(logger=lambda x: click.echo(x, err=True))


@main.command('split_reference')
@click.option('-o', '--output', type=click.Path(), default='./',
              help='Master output directory')
@click.argument('reference', type=click.Path())
def split_reference(reference, output):
    """Split reference sequence into gene-specific count files."""
    outdir = check_directory(output, 'genes/')
    lname = join(output, prefix(reference) + '.genes.list')
    for gene in SeqIO.parse(reference, "fasta"):  # For gene in the reference sequence
        outfile = join(outdir, gene.id + '.counts.csv')
        with open(lname, 'a+') as lf:
            lf.write(outfile + '\n')
        with open(outfile, 'w+') as cf:
            header = ['Sample']
            for i in range(len(gene)):
                pos = str(i + 1)
                header += [str(pos) + '-A', str(pos) + '-C', str(pos) + '-G', str(pos) + '-T']
            writer(cf).writerow(header)


@main.command('split_pileup')  # Adapted from DESMAN's pileup_to_freq_table.py.
@click.option('-o', '--output', type=click.Path(), default='./',
              help='Master output directory')
@click.argument('pileup')
def split_pileup(pileup, output):
    """Add pileup read-counts to each gene-specific file."""
    # Each pileup line: NODE_23_length_20156_cov_7.51057        1813    G       1       ^M.     <
    pname = prefix(pileup)
    curr_gene = open(pileup, 'r').readline().strip().split('\t')[0]
    pinfo = [pname]
    last_pos = 0

    outdir = check_directory(output, 'genes/')
    with open(pileup) as pf:  # A samtools mpileup parser.
        for line in pf:  # For each position of each scaffold with read coverage in this sample...
            tkns = line.strip().split('\t')
            if int(tkns[3]) == 0:
                continue
            if tkns[0] != curr_gene:  # If new gene is being processed, finish the line corresponding to the last gene.
                write_to_gene_file(outdir, curr_gene, last_pos, pinfo)
                curr_gene = tkns[0]  # Update to the current gene.
                pinfo = [pname]
                last_pos = 0  # Reset to the start of the scaffold.
            curr_pos = int(tkns[1])
            for p in range(int(curr_pos - last_pos - 1)):
                # If not starting at the previously-recorded position on the gene, fill in missed columns with 0.
                pinfo += ['0', '0', '0', '0']
            pinfo += parse_pileup_bases(tkns[2], tkns[4], tkns[5])  # ...write the read coverage at that position.
            last_pos = curr_pos
    write_to_gene_file(outdir, curr_gene, last_pos, pinfo)

@main.command('reduce')
@click.option('-m', '--metric', default='cosine')
@click.option('-r', '--radius', default=0.01)
@click.option('-o', '--output', type=click.Path(), default='./',
              help='Master output directory')
@click.argument('filename')
def reduce(metric, radius, output, filename):   # Adapted from DD's gimmebio.stat-strains.reduce.
    """Reduce number of SNP columns by radial set coverage."""
    def logger(n_centroids, n_cols):
        if (n_cols % 100) == 0:
            click.echo(f'{n_centroids} centroids, {n_cols} columns', err=True)

    outdir = check_directory(output, 'reduced/')
    matrix = pd.read_csv(filename, index_col=0, header=0)
    full_reduced = entropy_reduce_position_matrix(
        matrix,
        radius,
        metric,
        logger=logger
    )
    click.echo(full_reduced.shape, err=True)
    outfile = join(outdir, prefix(filename) + 'reduced.csv')
    full_reduced.to_csv(outfile)
    with open(join(outdir, '.reduced.list'), 'a+', newline='') as lf:
        writer(lf).writerow(outfile)


@main.command('lda_train')
@click.option('-bf', '--band_filter', default=0.001,
              help='Threshold to filter out uninformative position-alleles')
@click.option('-o', '--output', type=click.Path(), default='./',
              help='Master output directory')
@click.argument('file_list')
def lda_train(file_list, band_filter, output):
    """Generate sample and strain composition matrices via LDA."""
    def logger(i):
        if (i % 100) == 0:
            click.echo(f'Gene file number {i} loaded', err=True)

    fname = prefix(file_list)
    dframe, summary = concat_matrices(file_list, logger)
        # To print the massive dataframe, add directory and filename.
    click.echo("Finished concatenating from list of dataframes", err=True)
    click.echo(f'{summary[0]} SNPs, {summary[1]} total coverage, {summary[2]} samples', err=True)

    if band_filter:
        dframe, summary, filter_info = remove_uninformative_snps(dframe, band_filter)
            # To print the massive dataframe, add directory and filename.
        click.echo("Finished band-filtering dataframe", err=True)
        click.echo(f'{summary[0]} SNPs, {summary[1]} total coverage, {summary[2]} samples', err=True)
        click.echo(f'Removed {filter_info[0]} position-alleles that may be sequencing errors', err=True)
        click.echo(f'Removed {filter_info[1]} position-alleles that are not informative', err=True)

    lda_gensim(dframe, output, fname, logger=lambda x: click.echo(x, err=True))


@main.command('cluster_map')
@click.option('-m', '--metadata', default=False,
              help='Reduced metadata file to label samples for cluster-mapping')
@click.option('-o', '--output', type=click.Path(), default='./',
              help='Output directory for the graphics.')
@click.argument('filename')
def cluster_map(filename, metadata, output):
    """Generate seaborn clustermap for the input file."""
    fname = prefix(filename)
    outdir = check_directory(output, 'model/')

    final = pd.read_csv(filename, header=0, index_col=0).join(pd.read_csv(metadata, header=0, index_col=0), how='left')
    final.fillna('experiment', inplace=True)
        # This fills in the metadata (city and continent) of samples without an entry with 'experiment'.
    final_filter = final[final['Type'] != 'excluded']  # Pick all samples that don't have inconsistencies.
    controls = final_filter[final_filter['Type'] != 'experiment']
    controls.to_csv(join(outdir, fname + "_controls.csv"), index=True)  # Output samples that aren't labeled as cases.

    # Make the base figure.
    results_only = final_filter.iloc[:, 0:(len(final.columns) - 2)]
    types = final_filter.pop("Type")
    targets = ['experiment', 'negative_control_air', 'negative_control_lab', 'positive_control']
    colors = ['0.75', 'c', 'b', 'r']
    colordict = dict(zip(targets, colors))
    row_colors = types.map(colordict)
    fig = clustermap(results_only, cmap="Blues", col_cluster=False, linewidths=0, yticklabels=False,
                     row_colors=row_colors)

    # Make the legends for the row_color side-bar as well as the heatmap.
    for label in types.unique():
        fig.ax_col_dendrogram.bar(0, 0, color=colordict[label], label=label, linewidth=0)
    fig.ax_col_dendrogram.legend(loc="center", ncol=4)
    fig.cax.set_position([.97, .2, .03, .45])

    fig.savefig(join(outdir, fname + '.cmap.png'))
    click.echo('Finished making document-topic cluster-map', err=True)


if __name__ == '__main__':
    main()
