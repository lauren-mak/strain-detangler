import luigi
import pandas as pd
import subprocess

from os import listdir, makedirs
from os.path import abspath, basename, exists, join, splitext


# python -m luigi --module test_complete Strain_Finder --local-scheduler

def prefix(filename):
    lst = basename(filename).split('.')  # Cuts off the information and suffix.
    if len(lst) > 2:
        lst = lst[:-2]
    else:
        lst = lst[:-1]
    return '.'.join([str(i) for i in lst])


class Prep_Reference(luigi.Task):
    """Splits the reference genome into empty gene-specific SNP count files."""
    reference = luigi.Parameter()
    outdir = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(join('reference', basename(self.reference) + '.rev.2.bt2'))

    def run(self):
        if not exists(join(self.outdir, 'reference')):
            makedirs(join(self.outdir, 'reference'))
        subprocess.run(['/Users/laurenmak/Library/Python/3.7/bin/strain-detangler', 'split_reference', '-o', self.outdir, self.reference])
        subprocess.run(['bowtie2-build', self.reference, join('reference', basename(self.reference))])


class MultiSample_Wrapper(luigi.WrapperTask):
    """Loads samples (pairs of FastQ files) and coordinates mapping-pileup jobs, then distributing gene counts for each sample."""
    outdir = luigi.Parameter()

    def requires(self):  
        if not exists(join(self.outdir, 'tmp')):
            makedirs(join(self.outdir, 'tmp'))

        read_files = [f for f in listdir(join(self.outdir, 'reads')) if not f.startswith('.')]
        processed = []  
        for i in range(0, len(read_files), 2):
            processed.append(Sample_Processor(forward=join('reads', read_files[i]), reverse=join('reads', read_files[i + 1])))
        return processed


class Sample_Processor(luigi.Task):
    """Runs a mapping-pileup jobs for a single sample (pair of FastQ files)."""
    forward = luigi.Parameter()
    reverse = luigi.Parameter()
    reference = luigi.Parameter()
    outdir = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(join(self.outdir, 'tmp', prefix(self.forward) + '.out'))

    def run(self):
        if not exists('./pileups'):
            makedirs('./pileups')
        pileup_prefix = join(self.outdir, 'pileups', prefix(self.forward))
        raw_sam_file = pileup_prefix + '.sam'
        sorted_bam_file = pileup_prefix + '.sort.bam'
        mpileup_file = pileup_prefix + '.mpileup'

        # TODO Write optional downsampling code 
        subprocess.run(['bowtie2', '--sensitive-local', '-p', '8', '-x', join('reference', basename(self.reference)), '-1', self.forward, '-2', self.reverse, '-S', raw_sam_file])
        subprocess.run(['samtools', 'sort', raw_sam_file, '-o', sorted_bam_file])
        subprocess.run(['samtools', 'index', sorted_bam_file])
        subprocess.run(['samtools', 'mpileup', '-f', self.reference, sorted_bam_file, '-o', mpileup_file])
        subprocess.run(['/Users/laurenmak/Library/Python/3.7/bin/strain-detangler', 'split_pileup', '-o', self.outdir, mpileup_file]) # Use all defaults. 
        with self.output().open('w') as tmp_out:
            tmp_out.write('{} preprocessing has finished.\n'.format(prefix(self.forward)))
        

class MultiGene_Wrapper(luigi.WrapperTask):
    """Loads filled gene-specific count files and coordinates SNP downsizing."""
    reference = luigi.Parameter()
    outdir = luigi.Parameter()

    def requires(self): 
        processed = []  
        for g in listdir(join(self.outdir, 'genes')):
            processed.append(Gene_Processor(gene = g.strip()))
        return processed

    def output(self):
        ref_prefix = prefix(self.reference)
        return luigi.LocalTarget(join(self.outdir, ref_prefix + '.reduced.list'))

    def run(self):
        with self.output().open('w') as out_list:
            for f in self.input():
                out_list.write(abspath(f.path) + '\n')


class Gene_Processor(luigi.Task):
    """Downsizes the number of SNPs per gene by entropy scaling on the columns."""
    gene = luigi.Parameter()
    radius = luigi.Parameter(default='0.01')
    outdir = luigi.Parameter()

    def output(self):
        gene_prefix = prefix(self.gene)
        return luigi.LocalTarget(join(self.outdir, 'reduced', gene_prefix + '.reduced.csv'))

    def run(self):
        subprocess.run(['/Users/laurenmak/Library/Python/3.7/bin/strain-detangler', 'reduce', '-r', self.radius, '-o', self.outdir, join('genes', self.gene)])


class Reads2Counts(luigi.Task): 
    reference = luigi.Parameter()
    outdir = luigi.Parameter()

    def output(self):
        ref_prefix = prefix(self.reference)
        return luigi.LocalTarget(join(self.outdir, ref_prefix + '.reduced.list'))

    def run(self):
        yield Prep_Reference()
        yield MultiSample_Wrapper()
        yield MultiGene_Wrapper()


class Strain_Finder(luigi.Task):
    """Train LDA model on whole-genome SNP counts to infer strain composition of each sample."""
    reference = luigi.Parameter()
    outdir = luigi.Parameter()
    band_filter = luigi.Parameter(default='0.001')
    num_topics = luigi.Parameter(default='10')

    def requires(self):
        return Reads2Counts()

    def output(self):
        return luigi.LocalTarget(join(self.outdir, 'model', prefix(self.reference) + '.gensim_doctopic.csv'))

    def run(self):
        subprocess.run(['/Users/laurenmak/Library/Python/3.7/bin/strain-detangler', 'lda_train', '-bf', self.band_filter, '-t', self.num_topics, self.input().path]) 



if __name__ == '__main__':
    luigi.run(main_cls_task=Strain_Finder)
