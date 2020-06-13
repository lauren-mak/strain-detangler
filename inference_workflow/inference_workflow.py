import csv
import luigi
import pandas as pd
import subprocess

from os import listdir, makedirs
from os.path import abspath, basename, exists, join, splitext


from strain_detangler.api import (
    prefix,
    check_make,
    python_sed
)

# PYTHONPATH='.' luigi --module strain_workflow Strain_Finder --workers 10


class Prep_Reference(luigi.Task):
    """Splits the reference genome into empty gene-specific SNP count files."""
    reference_raw = luigi.Parameter()
    reference = luigi.Parameter()
    outdir = luigi.Parameter()
    ref_prefix = luigi.Parameter(default = '')
    ref_outdir = luigi.Parameter(default = '')
    genes_outdir = luigi.Parameter(default = '')

    def __init__(self, *args, **kwargs):
        super(Prep_Reference, self).__init__(*args, **kwargs)
        self.ref_prefix = prefix(self.reference, 1)
        self.ref_outdir = check_make(self.outdir, 'reference')
        self.genes_outdir = check_make(self.outdir, 'genes')

    def output(self):
        return luigi.LocalTarget(join(self.ref_outdir, self.ref_prefix + '.rev.2.bt2'))

    def run(self):
        python_sed(self.reference_raw, self.reference)
        split_step = subprocess.run(['strain-detangler', 'split_reference', '-o', self.genes_outdir, self.reference])
        subprocess.run(['bowtie2-build', self.reference, join(self.ref_outdir, self.ref_prefix)])


class MultiSample_Wrapper(luigi.WrapperTask):
    """Loads samples (pairs of FastQ files) and coordinates mapping-pileup jobs, then distributing gene counts for each sample."""
    reference = luigi.Parameter()
    reads = luigi.Parameter()
    outdir = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super(MultiSample_Wrapper, self).__init__(*args, **kwargs)
        log_outdir = check_make(self.outdir, 'workflow_logs')

    def requires(self):  
        read_files = [f for f in listdir(self.reads) if not f.startswith('.')]
        processed = []  
        for i in range(0, len(read_files), 2):
            processed.append(Sample_Processor(forward = join(self.reads, read_files[i]), reverse = join(self.reads, read_files[i + 1])))
        return processed

    def output(self):
        return luigi.LocalTarget(join(self.outdir, 'workflow_logs', prefix(self.reference, 1) + '_preprocessing.out'))

    def run(self):
        with self.output().open('w') as tmp_out:
            tmp_out.write('Finished')


class Sample_Processor(luigi.Task):
    """Runs a mapping-pileup jobs for a single sample (pair of FastQ files)."""
    forward = luigi.Parameter()
    reverse = luigi.Parameter()
    reference = luigi.Parameter()
    outdir = luigi.Parameter()

    ref_outdir = luigi.Parameter(default = '')
    mp_outdir = luigi.Parameter(default = '')
    genes_outdir = luigi.Parameter(default = '')

    pileup_prefix = luigi.Parameter(default = '')
    raw_sam_file = luigi.Parameter(default = '')
    sorted_bam_file = luigi.Parameter(default = '')
    mpileup_file = luigi.Parameter(default = '')

    def __init__(self, *args, **kwargs):
        super(Sample_Processor, self).__init__(*args, **kwargs)

        self.ref_outdir = join(self.outdir, 'reference')
        self.mp_outdir = check_make(self.outdir, 'pileups')
        self.genes_outdir = join(self.outdir, 'genes')

        self.pileup_prefix = prefix(self.forward, 1)
        self.raw_sam_file = join(self.mp_outdir, self.pileup_prefix + '.sam')
        self.sorted_bam_file = join(self.mp_outdir, self.pileup_prefix + '.sort.bam')
        self.mpileup_file = join(self.mp_outdir, self.pileup_prefix + '.mpileup')

    def output(self):
        return luigi.LocalTarget(join(self.outdir, 'workflow_logs', self.pileup_prefix + '.out'))

    def run(self):

        # TODO Write optional downsampling code 
        map_step = subprocess.run(['bowtie2', '--sensitive-local', '-p', '8', '-x', join(self.ref_outdir, prefix(self.reference, 1)), '-1', self.forward, '-2', self.reverse, '-S', self.raw_sam_file])
        sort_step = subprocess.run(['samtools', 'sort', self.raw_sam_file, '-o', self.sorted_bam_file])
        index_step = subprocess.run(['samtools', 'index', self.sorted_bam_file])
        pile_step = subprocess.run(['samtools', 'mpileup', '-f', self.reference, self.sorted_bam_file, '-o', self.mpileup_file])
        split_step = subprocess.run(['strain-detangler', 'split_pileup', '-o', self.genes_outdir, self.mpileup_file]) # Use all defaults. 
        with self.output().open('w') as tmp_out:
            csv.writer(tmp_out).writerow([self.pileup_prefix, map_step.returncode, sort_step.returncode, pile_step.returncode, split_step.returncode]) # Now a useful log file to see if any fail at the preprocessing step.


class MultiGene_Wrapper(luigi.WrapperTask):
    """Loads filled gene-specific count files and coordinates SNP downsizing."""
    reference = luigi.Parameter()
    outdir = luigi.Parameter()
    genes_outdir = luigi.Parameter(default = '')
    red_outdir = luigi.Parameter(default = '')

    def __init__(self, *args, **kwargs):
        super(MultiGene_Wrapper, self).__init__(*args, **kwargs)
        self.genes_outdir = join(self.outdir, 'genes')
        self.red_outdir = check_make(self.outdir, 'reduced')

    def requires(self): 
        processed = []  
        for g in listdir(self.genes_outdir):
            processed.append(Gene_Processor(gene = join(self.genes_outdir, g), red_outdir = self.red_outdir))
        return processed

    def output(self):
        return luigi.LocalTarget(join(self.outdir, prefix(self.reference, 1) + '.reduced.list'))

    def run(self):
        with self.output().open('w') as out_list:
            for g in self.input():
                with open(g.path, 'r') as f:
                    for i in f.readlines():
                        info = i.split(',')
                        if int(info[1]) > 0:  # Only write to the reduced list if there are samples covering the gene. 
                            out_list.write(info[0] + '\n')


class Gene_Processor(luigi.Task):
    """Downsizes the number of SNPs per gene by entropy scaling on the columns."""
    gene = luigi.Parameter()
    radius = luigi.Parameter(default='0.01')
    outdir = luigi.Parameter()
    red_outdir = luigi.Parameter(default = '')

    def __init__(self, *args, **kwargs):
        super(Gene_Processor, self).__init__(*args, **kwargs)
        self.red_outdir = join(self.outdir, 'reduced')

    def output(self):
        return luigi.LocalTarget(join(self.outdir, 'workflow_logs', prefix(self.gene, 2) + '.out'))

    def run(self):
        reduce_step = subprocess.run(['strain-detangler', 'reduce', '-r', self.radius, '-o', self.red_outdir, self.gene])
        reduced_file = join(self.red_outdir, prefix(self.gene, 2) + '.reduced.csv')
        with self.output().open('w') as tmp_out:
            info = pd.read_csv(reduced_file, header = 0, index_col = 0)
            tmp_out.write(reduced_file + ',' + str(info.shape[0])) # Now a useful log file to see if there are non-covered genes.


class Reads2Counts(luigi.Task): 
    reference = luigi.Parameter()
    outdir = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(join(self.outdir, prefix(self.reference, 1) + '.reduced.list'))

    def run(self):
        yield Prep_Reference()
        yield MultiSample_Wrapper()
        yield MultiGene_Wrapper()


class Strain_Finder(luigi.Task):
    """Train LDA model on whole-genome SNP counts to infer strain composition of each sample."""
    reference = luigi.Parameter()
    outdir = luigi.Parameter()
    model_outdir = luigi.Parameter(default = '')
    band_filter = luigi.Parameter(default='0.001')
    num_topics = luigi.Parameter(default='10')

    def __init__(self, *args, **kwargs):
        super(Strain_Finder, self).__init__(*args, **kwargs)
        self.model_outdir = check_make(self.outdir, 'model')

    def requires(self):
        return Reads2Counts()

    def output(self):
        return luigi.LocalTarget(join(self.model_outdir, prefix(self.reference, 1) + '.gensim_doctopic.csv'))

    def run(self):
        subprocess.run(['strain-detangler', 'lda_train', '-bf', self.band_filter, '-t', self.num_topics, '-o', self.model_outdir, self.input().path]) 


if __name__ == '__main__':
    luigi.run(main_cls_task=Strain_Finder)
