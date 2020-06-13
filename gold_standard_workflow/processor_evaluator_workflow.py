
import csv
import luigi
import pandas as pd
import subprocess

from Bio.Alphabet import generic_dna
from Bio.Align.Applications import MafftCommandline
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from os import listdir, makedirs
from os.path import abspath, basename, exists, join, splitext

from strain_detangler.api import (
    prefix,
    check_make,
    python_sed,
    concat_matrices,
    make_strains_lda,
    intersect_snps, 
    make_pool, 
    pairwise_manhattan,
    mcc_calculator,
    jsd_calculator
)


# PYTHONPATH='.' luigi --module strain_gstd_processor Strain_GSTD_Processor --local
# PYTHONPATH='.' luigi --module strain_gstd_processor Strain_Evaluator --local


def check_make(output, subdir):
    outdir = join(output, subdir)
    if not exists(outdir):
        makedirs(outdir)
    return outdir


def python_sed(in_file, out_file):
    with open(in_file, 'r') as fin, open(out_file, 'w') as fout:
        list(fout.write(line.replace('lcl|', '')) for line in fin)


def get_gene_names(gl):
    genes_in_inference = []
    with open(gl, 'r') as lfile:
        for gene_path in lfile:
            genes_in_inference.append(prefix(gene_path.strip(), 2))
    return genes_in_inference


class Gene_Split_Wrapper(luigi.WrapperTask):
    """Split reference sequence into separate gene FASTAs."""
    rep_genes_start = luigi.Parameter()
    rep_genes_clean = luigi.Parameter()
    rep_genes_inf = luigi.Parameter()
    outdir = luigi.Parameter()
    gene_list = luigi.Parameter() # These are the genes that are in the inference process. 

    def requires(self):  
        genes_in_inference = get_gene_names(self.gene_list)  # Makes sure that the only genes in comparison are the only ones in inference.
        python_sed(self.rep_genes_start, self.rep_genes_clean)
        genes_outdir = check_make(self.outdir, 'representative_genes')
        inference_gene_map = {}
        for curr_gene in SeqIO.parse(self.rep_genes_clean, 'fasta'):  # For gene in the reference sequence
            if curr_gene.id in genes_in_inference:
                inference_gene_map[genes_in_inference.index(curr_gene.id)] = Gene_Writer(gene = curr_gene, g_outdir = genes_outdir)
        inference_gene_paths = []
        for key in sorted(inference_gene_map.keys()): # {index : path}
            inference_gene_paths.append(inference_gene_map[key])
        return inference_gene_paths

    def output(self):
        return luigi.LocalTarget(join(self.outdir, prefix(self.rep_genes_inf, 1) + '.genes.list'))

    def run(self):
        with self.output().open('w') as out_list:
            for g in self.input():
                gp = abspath(g.path)
                out_list.write(gp + '\n')
                with open(self.rep_genes_inf, 'a+') as fout, open(gp, 'r') as fin:
                    print(gp)
                    for line in fin:
                        fout.write(line)


class Gene_Writer(luigi.Task):
    gene = luigi.Parameter()
    g_outdir = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(join(self.g_outdir, self.gene.id + '.fasta'))

    def run(self): 
        with self.output().open('w') as out_fasta:
            out_fasta.write(self.gene.format('fasta'))


class MultiStrain_Wrapper(luigi.WrapperTask):
    """Loads strain genomes and coordinates gene BLASTn jobs."""
    rep_genes_inf = luigi.Parameter()
    strain_dir = luigi.Parameter()
    outdir = luigi.Parameter()
    log_outdir = luigi.Parameter(default = '')

    def __init__(self, *args, **kwargs):
        super(MultiStrain_Wrapper, self).__init__(*args, **kwargs)
        self.log_outdir = check_make(self.outdir, 'workflow_logs')

    def requires(self):  
        strain_genomes = [f for f in listdir(self.strain_dir) if not f.startswith('.')]
        processed = []  
        for i in range(len(strain_genomes)):
            processed.append(Strain_Processor(strain = join(self.strain_dir, strain_genomes[i])))
        return processed

    def output(self):
        return luigi.LocalTarget(join(self.log_outdir, prefix(self.rep_genes_inf, 1) + '_blastn.out'))

    def run(self):
        with self.output().open('w') as tmp_out:
            tmp_out.write('Finished')


class Strain_Processor(luigi.Task):
    """Makes a BLASTdb from the strain genome, then BLASTns the representative genome genes against it."""
    strain = luigi.Parameter()
    rep_genes_inf = luigi.Parameter()
    outdir = luigi.Parameter()
    blast_dbs = luigi.Parameter(default = '')
    blast_results = luigi.Parameter(default = '')
    strain_prefix = luigi.Parameter(default = '')
    genes_outdir = luigi.Parameter(default = '')

    def __init__(self, *args, **kwargs):
        super(Strain_Processor, self).__init__(*args, **kwargs)
        self.blast_dbs = check_make(self.outdir, 'blast_dbs')
        self.blastn_results = check_make(self.outdir, 'blastn_results')
        self.strain_prefix = prefix(self.strain, 1)  
        self.genes_outdir = join(self.outdir, 'representative_genes')

    def output(self):
        return luigi.LocalTarget(join(self.outdir, 'workflow_logs', self.strain_prefix + '.out'))

    def run(self):
        strain_db = join(self.blast_dbs, self.strain_prefix)
        strain_outfile = join(self.blastn_results, self.strain_prefix + '.csv')
        blastdb_step = subprocess.run(['makeblastdb', '-in', self.strain, '-out', strain_db, '-dbtype', 'nucl', '-title', self.strain_prefix, '-parse_seqids'])
        blastn_step = subprocess.run(['blastn', '-query', self.rep_genes_inf, '-db', strain_db, '-out', strain_outfile, '-outfmt', '10 qseqid sseqid sseq'])
        inference_gene_paths = [prefix(f, 1) for f in listdir(self.genes_outdir) if not f.startswith('.')]
        with open(strain_outfile, 'r') as sfile:
            for line in sfile:  # qseqid sseqid sseq
                row = line.strip().split(',')
                if row[0] in inference_gene_paths:  # Restricts writing to genes that were used in inference
                    with open(join(self.genes_outdir, row[0] + '.fasta'), 'a+') as gene_fasta:  # qseqid = gene name
                        strain_gene = SeqRecord(seq = Seq(row[2].replace('-', ''), generic_dna),  # sseq = strain aligned sequence
                            id = row[1], name='', description='', dbxrefs=[])                    # sseqid = strain name
                        gene_fasta.write(strain_gene.format('fasta'))
        with self.output().open('w') as tmp_out:
            csv.writer(tmp_out).writerow([self.strain_prefix, blastdb_step.returncode, blastn_step.returncode]) # Now a useful log file to see if any fail at the preprocessing step.


class Gene_Align_Wrapper(luigi.WrapperTask):
    """Coordinates the alignment process and dataframe generation."""
    rep_genes_inf = luigi.Parameter()
    outdir = luigi.Parameter()
    rep_genes_prefix = luigi.Parameter(default = '')

    def __init__(self, *args, **kwargs):
        super(Gene_Align_Wrapper, self).__init__(*args, **kwargs)
        self.rep_genes_prefix = prefix(self.rep_genes_inf, 1)
        align_outdir = check_make(self.outdir, 'gene_alignments')
        df_outdir = check_make(self.outdir, 'gene_dataframes')

    def requires(self):  
        processed = []
        with open(join(self.outdir, prefix(self.rep_genes_inf, 1) + '.genes.list'), 'r') as gl:
            for i, line in enumerate(gl):
                processed.append(Strain_Comp_Maker(gene_index = i, gene = line))  # {gene_file_path : genes_in_inference_index}
        return processed

    def output(self):
        return luigi.LocalTarget(join(self.outdir, self.rep_genes_prefix + '.aligned.list'))

    def run(self):
        with self.output().open('w') as out_list:
            for g in self.input():
                out_list.write(abspath(g.path) + '\n')


class Strain_Comp_Maker(luigi.Task):
    """Convert the multiple sequence alignment to a SNP composition matrix of the strains."""
    gene = luigi.Parameter()
    gene_index = luigi.Parameter()
    outdir = luigi.Parameter()
    align_outdir = luigi.Parameter(default = '')
    df_outdir = luigi.Parameter(default = '')

    def __init__(self, *args, **kwargs):
        super(Strain_Comp_Maker, self).__init__(*args, **kwargs)
        self.align_outdir = join(self.outdir, 'gene_alignments')
        self.df_outdir = join(self.outdir, 'gene_dataframes')

    def requires(self):
        return Gene_Aligner(gene = self.gene, align_outdir = self.align_outdir)

    def output(self):
        return luigi.LocalTarget(join(self.df_outdir, prefix(self.gene, 1) + '.csv'))

    def run(self):
        rep_gene_seq = ''
        strain_ids = []
        aligned_seqs = []
        for curr_gene in SeqIO.parse(self.input().path, 'fasta'):  # For gene in the reference sequence
            if not rep_gene_seq:
                rep_gene_seq = curr_gene.seq
                continue  # Skips the representative gene sequence
            strain_ids.append(curr_gene.id)
            aligned_seqs.append(list(curr_gene.seq))
        align_df = pd.DataFrame(aligned_seqs)

        aligned_snps_names = []
        aligned_snps = [] # List of the SNP composition of each strain, convert to dataframe later
        rep_gene_pos = 1 # Track true position within the representative gene
        for (i, col) in align_df.iteritems():
            rep_gene_allele = rep_gene_seq[int(i)]
            alleles = set(col)
            if rep_gene_allele == '-': # If the representative gene sequence does not contain this position, skip
                continue
            if ('-' not in alleles and len(alleles) > 1) or ('-' in alleles and len(alleles) > 2): 
            # If there are polymorphisms at this alignment position OR some strains have a gap
                for a in sorted(alleles):
                    if a == '-': # If the representative gene sequence does not contain this position, skip
                        continue
                    snp_column = []
                    for j in range(len(strain_ids)): # Check if each strain contains the current allele
                        if col[j] == a:
                            snp_column.append(1)
                        else:
                            snp_column.append(0)
                    aligned_snps_names.append('-'.join([str(rep_gene_pos), a.upper()]))
                    aligned_snps.append(snp_column) # Gene_index-position-allele: [allele_0, allele_1, ..., allele_n]
            rep_gene_pos += 1
        strain_df = pd.DataFrame(aligned_snps, index = aligned_snps_names, columns = strain_ids).T #, columns = strain_ids).T
        # print(strain_df.iloc[1:5,1:5])
        strain_df.to_csv(self.output().path, header = True, index = True)


class Gene_Aligner(luigi.Task):
    """Align all matching sequences together to identify mismatching positions."""
    gene = luigi.Parameter()
    align_outdir = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(join(self.align_outdir, prefix(self.gene, 1) + '.aln.fa'))

    def run(self):
        mafft_cmd = MafftCommandline(input = self.gene, localpair = True, maxiterate = 1000)
        stdout, stderr = mafft_cmd()
        with self.output().open('w') as afile:
            afile.write(stdout)


class Strain_GSTD_Processor(luigi.Task): 
    rep_genes_inf = luigi.Parameter()
    outdir = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(join(self.outdir, prefix(self.rep_genes_inf, 2) + '.preprocessing.out'))

    def run(self):
        yield Gene_Split_Wrapper()
        yield MultiStrain_Wrapper()
        yield Gene_Align_Wrapper()
        with self.output().open('w') as out_file:
            out_file.write('Finished pre-processing gold-standard strain genomes.')


class Strain_Evaluator(luigi.Task):
    rep_genes_inf = luigi.Parameter()    
    outdir = luigi.Parameter()
    doc_topic = luigi.Parameter()
    topic_wrd = luigi.Parameter()
    gstd_freq_file = luigi.Parameter()
    frequency_cutoff = luigi.Parameter(default = 0.01)
    similarity_cutoffs = luigi.Parameter(default = '0.02') # The threshold for a recovered strain is set by the parameter 'similarity_cutoffs'. See the API function (pairwise_manhattan()) for its use.

    def requires(self):
        return Strain_GSTD_Processor()

    def output(self): 
        return luigi.LocalTarget(join(self.outdir, prefix(self.rep_genes_inf, 2) + '.evaluation.out'))

    def run(self):
        infer_freq = pd.read_csv(self.doc_topic, header = 0, index_col = 0)
        gstd_freq_final = pd.read_csv(self.gstd_freq_file, header = 0, index_col = 0)  # TODO Need some way to generate in the future
        infer_snp_raw = make_strains_lda(self.topic_wrd, infer_freq) 
        # Process the former into the strain-SNP dataframe, using the latter to select which strains to take. 

        gstd_snp_raw, tmp = concat_matrices(join(self.outdir, prefix(self.rep_genes_inf, 1) + '.aligned.list'), logger = None) # Reuse previous function. Take in the strain-SNP dataframe. 
        infer_snp_final, gstd_std_final, infer_freq_final = intersect_snps(infer_snp_raw, gstd_snp_raw, infer_freq) 
        # ‘Inferred’ strains and ‘gold-standard’ strains: Format the dataframes such that only the true positive SNP columns are included.  
        # Removes duplicated strains because some strains may not be differentiated with some SNP positions removed. 
        # Report: number of SNPs in infer, number of SNPs in gstd, number of SNPs in common, number of infer strains in final, number of gstd strains in final

        report_list = [] # Summary information about the strain inference
        for t in self.similarity_cutoffs.split(','): # (*) For a certain number of thresholds:
            for p in range(infer_freq_final.shape[0]): # For each pool:
                infer_snp_pool, infer_freq_pool = make_pool(infer_snp_final, infer_freq_final, p, self.frequency_cutoff) 
                gstd_snp_pool, gstd_freq_pool = make_pool(gstd_std_final, gstd_freq_final, p, self.frequency_cutoff) 
                paired_actual, paired_predict, min_strain_dist = pairwise_manhattan(infer_snp_pool, gstd_snp_pool, float(t)) 
                # paired_*: Two arrays consisting of pairs of actual (gold-standard) and predicted (inferred) strains. 
                # min_strain_dist: Minimum Manhattan distances to each of the gold-standard strains
                mcc, num_success = mcc_calculator(paired_actual, paired_predict)
                jsd = jsd_calculator(paired_actual, paired_predict, gstd_freq_pool, infer_freq_pool) 
                report_list.append([t, p, num_success, mcc, jsd, ';'.join(str(x) for x in min_strain_dist)])
                    # Threshold, pool number, number of strains found, mcc, jsd, (min manhattan distances for each of gstd strains)
        report_df = pd.DataFrame(report_list, columns = ['Threshold', 'Pool', 'Num. Strains Rec.', 'MCC', 'JSD', 'Min.  Manh. Distances'])
        report_df.to_csv(self.output().path, header = True)


if __name__ == '__main__':
    luigi.run(main_cls_task=Strain_GSTD_Processor)
