import pandas as pd

from click.testing import CliRunner
from os.path import dirname, join
from os import listdir
from unittest import TestCase

from strain_detangler.cli import split_reference, split_pileup, lda_train, cluster_map

from strain_detangler import (
    entropy_reduce_position_matrix,
)

RUNNER = CliRunner()
C_ACNES_REFERENCE = join(dirname(__file__), 'c_acnes.test.fa')
GENE_1_FILENAME = join(dirname(__file__), 'genes/', 'lcl|NC_006085.1_cds_WP_002515747.1_1.counts.csv')
GENE_2_FILENAME = join(dirname(__file__), 'genes/', 'lcl|NC_006085.1_cds_WP_002517483.1_2.counts.csv')
PILEUP_1_FILENAME = join(dirname(__file__), 'haib17CEM4890_H2NYMCCXY_SL254800.pileup')
PILEUP_2_FILENAME = join(dirname(__file__), 'haib17CEM4890_H75CGCCXY_SL263643.pileup')
MATRIX_1_FILENAME = join(dirname(__file__), 'NC_006085.1_cds_WP_002515220.1_751.counts.txt.gz')
MATRIX_2_FILENAME = join(dirname(__file__), 'NC_006085.1_cds_WP_002516475.1_2279.counts.txt.gz')
MATRIX_LIST_FILENAME = join(dirname(__file__), 'test.reduced.list')
METADATA_FILENAME = join(dirname(__file__), 'metadata.reduced.csv')
DOCTOPIC_FILENAME = join(dirname(__file__), 'model/', 'test.gensim_doctopic.csv')
TOPWORD_FILENAME = join(dirname(__file__), 'model/', 'test.gensim_topwords.csv')


class TestStrainFinding(TestCase):

    def test_1_split_reference(self):
        """Test that two gene read-count files are generated, and 1_1.counts.csv has 6012 SNP columns."""
        RUNNER.invoke(split_reference, ['-o', dirname(__file__), C_ACNES_REFERENCE])
        outfile = pd.read_csv(GENE_1_FILENAME, index_col=0, header=0)
        self.assertEqual(len([name for name in listdir(join(dirname(__file__), 'genes/'))]), 2)
        self.assertEqual(len(outfile.columns), 6012)

    def test_2_split_pileup(self):
        """Test that the row- and column-sums of gene files 1 and 2 are 2 x 2, 1 x 4, 2 x 4, and 2 x 4."""
        RUNNER.invoke(split_pileup, ['-o', dirname(__file__), PILEUP_1_FILENAME])
        RUNNER.invoke(split_pileup, ['-o', dirname(__file__), PILEUP_2_FILENAME])
        gene_1 = pd.read_csv(GENE_1_FILENAME, index_col=0, header=0)
        gene_2 = pd.read_csv(GENE_2_FILENAME, index_col=0, header=0)
        gene_1_rows = gene_1.sum(axis=1)
        gene_2_rows = [count for count in gene_2.sum(axis=1) if count != 0]
        gene_2_cols = [count for count in gene_2.sum(axis=0) if count != 0]
        print(gene_1)
        self.assertEqual(gene_1_rows[0], gene_1_rows[1])
        self.assertEqual(gene_1.sum(axis=0).sum(), 4)
        self.assertEqual(len(gene_2_rows), len(gene_2_cols))
        self.assertEqual(sum(gene_2_rows), 8)

    def test_3_reduce_entropy_reduce(self):
        """Test that we only get 1 column."""
        original = pd.read_csv(MATRIX_1_FILENAME, index_col=0, header=0)
        full_reduced = entropy_reduce_position_matrix(
            original,
            0.1,
            'cosine'
        )
        self.assertEqual(full_reduced.shape[0], original.shape[0])
        self.assertLess(full_reduced.shape[1], original.shape[1])

    def test_3_reduce_entropy_reduce_2(self):
        """Test that we only get 1 column."""
        original = pd.read_csv(MATRIX_2_FILENAME, index_col=0, header=0)
        full_reduced = entropy_reduce_position_matrix(
            original,
            0.1,
            'cosine'
        )
        self.assertEqual(full_reduced.shape[0], original.shape[0])
        self.assertLess(full_reduced.shape[1], original.shape[1])


    def test_4_lda_train(self):
        """Make sure the LDA module works."""
        RUNNER.invoke(lda_train, ['-o', dirname(__file__), MATRIX_LIST_FILENAME])
        RUNNER.invoke(cluster_map, ['-m', METADATA_FILENAME, '-o', dirname(__file__), DOCTOPIC_FILENAME])
        doc_topic = pd.read_csv(DOCTOPIC_FILENAME, index_col=0, header=0)
        topic_word = pd.read_csv(TOPWORD_FILENAME, index_col=0, header=None)
        unique_snps = set()
        for row in topic_word.values:
            for item in row:
                unique_snps.add(item)

        self.assertEqual(len([name for name in listdir(join(dirname(__file__), 'model/'))]), 8)
        self.assertLess(len(doc_topic.sum(axis=0)), 10)  # Check that there are 10 or fewer strains modelled.
        self.assertLess(len(unique_snps), 1286)
            # Check that the number of SNPs represented across all strains is less than or equal to the number of SNPs
            # of non-zero frequency.
