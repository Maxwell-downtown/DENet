import pandas as pd
from scipy.stats import spearmanr
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Readin prediction files for SpearmanC calculation"
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to prediction file."
    )
    args = parser.parse_args()
    
    infile = args.input
    df = pd.read_csv(infile, sep='\t')

    x = df['score']
    y = df['prediction']

    corr, pvalue = spearmanr(x, y, nan_policy='omit')

    print(f"Spearman correlation for {infile}: {corr:.4f}")
    print(f"P-value: {pvalue:.4e}")

if __name__ == "__main__":
    main()
