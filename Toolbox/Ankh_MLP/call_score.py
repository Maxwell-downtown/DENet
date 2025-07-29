import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Readin prediction scores and mutations from previous round"
    )
    parser.add_argument(
        "-i1", "--input1",
        required=True,
        help="Path to prediction file from previous round."
    )
    parser.add_argument(
        "-i2", "--input2",
        required=True,
        help="Path to training file used in previous round."
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to output next round training data."
    )
    args = parser.parse_args()

    infile1 = args.input1
    infile2 = args.input2
    outfile = args.output
    
    df = pd.read_csv(infile1, sep='\t')
    df_pre = pd.read_csv(infile2, sep='\t')

    df_sorted = df.sort_values("prediction", ascending=False)
    existing = set(df_pre["mutant"])

    new_rows = df_sorted.loc[
            ~df_sorted["mutant"].isin(existing),["mutant","score"]].head(50)
    combined = pd.concat([df_pre, new_rows], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f'New training data contain {len(combined)} entries in total')

    combined.to_csv(outfile, sep='\t', index=False)

if __name__ == "__main__":
    main()
    
