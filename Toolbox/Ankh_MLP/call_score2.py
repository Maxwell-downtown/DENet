import pandas as pd
import argparse
import re

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
        "-r0", "--refer",
        required=True,
        help="Real function score from experiment for reference."
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to output next round training data."
    )
    args = parser.parse_args()

    infile1 = args.input1
    infile2 = args.input2
    ref = args.refer
    outfile = args.output

    df = pd.read_csv(infile1, sep='\t')
    df_pre = pd.read_csv(infile2, sep='\t')
    df_double = pd.read_csv(ref, sep='\t')

    def get_pos(m):
        return int(re.search(r'\d+', m).group())

    def sort_mutant_string(mut_str):
        parts = mut_str.split(';')
        parts_sorted = sorted(parts, key=get_pos)
        return ";".join(parts_sorted)

    df_double['mutant'] = df_double['mutant'].apply(sort_mutant_string)
    score_map = dict(zip(df_double['mutant'], df_double['score']))

    df_sorted = df.sort_values("prediction", ascending=False)
    existing = set(df_pre["mutant"])

    new_rows = df_sorted.loc[
            ~df_sorted["mutant"].isin(existing),["mutant","score"]].head(50)
    new_rows['score'] = new_rows['mutant'].map(score_map)
    combined = pd.concat([df_pre, new_rows], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f'New training data contain {len(combined)} entries in total')

    combined.to_csv(outfile, sep='\t', index=False)

if __name__ == "__main__":
    main()
    
