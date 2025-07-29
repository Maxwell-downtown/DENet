import pandas as pd
import os

# Load the TSV file into a DataFrame
df = pd.read_csv('./raw/0929_pp.tsv', sep='\t')
output_dir = './DMS/0929'
os.makedirs(output_dir, exist_ok=True)
random_num = 1

# Filter the DataFrame for single mutations
# Assuming single mutations have no semicolons
single_mutations_df = df[~df.iloc[:, 0].str.contains(';')]
high_mutations_df = df[df.iloc[:, 0].str.contains(';')]
#double_mutations_df = df[df.iloc[:, 0].str.count(';') == 1]
#triple_mutations_df = df[df.iloc[:, 0].str.count(';') == 2]
#quadru_mutations_df = df[df.iloc[:, 0].str.count(';') == 3]
#penta_mutations_df = df[df.iloc[:, 0].str.count(';') == 4]

single_mutations_df = single_mutations_df.sample(frac=1, random_state=random_num).reset_index(drop=True)
high_mutations_df = high_mutations_df.sample(frac=1, random_state=random_num).reset_index(drop=True)
'''
double_mutations_df = double_mutations_df.sample(frac=1, random_state=random_num).reset_index(drop=True)
triple_mutations_df = triple_mutations_df.sample(frac=1, random_state=random_num).reset_index(drop=True)
quadru_mutations_df = quadru_mutations_df.sample(frac=1, random_state=random_num).reset_index(drop=True)
penta_mutations_df = penta_mutations_df.sample(frac=1, random_state=random_num).reset_index(drop=True)
'''
# Save the single mutations to a new TSV file
single_mutations_df.to_csv('./DMS/0929/sin.tsv', sep='\t', index=False)
high_mutations_df.to_csv('./DMS/0929/high.tsv', sep='\t', index=False)
#double_mutations_df.to_csv('./DMS/0929/double.tsv', sep='\t', index=False)
#triple_mutations_df.to_csv('./DMS/0929/triple.tsv', sep='\t', index=False)
#quadru_mutations_df.to_csv('./DMS/0929/quadru.tsv', sep='\t', index=False)
#penta_mutations_df.to_csv('./DMS/0929/penta.tsv', sep='\t', index=False)

#six_mutations_df = df[df.iloc[:, 0].str.count(';') == 5]
#seven_mutations_df = df[df.iloc[:, 0].str.count(';') == 6]
#eight_mutations_df = df[df.iloc[:, 0].str.count(';') == 7]
#nine_mutations_df = df[df.iloc[:, 0].str.count(';') == 8]
#ten_mutations_df = df[df.iloc[:, 0].str.count(';') == 9]
#eleven_mutations_df = df[df.iloc[:, 0].str.count(';') == 10]
#nine_mutations_df.to_csv('./DMS/0929/nine.tsv', sep='\t', index=False)
#ten_mutations_df.to_csv('./DMS/0929/ten.tsv', sep='\t', index=False)
#eleven_mutations_df.to_csv('./DMS/0929/eleven.tsv', sep='\t', index=False)
