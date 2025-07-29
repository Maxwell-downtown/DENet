import pandas as pd
import numpy as np
import random
import re
from pathlib import Path

# ─── USER PARAMETERS ─────────────────────────────────────────────────────────

# Number of starting clones
POP0        = 100000
# Mutation probability per sequence, per step
MUT_RATE    = 0.5      
# Maximum number of mutations allowed in any clone
MAX_MUT     = 2        
# Rounds of “evolution” 
N_STEPS     = 6        
# Your wild‐type sequence (string of single‐letter AAs)
WT_SEQ      = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"  

# ─── SETUP ───────────────────────────────────────────────────────────────────

random.seed(42)
np.random.seed(42)

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
score_df = pd.read_csv("./data/GB_score1.tsv", sep="\t")
score_dict = dict(zip(score_df["mutant"], score_df["score"]))

# initialize population
population = [
    {"mutations": [], "mut_str": "", "score": None}
    for _ in range(POP0)
]

# ─── EVOLUTION LOOP ──────────────────────────────────────────────────────────

for step in range(1, N_STEPS + 1):
    reload = 0

    for individual in population:
        # only mutate if below max_mut and random draw indicates “yes”
        if len(individual["mutations"]) < MAX_MUT and random.random() > MUT_RATE:
            for attempt in range(3):
                pos = random.randint(2, len(WT_SEQ))
                wt_aa = WT_SEQ[pos - 1]
                choices = [aa for aa in AA_LIST if aa != wt_aa]
                new_aa = random.choice(choices)
                cand_list = individual["mutations"] + [f"{wt_aa}{pos}{random.choice(choices)}"]
                sorted_list = sorted(
                    cand_list,
                    key=lambda m: int(re.search(r"\d+", m).group()))
                cand_str = ";".join(sorted_list)
                
                # Accept the mutation if fitness score of the corresponding mutant is available
                if cand_str in score_dict:
                    individual["mutations"] = sorted_list
                    break
                else:
                    reload = reload+1
                    
        individual["mut_str"] = ";".join(individual["mutations"]) if individual["mutations"] else ""
        individual["score"] = score_dict.get(individual["mut_str"], float("-1e9"))

    pop_df = pd.DataFrame(population)
    pop_df = pop_df.sort_values("score", ascending=False)

    out_path = Path(f"./data/DEsimulation/step_{step}_All.tsv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pop_df[["mut_str", "score"]].rename(columns={"mut_str":"mutant"}).to_csv(out_path, sep="\t", index=False)
    print(f"Step {step}: wrote {len(population)} sequence to {out_path}")
    print(f'Reload number of trial is {reload}')

print("Simulation complete.")
