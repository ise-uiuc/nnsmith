import sys
import pandas as pd
import os

fuzz_report_rt = sys.argv[1]
df = pd.read_pickle(os.path.join(fuzz_report_rt, "profile.pkl"))
mstr = [f"{i}.onnx" for i in range(len(df))]
df["model_name"] = mstr
df["gen_t"] = df["gen_t"]
df1 = df[["gen_t", "model_name"]]
df1.to_csv(os.path.join(fuzz_report_rt, "gentime.csv"), index=False, header=False)
