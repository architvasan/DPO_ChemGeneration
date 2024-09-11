import scaffoldgraph as sg
import pandas as pd
#tree = sg.ScaffoldTree.from_smiles_file('my_smiles_file.smi')

network = sg.ScaffoldNetwork.from_smiles_file('data/top_rtcb_hits_100000.smi', progress=True)

nodes = list(network.nodes())

print(nodes)
nodes = [s.replace("MolNode-", "") for s in nodes]

nodes_df = pd.DataFrame({'smiles': nodes})
nodes_df.to_csv('data/rtcb_scaffolds_100000.smi')
