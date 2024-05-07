import GraphMeasuring as GrMeas
import GraphPlotting as GrPlot

import argparse as ap

parser = ap.ArgumentParser()

subparsers = parser.add_subparsers(dest = "plotType")

iterationsAnalysis = subparsers.add_parser("iteration")
iterationsAnalysis = subparsers.add_parser("consistency")

# iterationsAnalysis.add_argument()

args = parser.parse_args()

if args.plotType == "iteration":
    for gen_pars in GrMeas.GenerationPars:
        for clusterer in GrMeas.plottable_clusterers:
            clusterers = []
            
            for i in range(1, clusterer["iterations"]+1):
                copy = clusterer.copy()
                copy["iterations"] = i
                copy["label"] = f"Iteration {i}"
                clusterers.append(copy)
            
            title = f"{gen_pars['filename']} with {gen_pars['n_steps']*gen_pars['step_size']/32} turns in {gen_pars['n_steps']} steps\nIteration analysis of {clusterer['label']}"
            print(f"{clusterer['filename']}")
            GrPlot.PlotTestResults(gen_pars, clusterers, title, f"Plots/iterations/iterations_{gen_pars['filename']}_{clusterer['filename']}.pdf")
            
            
            
if args.plotType == "consistency":
    for gen_pars in GrMeas.GenerationPars:
        clusterers = []
        for clusterer in GrMeas.initialisable_clusterers:
            if "Initialised" in clusterer["label"]:
                continue
            
            copy = clusterer.copy()
            clusterers.append(copy)
            
        title = f"{gen_pars['filename']} with {gen_pars['n_steps']*gen_pars['step_size']/32} turns in {gen_pars['n_steps']} steps\nComparing the different consistency Leiden algorithms"
        print(f"{clusterer['filename']}")
        GrPlot.PlotTestResults(gen_pars, clusterers, title, f"Plots/consistency/consistency_{gen_pars['filename']}.pdf", figsize=(6,3.5))
    
