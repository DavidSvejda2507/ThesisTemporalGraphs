import GraphMeasuring as GrMeas
import GraphPlotting as GrPlot

import argparse as ap

parser = ap.ArgumentParser()

subparsers = parser.add_subparsers(dest = "plotType")

iterationsAnalysis = subparsers.add_parser("iteration")
iterationsAnalysis = subparsers.add_parser("consistency")
iterationsAnalysis = subparsers.add_parser("initialisation")
iterationsAnalysis = subparsers.add_parser("temporal")

# iterationsAnalysis.add_argument()

args = parser.parse_args()

def title_str(gen_pars):
    return f"{gen_pars['filename']} with {gen_pars['n_steps']*gen_pars['step_size']/32} turns in {gen_pars['n_steps']} steps"

if args.plotType == "iteration":
    for gen_pars in GrMeas.GenerationPars:
        for clusterer in GrMeas.plottable_clusterers:
            clusterers = []
            
            for i in range(1, clusterer["iterations"]+1):
                copy = clusterer.copy()
                copy["iterations"] = i
                copy["label"] = None
                # copy["label"] = f"Iteration {i}"
                clusterers.append(copy)
            
            title = title_str(gen_pars) + f"\nIteration analysis of {clusterer['label']}"
            # print(f"{clusterer['filename']}")
            GrPlot.PlotTestResults(gen_pars, clusterers, title, f"Plots/iterations/iterations_{gen_pars['filename']}_{clusterer['filename']}.pdf", figsize=(6,3.5))
            
            
            
if args.plotType == "consistency":
    for gen_pars in GrMeas.GenerationPars:
        clusterers = []
        for clusterer in GrMeas.initialisable_clusterers:
            if "Initialised" in clusterer["label"]:
                continue
            
            copy = clusterer.copy()
            clusterers.append(copy)
            
        title = title_str(gen_pars) + f"\nComparing the different consistency Leiden algorithms"
        # print(f"{clusterer['filename']}")
        GrPlot.PlotTestResults(gen_pars, clusterers, title, f"Plots/consistency/consistency_{gen_pars['filename']}.pdf", figsize=(6,3.5))
    

if args.plotType == "initialisation":
    for gen_pars in GrMeas.GenerationPars:
        clusterers1 = []
        clusterers2 = []
        clusterers_a = []
        clusterers_b = []
        for clusterer in GrMeas.initialisable_clusterers:
            copy = clusterer.copy()
            if "Leiden 2" in clusterer["label"]:
                clusterers2.append(copy)
            else:
                clusterers1.append(copy)
            if clusterer["label"][-1]=="a":
                clusterers_a.append(copy)
            else:
                clusterers_b.append(copy)
            
            
        title1 = title_str(gen_pars) + f"\nComparing the (un)initialised consistency Leiden 1 algorithms"
        title2 = title_str(gen_pars) + f"\nComparing the (un)initialised consistency Leiden 2 algorithms"
        title_a = title_str(gen_pars) + f"\nComparing the (un)initialised consistency Leiden a algorithms"
        title_b = title_str(gen_pars) + f"\nComparing the (un)initialised consistency Leiden b algorithms"
        # print(f"{clusterer['filename']}")
        GrPlot.PlotTestResults(gen_pars, clusterers1, title1, f"Plots/initialisation/initialisation1_{gen_pars['filename']}.pdf", figsize=(6,3.5))
        GrPlot.PlotTestResults(gen_pars, clusterers2, title2, f"Plots/initialisation/initialisation2_{gen_pars['filename']}.pdf", figsize=(6,3.5))
        GrPlot.PlotTestResults(gen_pars, clusterers_a, title_a, f"Plots/initialisation/initialisationa_{gen_pars['filename']}.pdf", figsize=(6,3.5))
        GrPlot.PlotTestResults(gen_pars, clusterers_b, title_b, f"Plots/initialisation/initialisationb_{gen_pars['filename']}.pdf", figsize=(6,3.5))
    
            
if args.plotType == "temporal":
    for gen_pars in GrMeas.GenerationPars:
        clusterers = []
        for clusterer in GrMeas.plottable_clusterers:
            if clusterer["label"] in ["Temporal Leiden", "Initialised consistency Leiden a", "Consistency Leiden a"]:
                copy = clusterer.copy()
                clusterers.append(copy)
            
        title = title_str(gen_pars) + f"\nComparing the consistency and temporal Leiden algorithms"
        # print(f"{clusterer['filename']}")
        GrPlot.PlotTestResults(gen_pars, clusterers, title, f"Plots/temporal/temporal_{gen_pars['filename']}.pdf", figsize=(6,3.5))