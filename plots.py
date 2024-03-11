import os
import pyhepmc
import graphviz
import matplotlib.pyplot as plt
import numpy as np

# check current working directory
os.getcwd()

# %%
bat_events = "output/events_bat.hepmc"
rambo_events = "output/events_sherpa.hepmc"
sherpa_events = "standalone/sherpa_events.hepmc"

# %%
bat = {"events": bat_events, "label": "BAT", "color": "#1f77b4"}
rambo = {"events": rambo_events, "label": "RAMBO", "color": "#2ca02c"}
sherpa = {"events": sherpa_events, "label": "SHERPA", "color": "#ff7f0e"}

# %%
# -------- Functions for computing observables -------------------------------------------
def leading_pT(event):
    pT_max = max([p.momentum.pt() for p in event.particles])
    return pT_max

def is_e(p): # is electron or positron
    return abs(p.pid)==11

def invariant_lepton_mass(event):
    e_lepton_momenta = [p.momentum for p in event.particles if is_e(p)]
    m = (e_lepton_momenta[0] + e_lepton_momenta[1]).m()
    return m


def get_observables(sample):
    invariant_mass = []
    weights = []

    with pyhepmc.open(sample["events"]) as f:
        for event in f:
            invariant_mass.append(invariant_lepton_mass(event))
            weights.append(event.weights[0])
        xs = event.attributes["GenCrossSection"].astype(float)
        sample["xs"] = xs

    sample["invariant_mass"] = np.array(invariant_mass)
    sample["weights"] = np.array(weights)


print("Get observables")
# %%
get_observables(bat)

# %%
get_observables(rambo)

# %%
get_observables(sherpa)
print("Got all observables")

# print("Get additional rambo sample")
# rambo2 = {"events": "output/events_sherpa2.hepmc", "label": "RAMBO", "color": "#2ca02c"}
# get_observables(rambo2)

# r_ivm = np.concatenate((rambo["invariant_mass"], rambo2["invariant_mass"]))
# print(rambo["xs"], " ", rambo2["xs"])
# r_w = np.concatenate((rambo["weights"], rambo2["weights"]))


# %%
# Define your scaling factor
thickness_scaling = 1.5

# Update the default rc settings
plt.rcParams['lines.linewidth'] = 1.5 * thickness_scaling
plt.rcParams['patch.linewidth'] = 1.5 * thickness_scaling
plt.rcParams['lines.markersize'] = 6 * thickness_scaling 
plt.rcParams['font.size'] = 12 * thickness_scaling 

# %%
#-------- Plot invariant mass ------------------------------------------------------------
print("Start plotting")
density = True
plt.figure(figsize=(8 * thickness_scaling, 6 * thickness_scaling)) 
x_range = (80, 150)

plt.hist(bat["invariant_mass"], bins=300,  histtype='step', label=bat["label"], color=bat["color"], density=density, alpha=0.9, range=x_range) 
#plt.hist(bat["invariant_mass"], bins=200,  histtype='step', label=bat["label"], color="blue", density=density, alpha=0.9, range=x_range) 

#plt.hist(rambo["invariant_mass"], weights=rambo["weights"]*rambo["xs"]/rambo["weights"].sum(), bins=300,  histtype='step', label=rambo["label"], color=rambo["color"], density=density, alpha=0.9, range=x_range) 
#plt.hist(r_ivm, weights=r_w*rambo["xs"]/r_w.sum(), bins=300,  histtype='step', label=rambo["label"], color=rambo["color"], density=density, alpha=0.9, range=x_range) 


plt.hist(sherpa["invariant_mass"],  weights=sherpa["weights"]*sherpa["xs"]/sherpa["weights"].sum(), bins=300,  histtype='step', label=sherpa["label"], color=sherpa["color"], density=density, alpha=0.9, range=x_range) 
#plt.hist(sherpa["invariant_mass"], bins=200,  histtype='step', label=sherpa["label"], color=sherpa["color"], density=density, alpha=0.9, range=x_range) 
#plt.hist(sherpa["invariant_mass"], bins=200,  histtype='step', label=sherpa["label"], color="green", density=density, alpha=0.9, range=x_range) 


#plt.yscale('log')
plt.legend(loc='best')

plt.xlabel('$m(e^+e^-)$ in GeV')
plt.ylabel('$\\frac{d\\sigma}{d m_{ee}}$ (normalized)')

plt.title('$ gg \\rightarrow e^+ e^- d \\bar d$ - Invariant lepton mass')

# Display the plot
plt.savefig("output/invariant_mass_new.pdf")
# %%
