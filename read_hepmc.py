import pyhepmc
import os

os.getcwd()
os.chdir("C:\\Users\\Cornelius\\Projects\\KISS")

with pyhepmc.open("events_sherpa.hepmc") as f:
    event = f.read()

event