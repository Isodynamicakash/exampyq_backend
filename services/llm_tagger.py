"""
services/llm_tagger.py
======================
LLM-powered chapter + topic + difficulty classifier for JEE / NEET questions.
Uses OpenAI gpt-4o-mini — fast, cheap, accurate.

HOW IT WORKS:
  - Full taxonomy hardcoded below:
      JEE  — Physics: 28ch/424t, Chemistry: 27ch/447t, Mathematics: 24ch/320t
      NEET — Biology: 38ch/600+t  (Class 11 + 12, Botany + Zoology)
  - Subject-specific chapter+topic list sent to LLM — closed list, no hallucination
  - LLM returns chapter NUMBER (from numbered list) + exact topic name + difficulty
  - Multi-fallback matching: number → exact name → case-insensitive → partial match
  - If question spans multiple topics, picks the PRIMARY (most central) one
  - DB taxonomy merged at runtime so any manually added chapters are picked up
  - If subject is unknown/empty/unrecognised, the FULL taxonomy across all four
    subjects is sent so the LLM self-identifies the subject — no wrong-subject fallback

COST:  ~$0.002 per 25-question paper (gpt-4o-mini pricing)
SPEED: All questions tagged concurrently in ~2 seconds
"""

import asyncio
import json
import logging
import os
import re

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

# Sentinel used to route questions whose subject could not be resolved.
# These questions get the full cross-subject taxonomy so the LLM picks the
# correct subject itself instead of defaulting to Physics.
UNKNOWN_SUBJECT = "__UNKNOWN__"


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLETE JEE + NEET TAXONOMY
# Physics: 28 chapters, 424 topics
# Chemistry: 27 chapters, 447 topics
# Mathematics: 24 chapters, 320 topics
# Biology: 38 chapters, 600+ topics  (NEET — Class 11 + 12, Botany + Zoology)
# ═══════════════════════════════════════════════════════════════════════════════

_TAXONOMY: dict = {
    "Physics": {
        "Units and Measurements": [
            "Dimensional Analysis", "SI Units and Conversions", "Errors in Measurement",
            "Significant Figures", "Vernier Callipers and Screw Gauge",
            "Dimensions of Physical Quantities", "Dimensional Formulae",
            "Applications of Dimensional Analysis", "Random and Systematic Errors",
            "Least Count and Precision",
        ],
        "Motion in a Straight Line": [
            "Kinematics Equations", "Displacement Velocity Acceleration",
            "Uniform and Non-Uniform Motion", "Graphs of Motion",
            "Relative Motion in 1D", "Free Fall", "Motion under Gravity",
            "Instantaneous Velocity and Acceleration",
        ],
        "Motion in a Plane": [
            "Vectors and Scalars", "Vector Addition and Subtraction",
            "Resolution of Vectors", "Projectile Motion",
            "Horizontal Projectile", "Oblique Projectile",
            "Uniform Circular Motion", "Centripetal Acceleration",
            "Relative Velocity in 2D", "Angular Displacement Velocity Acceleration",
        ],
        "Laws of Motion": [
            "Newtons First Law Inertia", "Newtons Second Law",
            "Newtons Third Law", "Friction Static and Kinetic",
            "Angle of Friction and Repose", "Free Body Diagrams",
            "Pulley and Constraint Motion", "Connected Bodies",
            "Pseudo Force and Non-Inertial Frames", "Equilibrium of Forces",
            "Banking of Roads", "Motion on Rough Inclined Plane",
            "Spring Force", "Tension in Strings",
        ],
        "Work Energy and Power": [
            "Work Done by Constant Force", "Work Done by Variable Force",
            "Work-Energy Theorem", "Kinetic Energy",
            "Potential Energy Gravitational", "Potential Energy Spring",
            "Conservation of Mechanical Energy", "Power",
            "Elastic Collision 1D and 2D", "Inelastic Collision",
            "Coefficient of Restitution", "Centre of Mass Frame",
            "Work Done by Friction", "Non-Conservative Forces",
        ],
        "System of Particles and Rotational Motion": [
            "Centre of Mass of System", "Centre of Mass of Continuous Bodies",
            "Momentum Conservation", "Moment of Inertia",
            "Parallel Axes Theorem", "Perpendicular Axes Theorem",
            "Radius of Gyration", "Torque", "Angular Momentum",
            "Conservation of Angular Momentum", "Rolling Without Slipping",
            "Rolling on Inclined Plane", "Angular Kinematics",
            "Rotational Kinetic Energy", "Toppling and Tipping",
        ],
        "Gravitation": [
            "Newtons Law of Gravitation", "Gravitational Field",
            "Gravitational Potential", "Gravitational Potential Energy",
            "Orbital Velocity", "Escape Velocity",
            "Keplers First Law", "Keplers Second Law", "Keplers Third Law",
            "Geostationary and Polar Satellites", "Binding Energy of Satellite",
            "Variation of g with Altitude", "Variation of g with Depth",
            "Variation of g with Latitude", "Weightlessness",
        ],
        "Mechanical Properties of Solids": [
            "Stress and Strain", "Youngs Modulus", "Bulk Modulus",
            "Shear Modulus", "Elastic Potential Energy", "Poissons Ratio",
            "Stress-Strain Curve", "Elastic Limit and Yield Point",
            "Ductile and Brittle Materials", "Hookes Law",
        ],
        "Mechanical Properties of Fluids": [
            "Pressure and Pascals Law", "Gauge Pressure and Absolute Pressure",
            "Archimedes Principle", "Buoyancy and Floatation",
            "Equation of Continuity", "Bernoullis Theorem",
            "Torricelli Theorem Speed of Efflux", "Venturimeter",
            "Viscosity and Stokes Law", "Terminal Velocity",
            "Streamline and Turbulent Flow", "Reynolds Number",
            "Surface Tension", "Excess Pressure in Bubble and Drop",
            "Capillarity and Capillary Rise", "Angle of Contact",
        ],
        "Thermal Properties of Matter": [
            "Temperature Scales Celsius Kelvin Fahrenheit",
            "Linear Expansion", "Area Expansion", "Volume Expansion",
            "Anomalous Expansion of Water", "Specific Heat Capacity",
            "Calorimetry", "Latent Heat Fusion and Vaporisation",
            "Thermal Conduction", "Thermal Resistance",
            "Convection", "Radiation and Blackbody",
            "Stefans Law", "Wiens Displacement Law",
            "Newtons Law of Cooling", "Solar Constant",
            "Emissivity and Absorptivity",
        ],
        "Thermodynamics": [
            "Zeroth Law and Thermal Equilibrium", "First Law of Thermodynamics",
            "Internal Energy", "Isothermal Process", "Adiabatic Process",
            "Isochoric Process", "Isobaric Process", "Polytropic Process",
            "PV Diagrams", "Second Law of Thermodynamics", "Entropy",
            "Carnot Engine", "Carnot Efficiency", "Heat Engines",
            "Refrigerators and Heat Pumps", "Efficiency of Engine", "COP of Refrigerator",
        ],
        "Kinetic Theory": [
            "Ideal Gas Equation", "Pressure of an Ideal Gas",
            "RMS Speed", "Mean Speed", "Most Probable Speed",
            "Degrees of Freedom", "Law of Equipartition of Energy",
            "Mean Free Path", "Specific Heat of Gases Cv and Cp",
            "Ratio of Specific Heats Gamma", "Real Gases",
            "Avogadros Number and Boltzmann Constant",
        ],
        "Oscillations": [
            "Simple Harmonic Motion Definition", "Displacement Velocity Acceleration in SHM",
            "Energy in SHM", "Spring-Mass System", "Simple Pendulum",
            "Compound Pendulum", "Torsional Pendulum", "Phase in SHM",
            "Superposition of SHM", "Damped Oscillations",
            "Forced Oscillations", "Resonance",
            "Time Period and Frequency", "Angular SHM",
        ],
        "Waves": [
            "Wave Equation and Wave Parameters", "Transverse Waves",
            "Longitudinal Waves", "Speed of Sound in Medium",
            "Speed of Sound in Gas Newton Laplace",
            "Superposition Principle", "Interference of Waves",
            "Beats and Beat Frequency", "Standing Waves in Strings",
            "Harmonics and Overtones in String",
            "Standing Waves in Open Pipe", "Standing Waves in Closed Pipe",
            "Harmonics in Pipes", "Doppler Effect",
            "Reflection and Transmission of Waves", "Resonance in Sound",
        ],
        "Electrostatics": [
            "Coulombs Law", "Principle of Superposition",
            "Electric Field Due to Point Charge", "Electric Field Due to Dipole",
            "Electric Field Lines", "Gauss Law",
            "Electric Field of Infinite Sheet",
            "Electric Field of Infinite Line Charge",
            "Electric Field of Uniformly Charged Sphere",
            "Electric Potential Due to Point Charge",
            "Electric Potential Due to Dipole",
            "Equipotential Surfaces",
            "Relation Between Electric Field and Potential",
            "Potential Energy of System of Charges",
            "Conductors in Electrostatic Field",
            "Capacitance of Parallel Plate Capacitor",
            "Capacitance of Spherical Capacitor",
            "Capacitance of Cylindrical Capacitor",
            "Series and Parallel Combination of Capacitors",
            "Dielectrics and Polarisation",
            "Energy Stored in Capacitor",
            "Force between Capacitor Plates",
            "Van de Graaff Generator",
        ],
        "Current Electricity": [
            "Electric Current and Drift Velocity", "Ohms Law",
            "Resistance and Resistivity",
            "Temperature Dependence of Resistance",
            "Series Combination of Resistors",
            "Parallel Combination of Resistors",
            "Kirchhoffs Current Law", "Kirchhoffs Voltage Law",
            "Wheatstone Bridge", "Metre Bridge",
            "Potentiometer Principle", "Potentiometer Applications",
            "EMF and Internal Resistance", "Terminal Voltage and Current",
            "Cells in Series and Parallel",
            "Heating Effect Joules Law", "Electric Power",
        ],
        "Moving Charges and Magnetism": [
            "Magnetic Force on Moving Charge",
            "Motion of Charge in Uniform Magnetic Field", "Cyclotron",
            "Magnetic Force on Current Carrying Conductor",
            "Biot Savart Law", "Magnetic Field due to Straight Wire",
            "Magnetic Field at Centre of Circular Loop",
            "Magnetic Field on Axis of Circular Loop",
            "Amperes Circuital Law", "Magnetic Field due to Solenoid",
            "Magnetic Field due to Toroid",
            "Force between Two Parallel Wires",
            "Torque on Current Loop in Magnetic Field",
            "Moving Coil Galvanometer",
            "Conversion to Ammeter", "Conversion to Voltmeter",
        ],
        "Magnetism and Matter": [
            "Bar Magnet and Magnetic Dipole", "Magnetic Dipole Moment",
            "Magnetic Field on Axial Line of Bar Magnet",
            "Magnetic Field on Equatorial Line of Bar Magnet",
            "Torque on Magnetic Dipole", "Earths Magnetic Field",
            "Magnetic Declination and Dip",
            "Diamagnetic Materials", "Paramagnetic Materials",
            "Ferromagnetic Materials", "Hysteresis Loop",
            "Retentivity and Coercivity", "Curie Temperature",
            "Magnetic Susceptibility",
        ],
        "Electromagnetic Induction": [
            "Magnetic Flux", "Faradays First Law", "Faradays Second Law",
            "Lenzs Law", "Motional EMF",
            "EMF of Rod Moving in Magnetic Field",
            "Eddy Currents", "Self Inductance",
            "Self Inductance of Solenoid", "Mutual Inductance",
            "Coefficient of Coupling", "Energy Stored in Inductor",
            "Growth of Current in LR Circuit",
            "Decay of Current in LR Circuit",
        ],
        "Alternating Current": [
            "AC Generator", "Instantaneous and Peak Values",
            "RMS Values of AC", "Phasors and Phasor Diagrams",
            "Purely Resistive AC Circuit",
            "Purely Inductive AC Circuit", "Inductive Reactance",
            "Purely Capacitive AC Circuit", "Capacitive Reactance",
            "Series RL Circuit", "Series RC Circuit",
            "Series RLC Circuit", "Impedance and Phase Angle",
            "Resonance in Series RLC", "Quality Factor Q",
            "Power in AC Circuit", "Power Factor", "Wattless Current",
            "Transformer", "Energy Losses in Transformer",
        ],
        "Electromagnetic Waves": [
            "Displacement Current", "Maxwells Equations",
            "EM Wave Properties", "Speed of EM Waves in Vacuum",
            "EM Spectrum Radio Waves", "EM Spectrum Microwaves",
            "EM Spectrum Infrared", "EM Spectrum Visible Light",
            "EM Spectrum UV Rays", "EM Spectrum X-Rays",
            "EM Spectrum Gamma Rays", "Energy and Momentum of EM Waves",
        ],
        "Ray Optics": [
            "Laws of Reflection", "Image Formation by Plane Mirror",
            "Image Formation by Concave Mirror", "Image Formation by Convex Mirror",
            "Mirror Formula", "Magnification in Mirrors",
            "Refraction at Plane Surface", "Snells Law",
            "Absolute and Relative Refractive Index",
            "Total Internal Reflection", "Critical Angle", "Optical Fibre",
            "Refraction at Spherical Surfaces", "Lens Makers Formula",
            "Thin Lens Formula", "Magnification in Lenses",
            "Power of Lens", "Combination of Lenses and Mirrors",
            "Refraction through Prism", "Angle of Deviation in Prism",
            "Minimum Deviation", "Dispersion by Prism",
            "Scattering of Light", "Simple Microscope",
            "Compound Microscope", "Astronomical Telescope",
            "Reflecting Telescope", "Human Eye and Defects of Vision",
        ],
        "Wave Optics": [
            "Huygens Principle and Wave Front",
            "Reflection using Huygens Principle",
            "Refraction using Huygens Principle",
            "Coherent Sources", "Youngs Double Slit Experiment",
            "Fringe Width and Fringe Pattern",
            "Intensity Distribution in YDSE",
            "Change of Phase on Reflection",
            "Thin Film Interference", "Single Slit Diffraction",
            "Diffraction Pattern",
            "Resolving Power of Microscope",
            "Resolving Power of Telescope",
            "Polarisation by Reflection", "Brewsters Law",
            "Polaroids and Malus Law", "Polarisation by Scattering",
        ],
        "Dual Nature of Radiation and Matter": [
            "Photoelectric Effect", "Einsteins Photoelectric Equation",
            "Work Function", "Threshold Frequency", "Stopping Potential",
            "Effect of Intensity and Frequency", "Photon Momentum and Energy",
            "de Broglie Hypothesis", "de Broglie Wavelength of Particles",
            "Davisson Germer Experiment", "Electron Emission Types",
        ],
        "Atoms": [
            "Rutherfords Scattering Experiment", "Rutherfords Model Limitations",
            "Bohrs Postulates", "Bohrs Model of Hydrogen",
            "Radius of Bohrs Orbit", "Velocity and Energy in Bohrs Orbit",
            "Energy Levels of Hydrogen", "Hydrogen Spectrum Series",
            "Lyman Series", "Balmer Series", "Paschen Series",
            "Brackett and Pfund Series",
            "Atomic Number and Mass Number", "Franck Hertz Experiment",
        ],
        "Nuclei": [
            "Nuclear Size and Nuclear Density", "Nuclear Binding Energy",
            "Mass Defect", "Binding Energy Per Nucleon",
            "Stability of Nucleus", "Alpha Decay", "Beta Decay",
            "Gamma Decay", "Half Life", "Mean Life",
            "Radioactive Decay Law", "Radioactive Equilibrium",
            "Nuclear Fission Chain Reaction", "Nuclear Reactor",
            "Nuclear Fusion", "Q-Value of Nuclear Reaction",
        ],
        "Semiconductor Electronics": [
            "Energy Bands in Solids",
            "Conductors Insulators Semiconductors",
            "Intrinsic Semiconductor",
            "Extrinsic Semiconductor n-type",
            "Extrinsic Semiconductor p-type",
            "p-n Junction Formation",
            "p-n Junction Forward Bias", "p-n Junction Reverse Bias",
            "Half Wave Rectifier", "Full Wave Rectifier",
            "Zener Diode and Voltage Regulation",
            "Photodiode", "LED", "Solar Cell",
            "Transistor Action",
            "Transistor as Amplifier CE Configuration",
            "Transistor as Switch", "Transistor Characteristics",
            "Logic Gates AND OR NOT", "Logic Gates NAND NOR XOR",
            "Boolean Algebra and De Morgans Theorem",
            "Combination of Gates",
        ],
        "Communication Systems": [
            "Elements of Communication System", "Bandwidth of Signals",
            "Propagation of EM Waves", "Modulation AM and FM",
            "AM Demodulation", "Need for Modulation",
            "Space Wave Ground Wave Sky Wave",
            "Satellite Communication", "Optical Fibre Communication",
        ],
    },

    "Chemistry": {
        "Some Basic Concepts of Chemistry": [
            "Mole Concept", "Avogadros Number",
            "Atomic Mass and Molecular Mass", "Equivalent Weight",
            "Stoichiometry", "Limiting Reagent",
            "Percentage Composition", "Empirical Formula", "Molecular Formula",
            "Laws of Chemical Combination",
            "Mole-Mole Relationship",
            "Concentration Terms Molarity Molality Normality",
        ],
        "Structure of Atom": [
            "Thomson Model", "Rutherford Model",
            "Limitations of Rutherfords Model",
            "Electromagnetic Radiation", "Atomic Spectrum",
            "Bohrs Model", "Energy of Bohrs Orbit",
            "Failure of Bohrs Model", "de Broglie Relation",
            "Heisenberg Uncertainty Principle",
            "Quantum Mechanical Model", "Schrodinger Wave Equation",
            "Quantum Numbers n l m s", "Orbital Shapes s p d f",
            "Electronic Configuration", "Aufbau Principle",
            "Hunds Rule", "Paulis Exclusion Principle",
        ],
        "Classification of Elements and Periodicity": [
            "Modern Periodic Law", "Long Form of Periodic Table",
            "s p d f Block Elements",
            "Atomic Radius", "Ionic Radius", "Covalent Radius",
            "Ionization Enthalpy First and Second",
            "Electron Gain Enthalpy", "Electronegativity Pauling Scale",
            "Valence Electrons and Valency",
            "Periodic Trends in Oxidation State",
            "Diagonal Relationship",
            "Anomalous Properties of Second Period",
            "Metallic and Non-Metallic Character",
        ],
        "Chemical Bonding and Molecular Structure": [
            "Kossel Lewis Approach", "Ionic Bond Formation",
            "Lattice Energy Born Haber Cycle",
            "Covalent Bond Formation", "Lewis Dot Structures",
            "Formal Charge", "VSEPR Theory",
            "Linear and Angular Shapes",
            "Trigonal Planar and Pyramidal",
            "Tetrahedral and See-Saw",
            "Octahedral and Square Planar",
            "Valence Bond Theory",
            "Orbital Overlap Sigma and Pi Bonds",
            "sp sp2 sp3 Hybridisation",
            "sp3d sp3d2 Hybridisation",
            "Resonance Structures",
            "Molecular Orbital Theory",
            "Bond Order from MO Theory",
            "Bonding in O2 N2 F2 CO",
            "Hydrogen Bond Intermolecular",
            "Hydrogen Bond Intramolecular",
            "van der Waals Forces", "Dipole Moment",
            "Bond Length Bond Angle Bond Energy",
        ],
        "States of Matter": [
            "Gas Laws Boyles Charles Gay Lussac",
            "Ideal Gas Equation",
            "Daltons Law of Partial Pressure",
            "Diffusion and Effusion Grahams Law",
            "Kinetic Molecular Theory of Gases",
            "Maxwell Boltzmann Distribution",
            "Average RMS Most Probable Speed",
            "Real Gases Deviation from Ideal",
            "van der Waals Equation", "Boyle Temperature",
            "Critical Constants", "Liquefaction of Gases",
            "Intermolecular Forces",
            "Vapour Pressure and Boiling Point",
            "Surface Tension Viscosity",
        ],
        "Thermodynamics": [
            "System Surroundings and Boundary",
            "Intensive and Extensive Properties",
            "State Functions and Path Functions",
            "Internal Energy", "Enthalpy H and Relation to U",
            "First Law of Thermodynamics",
            "Heat Capacity at Constant Volume",
            "Heat Capacity at Constant Pressure",
            "Hesss Law of Constant Heat Summation",
            "Enthalpy of Formation", "Enthalpy of Combustion",
            "Enthalpy of Atomisation", "Bond Enthalpy",
            "Enthalpy of Neutralisation", "Enthalpy of Dissolution",
            "Entropy and Disorder",
            "Second Law of Thermodynamics",
            "Gibbs Free Energy", "Spontaneity Criteria",
            "Third Law Absolute Entropy",
        ],
        "Equilibrium": [
            "Law of Mass Action", "Equilibrium Constant Kc",
            "Equilibrium Constant Kp", "Relation between Kp and Kc",
            "Homogeneous Equilibrium", "Heterogeneous Equilibrium",
            "Le Chateliers Principle",
            "Effect of Concentration on Equilibrium",
            "Effect of Pressure on Equilibrium",
            "Effect of Temperature on Equilibrium",
            "Degree of Dissociation",
            "Ionic Equilibrium", "Bronsted Lowry Theory",
            "Lewis Theory", "pH and pOH Scale",
            "Strong and Weak Acids Bases", "Ka and Kb",
            "Ostwalds Dilution Law",
            "Buffer Solutions Henderson Equation",
            "Solubility Product Ksp",
            "Common Ion Effect", "Hydrolysis of Salts",
            "Indicators and Neutralisation",
        ],
        "Redox Reactions": [
            "Oxidation Number Rules",
            "Oxidation and Reduction",
            "Oxidising and Reducing Agents",
            "Balancing Redox by Oxidation Number",
            "Balancing Redox by Half Reaction Method",
            "Disproportionation Reactions",
        ],
        "Hydrogen": [
            "Position of Hydrogen in Periodic Table",
            "Isotopes of Hydrogen Protium Deuterium Tritium",
            "Preparation of Dihydrogen",
            "Properties of Hydrogen",
            "Structure and Properties of Water",
            "Hard Water and Softening",
            "Hydrogen Peroxide Structure Properties",
            "Heavy Water", "Hydrides Ionic Covalent Metallic",
        ],
        "The s-Block Elements": [
            "General Properties of Alkali Metals",
            "Anomalous Properties of Lithium",
            "Sodium Hydroxide Preparation and Properties",
            "Sodium Chloride Uses",
            "Sodium Carbonate Solvay Process",
            "Sodium Bicarbonate",
            "Biological Importance of Na and K",
            "General Properties of Alkaline Earth Metals",
            "Anomalous Properties of Beryllium",
            "Diagonal Relationship Be Mg",
            "Calcium Oxide Hydroxide Carbonate",
            "Plaster of Paris",
            "Biological Importance of Mg and Ca",
        ],
        "The p-Block Elements": [
            "Boron Family Group 13", "Borax and Boric Acid",
            "Diborane Structure and Properties",
            "Allotropes of Boron", "Carbon Family Group 14",
            "Allotropes of Carbon Diamond Graphite Fullerene",
            "Oxides of Carbon CO CO2",
            "Silicon and Silicates", "Nitrogen Family Group 15",
            "Allotropes of Phosphorus", "Oxoacids of Phosphorus",
            "Ammonia Preparation and Properties",
            "Nitric Acid Preparation and Properties",
            "Oxygen Family Group 16", "Allotropes of Sulphur",
            "Oxoacids of Sulphur",
            "Sulphuric Acid Contact Process",
            "Ozone Structure and Properties",
            "Halogens Group 17",
            "Preparation and Properties of Halogens",
            "Hydrogen Halides", "Interhalogen Compounds",
            "Oxoacids of Halogens",
            "Noble Gases Group 18", "Xenon Compounds",
        ],
        "The d and f Block Elements": [
            "General Properties of Transition Metals",
            "Electronic Configuration of d Block",
            "Variable Oxidation States d Block",
            "Atomic and Ionic Radii d Block",
            "Ionization Enthalpy d Block",
            "Magnetic Properties d Block",
            "Colour of Transition Metal Compounds",
            "Catalytic Properties d Block",
            "Interstitial Compounds", "Alloy Formation",
            "Potassium Dichromate Preparation Uses",
            "Potassium Permanganate Preparation Uses",
            "Lanthanides Electronic Configuration",
            "Lanthanoid Contraction", "Actinides Properties",
            "Inner Transition Elements Comparison",
        ],
        "Coordination Compounds": [
            "Ligands Classification", "Coordination Number",
            "IUPAC Nomenclature of Complexes",
            "Isomerism Structural",
            "Isomerism Stereochemical Geometrical",
            "Optical Isomerism in Complexes",
            "Valence Bond Theory of Complexes",
            "Inner and Outer Orbital Complexes",
            "Crystal Field Theory",
            "Crystal Field Splitting Octahedral",
            "Crystal Field Splitting Tetrahedral",
            "High Spin and Low Spin", "CFSE",
            "Colour and Spectrochemical Series",
            "Stability Constant of Complexes",
            "Chelates and Chelate Effect",
            "Applications Metallurgy Medicine Analytical",
        ],
        "Haloalkanes and Haloarenes": [
            "Classification of Halides",
            "Preparation of Haloalkanes",
            "Physical Properties",
            "SN1 Mechanism", "SN2 Mechanism",
            "Factors Affecting SN1 SN2",
            "E1 Elimination",
            "E2 Elimination Saytzeff Rule",
            "Nucleophilicity vs Basicity",
            "Leaving Group Ability",
            "Reactions of Haloalkanes",
            "Preparation of Haloarenes",
            "Reactions of Haloarenes Nucleophilic",
            "Electrophilic Substitution on Haloarenes",
            "Polyhalogen Compounds DDT BHC",
            "Freons and Ozone Depletion",
        ],
        "Alcohols Phenols and Ethers": [
            "Classification of Alcohols",
            "Preparation of Alcohols",
            "Physical Properties of Alcohols",
            "Chemical Reactions of Alcohols",
            "Dehydration of Alcohols",
            "Oxidation of Alcohols", "Esterification",
            "Lucas Test", "Acidity of Alcohols",
            "Preparation of Phenols",
            "Physical Properties of Phenols",
            "Acidity of Phenols",
            "Electrophilic Substitution on Phenol",
            "Reactions of Phenols",
            "Preparation of Ethers Williamson",
            "Reactions of Ethers", "Cleavage of Ethers",
        ],
        "Aldehydes Ketones and Carboxylic Acids": [
            "Preparation of Aldehydes and Ketones",
            "Physical Properties",
            "Nucleophilic Addition Mechanism",
            "Addition of HCN",
            "Addition of Grignard Reagent",
            "Aldol Condensation",
            "Cross Aldol Condensation",
            "Cannizzaro Reaction",
            "Clemmensen Reduction",
            "Wolff Kishner Reduction",
            "Tollens Test Silver Mirror",
            "Fehlings Solution Test",
            "Iodoform Reaction",
            "Preparation of Carboxylic Acids",
            "Physical Properties of Carboxylic Acids",
            "Acidity of Carboxylic Acids",
            "Reactions of Carboxylic Acids",
            "Hell Volhard Zelinsky Reaction",
            "Acid Derivatives Ester Amide Anhydride",
        ],
        "Amines": [
            "Classification of Amines",
            "IUPAC Nomenclature of Amines",
            "Preparation of Primary Amines",
            "Physical Properties of Amines",
            "Basicity of Amines", "Factors Affecting Basicity",
            "Chemical Reactions of Amines",
            "Acylation and Benzoylation",
            "Carbylamine Reaction",
            "Reaction with HNO2",
            "Diazonium Salts Preparation",
            "Coupling Reaction Azo Dyes",
            "Sandmeyer Reaction", "Gattermann Reaction",
            "Balz Schiemann Reaction",
        ],
        "Biomolecules": [
            "Carbohydrates Classification",
            "Monosaccharides Glucose Fructose",
            "Optical Activity in Glucose",
            "Open Chain and Cyclic Forms",
            "Disaccharides Sucrose Lactose Maltose",
            "Polysaccharides Starch Cellulose Glycogen",
            "Reducing and Non-Reducing Sugars",
            "Proteins and Amino Acids",
            "Peptide Bond and Polypeptides",
            "Structure of Proteins Primary Secondary Tertiary",
            "Denaturation of Proteins",
            "Enzymes and Enzyme Action",
            "Vitamins Fat and Water Soluble",
            "Nucleic Acids DNA RNA",
            "DNA Double Helix Watson Crick",
            "Hormones",
        ],
        "Polymers": [
            "Addition Polymerisation",
            "Condensation Polymerisation",
            "Natural Rubber and Vulcanisation",
            "Synthetic Rubber Neoprene Buna",
            "Nylon 6 and Nylon 66",
            "Polyester Terylene Dacron",
            "Bakelite Phenol Formaldehyde",
            "Polythene LDPE HDPE",
            "PVC and Teflon",
            "Biodegradable Polymers PHBV",
            "Copolymers",
        ],
        "Electrochemistry": [
            "Electrolytic Conduction", "Specific Conductance",
            "Molar Conductance", "Equivalent Conductance",
            "Variation with Dilution", "Kohlrausch Law",
            "Faradays First Law of Electrolysis",
            "Faradays Second Law of Electrolysis",
            "Electrochemical Cells", "Galvanic Cell Daniel Cell",
            "Salt Bridge", "Standard Electrode Potential",
            "Standard Hydrogen Electrode",
            "Electrochemical Series", "Nernst Equation",
            "EMF and Gibbs Energy", "Concentration Cell",
            "Batteries Primary Secondary",
            "Fuel Cells", "Corrosion Electrochemical Theory",
        ],
        "Chemical Kinetics": [
            "Rate of Reaction", "Average and Instantaneous Rate",
            "Factors Affecting Rate", "Rate Law and Rate Expression",
            "Rate Constant", "Order of Reaction", "Molecularity",
            "Zero Order Reactions", "First Order Reactions",
            "Second Order Reactions",
            "Half Life of Zero and First Order",
            "Integrated Rate Equations", "Arrhenius Equation",
            "Activation Energy", "Effect of Temperature on Rate",
            "Collision Theory", "Transition State Theory",
            "Catalysis Homogeneous", "Catalysis Heterogeneous",
            "Enzyme Catalysis",
        ],
        "Surface Chemistry": [
            "Adsorption Physisorption Chemisorption",
            "Adsorption Isotherms Freundlich",
            "Adsorption Isotherms Langmuir",
            "Catalysis and Mechanism",
            "Promoters and Poisons",
            "Preparation of Colloids",
            "Types of Colloids Sols Gels Emulsions",
            "Properties of Colloids", "Tyndall Effect",
            "Brownian Motion",
            "Electrophoresis and Electroosmosis",
            "Coagulation Hardy Schulze Rule",
            "Protection of Colloids", "Gold Number",
        ],
        "Solutions": [
            "Types of Solutions",
            "Concentration Molarity Molality Mole Fraction",
            "Normality and Equivalents",
            "Solubility and Factors", "Henrys Law",
            "Raoults Law for Volatile Solute",
            "Raoults Law for Non-Volatile Solute",
            "Ideal Solutions",
            "Non-Ideal Solutions Positive Deviation",
            "Non-Ideal Solutions Negative Deviation",
            "Azeotropes", "Colligative Properties",
            "Relative Lowering of Vapour Pressure",
            "Elevation in Boiling Point",
            "Depression in Freezing Point",
            "Osmosis and Osmotic Pressure",
            "van t Hoff Factor", "Abnormal Molar Mass",
        ],
        "Solid State": [
            "Classification of Solids",
            "Crystalline and Amorphous Solids",
            "Crystal Lattice and Unit Cell",
            "Primitive and Centred Unit Cells",
            "Simple Cubic BCC FCC Unit Cells",
            "Packing Efficiency",
            "Number of Atoms per Unit Cell",
            "Density Calculation of Unit Cell",
            "Tetrahedral and Octahedral Voids",
            "Radius Ratio and Coordination Number",
            "Close Packing HCP and CCP",
            "Point Defects Schottky Frenkel",
            "Impurity Defects", "Non-Stoichiometric Defects",
            "Electrical Properties Band Theory",
            "Magnetic Properties Dia Para Ferro",
        ],
        "General Principles of Metallurgy": [
            "Minerals and Ores", "Concentration of Ores",
            "Froth Flotation", "Magnetic Separation",
            "Leaching Hydrometallurgy",
            "Smelting and Roasting",
            "Reduction Methods Thermodynamic",
            "Ellingham Diagram", "Electrolytic Reduction",
            "Refining of Metals", "Vapour Phase Refining",
            "Extraction of Copper", "Extraction of Zinc",
            "Extraction of Iron Blast Furnace",
            "Extraction of Aluminium Hall Heroult",
        ],
        "Organic Chemistry Basic Principles": [
            "Hybridisation sp sp2 sp3",
            "IUPAC Nomenclature Organic",
            "Structural Isomerism",
            "Stereoisomerism",
            "Geometrical Isomerism E Z",
            "Optical Isomerism Chirality",
            "R S Configuration",
            "Inductive Effect I and -I",
            "Mesomeric Resonance Effect",
            "Hyperconjugation", "Electromeric Effect",
            "Electrophile and Nucleophile",
            "Carbocation Stability", "Carbanion Stability",
            "Free Radical",
            "Types of Organic Reactions Addition Substitution Elimination",
        ],
        "Hydrocarbons": [
            "IUPAC Nomenclature Alkanes",
            "Conformations of Ethane",
            "Free Radical Halogenation of Alkanes",
            "Cracking and Pyrolysis",
            "IUPAC Nomenclature Alkenes",
            "Electrophilic Addition to Alkenes",
            "Markovnikov Rule",
            "Anti Markovnikov Peroxide Effect",
            "Ozonolysis of Alkenes", "Oxidation of Alkenes",
            "IUPAC Nomenclature Alkynes",
            "Reactions of Alkynes",
            "Acidic Nature of Terminal Alkynes",
            "Benzene Structure",
            "Resonance and Aromaticity",
            "Electrophilic Aromatic Substitution",
            "Activating and Deactivating Groups",
            "Ortho Para and Meta Directors",
        ],
    },

    "Mathematics": {
        "Sets Relations and Functions": [
            "Representation of Sets",
            "Types of Sets Empty Finite Infinite",
            "Subsets and Power Set",
            "Set Operations Union Intersection Difference",
            "Complement of a Set", "Venn Diagrams",
            "Cartesian Product",
            "Types of Relations Reflexive Symmetric Transitive",
            "Equivalence Relations",
            "Types of Functions One-One Onto",
            "Bijective Functions", "Composition of Functions",
            "Inverse of a Function", "Even and Odd Functions",
            "Periodic Functions", "Graph Transformations",
        ],
        "Complex Numbers": [
            "Introduction and Definition",
            "Algebra Addition Subtraction Multiplication",
            "Division of Complex Numbers",
            "Conjugate and Modulus", "Argument and Amplitude",
            "Polar Form", "Euler Form",
            "De Moivres Theorem",
            "Rotation in Complex Plane",
            "Cube Roots of Unity", "nth Roots of Unity",
            "Geometry Circles and Lines in Complex Plane",
            "Locus Problems in Complex Plane",
            "Triangle and Polygon in Complex Plane",
        ],
        "Sequences and Series": [
            "Arithmetic Progression nth Term", "Sum of AP",
            "Geometric Progression nth Term",
            "Sum of GP Finite", "Sum of GP Infinite",
            "Harmonic Progression",
            "AM GM HM Inequalities",
            "Relation Between AM GM HM",
            "Arithmetico-Geometric Series",
            "Telescoping Series",
            "Sum of Natural Numbers Squares Cubes",
            "Method of Differences", "Vn Method",
        ],
        "Quadratic Equations": [
            "Roots of Quadratic", "Nature of Roots Discriminant",
            "Sum and Product of Roots Vietas",
            "Symmetric Functions of Roots",
            "Formation of Quadratic with Given Roots",
            "Common Roots",
            "Range of Quadratic Expression",
            "Location of Roots",
            "Roots in a Given Interval",
            "Sign of Quadratic Expression",
            "Equations Reducible to Quadratic",
            "Irrational Equations",
        ],
        "Permutations and Combinations": [
            "Fundamental Counting Principle",
            "Factorial Notation", "Permutations nPr",
            "Permutations with Repetition",
            "Circular Permutations", "Combinations nCr",
            "Properties of nCr",
            "Arrangements with Identical Objects",
            "Arrangements with Restrictions",
            "Selection with Restrictions",
            "Distribution of Identical Objects",
            "Distribution of Distinct Objects",
            "Derangements", "Number of Divisors",
        ],
        "Binomial Theorem": [
            "Binomial Expansion for Positive Integer n",
            "General Term Tr+1", "Middle Term of Binomial",
            "Binomial Coefficients Properties Pascal Triangle",
            "Sum of Binomial Coefficients",
            "Alternate Sum of Binomial Coefficients",
            "Term Independent of x",
            "Greatest Binomial Coefficient",
            "Numerically Greatest Term",
            "Multinomial Expansion",
            "Binomial Theorem for Rational Index",
            "Approximations using Binomial",
        ],
        "Limits and Derivatives": [
            "Concept of Limit",
            "Left Hand and Right Hand Limits",
            "Standard Limit Sinx over x",
            "Standard Exponential and Log Limits",
            "LHopitals Rule", "Sandwich Theorem",
            "Continuity and Limit",
            "First Principles Derivative Definition",
            "Derivatives of Standard Functions",
            "Sum Difference Product Quotient Rules",
            "Derivative of Polynomials",
            "Derivative of Trigonometric Functions",
            "Derivative of Exponential and Log",
        ],
        "Continuity and Differentiability": [
            "Continuity at a Point",
            "Continuity on Closed Interval",
            "Types of Discontinuity Removable Jump Infinite",
            "Differentiability at a Point",
            "Relation Between Differentiability and Continuity",
            "Chain Rule", "Implicit Differentiation",
            "Parametric Differentiation",
            "Differentiation of Inverse Trig Functions",
            "Logarithmic Differentiation",
            "Second and Higher Order Derivatives",
            "Rolles Theorem",
            "Lagranges Mean Value Theorem",
        ],
        "Application of Derivatives": [
            "Rate of Change", "Tangent to Curve Slope",
            "Normal to Curve", "Angle Between Curves",
            "Increasing and Decreasing Functions",
            "Monotonicity on Interval",
            "Maxima and Minima Local",
            "Maxima and Minima Global",
            "First Derivative Test", "Second Derivative Test",
            "Point of Inflection",
            "Approximations Using Derivatives",
            "Curve Sketching", "Optimization Problems",
        ],
        "Integrals": [
            "Standard Integration Formulas",
            "Integration by Substitution",
            "Integration by Parts ILATE",
            "Integration by Partial Fractions",
            "Integration of Rational Functions",
            "Integration of Trigonometric Functions",
            "Integration of sqrt Forms",
            "Integration of 1 over Linear Quadratic",
            "Definite Integral Definition",
            "Properties of Definite Integrals",
            "Definite Integral as Limit of Sum",
            "Walli Formula", "Reduction Formulae",
        ],
        "Application of Integrals": [
            "Area under Curve", "Area between Two Curves",
            "Area using Vertical Strips",
            "Area using Horizontal Strips",
            "Area of Bounded Region with Lines",
            "Area of Circle Parabola Ellipse",
            "Symmetry in Area Problems",
        ],
        "Differential Equations": [
            "Order and Degree",
            "Formation of Differential Equations",
            "Variable Separable Method",
            "Homogeneous Differential Equations",
            "Linear First Order DE Integrating Factor",
            "Bernoullis Equation",
            "Exact Differential Equations",
            "Applications of DE Growth Decay",
            "Applications of DE Newtons Cooling",
        ],
        "Vector Algebra": [
            "Types of Vectors", "Vector Addition and Subtraction",
            "Scalar Multiplication", "Position Vector",
            "Section Formula Internal External",
            "Dot Product Definition",
            "Dot Product Properties and Applications",
            "Cross Product Definition", "Cross Product Properties",
            "Area using Cross Product",
            "Scalar Triple Product",
            "Volume of Parallelepiped",
            "Vector Triple Product", "Coplanarity of Vectors",
        ],
        "3D Geometry": [
            "Direction Cosines", "Direction Ratios",
            "Angle Between Lines using DR",
            "Equation of Line Passing Through Point",
            "Symmetrical Form of Line",
            "Equation of Line Through Two Points",
            "Angle Between Two Lines in 3D",
            "Distance of Point from Line",
            "Skew Lines and Shortest Distance",
            "Equation of Plane Normal Form",
            "Equation of Plane Three Point Form",
            "Angle Between Two Planes",
            "Distance of Point from Plane",
            "Line Plane Intersection",
            "Angle Between Line and Plane",
            "Family of Planes", "Sphere Equation and Properties",
        ],
        "Probability": [
            "Classical Definition", "Axiomatic Definition",
            "Addition Theorem", "Mutually Exclusive Events",
            "Conditional Probability", "Multiplication Theorem",
            "Independent Events", "Bayes Theorem",
            "Theorem of Total Probability",
            "Random Variables Discrete",
            "Probability Distribution Table",
            "Mean and Variance of Distribution",
            "Binomial Distribution", "Poisson Distribution",
        ],
        "Statistics": [
            "Mean Arithmetic", "Median", "Mode",
            "Mean Deviation from Mean",
            "Mean Deviation from Median",
            "Variance", "Standard Deviation",
            "Coefficient of Variation",
            "Combined Mean and Variance",
            "Frequency Distribution",
        ],
        "Straight Lines": [
            "Slope and Inclination",
            "Slope Intercept Form", "Point Slope Form",
            "Two Point Form", "Intercept Form", "Normal Form",
            "General Form ax+by+c=0",
            "Distance of Point from Line",
            "Distance Between Parallel Lines",
            "Angle Between Two Lines",
            "Foot of Perpendicular", "Image of Point in Line",
            "Family of Lines", "Locus and its Equation",
            "Pair of Straight Lines Homogeneous",
            "Pair of Straight Lines General Second Degree",
            "Angle Between Pair of Lines",
        ],
        "Circles": [
            "Standard Equation of Circle",
            "General Equation of Circle",
            "Centre and Radius", "Equation from Diameter",
            "Position of Point wrt Circle",
            "Length of Tangent",
            "Tangent from External Point",
            "Equation of Tangent at Point",
            "Equation of Normal at Point",
            "Chord of Contact",
            "Chord with Given Midpoint",
            "Pair of Tangents SS1 T2",
            "Family of Circles", "Radical Axis",
            "Radical Centre", "Common Chord",
            "Intersection of Two Circles",
            "Orthogonal Circles",
        ],
        "Conic Sections": [
            "Parabola Standard Equations",
            "Focus Directrix Axis Vertex of Parabola",
            "Focal Chord of Parabola",
            "Tangent to Parabola", "Normal to Parabola",
            "Chord of Contact Parabola",
            "Ellipse Standard Equation",
            "Major Minor Axis Eccentricity",
            "Focal Distances Sum Property",
            "Tangent to Ellipse", "Normal to Ellipse",
            "Auxiliary Circle and Eccentric Angle",
            "Hyperbola Standard Equation",
            "Eccentricity Asymptotes of Hyperbola",
            "Tangent to Hyperbola", "Normal to Hyperbola",
            "Rectangular Hyperbola",
            "Chord of Contact for Conics",
        ],
        "Trigonometry": [
            "Trigonometric Ratios Definition",
            "Trigonometric Ratios of Standard Angles",
            "Trigonometric Identities Pythagorean",
            "Compound Angle Formulas",
            "Multiple Angle Formulas 2A 3A",
            "Sub-Multiple Angle Formulas Half Angle",
            "Product to Sum Formulas",
            "Sum to Product Formulas",
            "Conditional Identities",
            "Trigonometric Equations General Solution",
            "Principal Value and General Solution",
            "Sine Rule", "Cosine Rule",
            "Projection Formula", "Area of Triangle",
            "Half Angle in Triangle",
            "Circumradius Inradius",
            "Heights and Distances Elevation Depression",
        ],
        "Inverse Trigonometry": [
            "Domain Range of arcsin arccos arctan",
            "Principal Value Branch",
            "Graphs of Inverse Trig Functions",
            "Properties of arcsin and arccos",
            "Properties of arctan and arccot",
            "Identities Involving Inverse Trig",
            "Sum of arctan Formula",
            "Simplification of Inverse Trig Expressions",
            "Inverse Trig Equations",
            "Substitution in Inverse Trig",
        ],
        "Matrices and Determinants": [
            "Types of Matrices",
            "Matrix Addition and Subtraction",
            "Scalar Multiplication",
            "Matrix Multiplication", "Transpose Properties",
            "Symmetric and Skew-Symmetric",
            "Orthogonal Matrix",
            "Determinant of 2x2 and 3x3",
            "Properties of Determinants",
            "Cofactor and Minors",
            "Adjoint of Matrix", "Inverse of Matrix",
            "Singular Matrix",
            "System of Linear Equations Cramers Rule",
            "System by Matrix Inversion",
            "Consistency of System", "Rank of Matrix",
            "Cayley Hamilton Theorem",
        ],
        "Mathematical Reasoning": [
            "Statements and Propositions",
            "Logical Connectives And Or Not",
            "Conditional and Biconditional",
            "Truth Tables", "Tautology and Contradiction",
            "Converse Inverse Contrapositive",
            "Quantifiers Universal Existential",
            "Validity of Arguments",
        ],
        "Mathematical Induction": [
            "Principle of Mathematical Induction",
            "Proof by Induction Sum Formulas",
            "Divisibility by Induction",
            "Inequality Proofs by Induction",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # NEET BIOLOGY  (Class 11 + 12, Botany + Zoology)
    # ─────────────────────────────────────────────────────────────────────────
    "Biology": {
        # ── CLASS 11 ─────────────────────────────────────────────────────────
        "The Living World": [
            "What is Living", "Defining Life",
            "Biodiversity", "Nomenclature Binomial System",
            "Taxonomy and Systematics",
            "Need for Classification",
            "Three Domains of Life",
            "Taxonomic Categories Species Genus Family Order Class Phylum Kingdom",
            "Taxonomical Aids",
            "Herbarium", "Botanical Gardens",
            "Museum", "Zoological Parks",
            "Monograph and Flora", "Taxonomic Keys",
        ],
        "Biological Classification": [
            "Two Kingdom Classification", "Five Kingdom Classification",
            "Kingdom Monera Characteristics",
            "Archaebacteria", "Eubacteria",
            "Mycoplasma", "Cyanobacteria",
            "Kingdom Protista Characteristics",
            "Chrysophytes Diatoms Golden Algae",
            "Dinoflagellates", "Euglenoids",
            "Slime Moulds", "Protozoans",
            "Kingdom Fungi Characteristics",
            "Phycomycetes", "Ascomycetes",
            "Basidiomycetes", "Deuteromycetes",
            "Reproduction in Fungi",
            "Kingdom Plantae Overview",
            "Kingdom Animalia Overview",
            "Viruses Structure and Types",
            "Bacteriophage", "Plant Viruses Animal Viruses",
            "Viroids", "Prions",
            "Lichens Mutualistic Association",
        ],
        "Plant Kingdom": [
            "Algae General Characteristics",
            "Chlorophyceae Green Algae Characteristics and Examples",
            "Phaeophyceae Brown Algae Characteristics and Examples",
            "Rhodophyceae Red Algae Characteristics and Examples",
            "Economic Importance of Algae",
            "Bryophytes General Characteristics",
            "Liverworts Hepaticopsida",
            "Mosses Bryopsida",
            "Economic Importance of Bryophytes",
            "Pteridophytes General Characteristics",
            "Lycopsida Sphenopsida Pteropsida",
            "Heterospory and Seed Habit",
            "Economic Importance of Pteridophytes",
            "Gymnosperms General Characteristics",
            "Cycas Pinus Gnetum",
            "Economic Importance of Gymnosperms",
            "Angiosperms General Characteristics",
            "Dicots vs Monocots",
            "Alternation of Generations",
            "Haplontic Diplontic Diplohaplontic Life Cycles",
        ],
        "Animal Kingdom": [
            "Basis of Classification Symmetry Coelom Segmentation Notochord",
            "Symmetry Radial Bilateral Asymmetry",
            "Diploblastic and Triploblastic Organisation",
            "Coelom Acoelomate Pseudocoelomate Coelomate",
            "Segmentation Metamerism",
            "Notochord",
            "Phylum Porifera Characteristics and Examples",
            "Canal System in Porifera",
            "Phylum Coelenterata Cnidaria Characteristics and Examples",
            "Polymorphism in Coelenterata",
            "Phylum Ctenophora Characteristics and Examples",
            "Phylum Platyhelminthes Characteristics and Examples",
            "Phylum Aschelminthes Nematoda Characteristics and Examples",
            "Phylum Annelida Characteristics and Examples",
            "Phylum Arthropoda Characteristics and Examples",
            "Economic Importance of Arthropoda",
            "Phylum Mollusca Characteristics and Examples",
            "Phylum Echinodermata Characteristics and Examples",
            "Phylum Hemichordata Characteristics and Examples",
            "Phylum Chordata Characteristics",
            "Subphylum Urochordata Cephalochordata",
            "Class Cyclostomata",
            "Class Chondrichthyes Cartilaginous Fishes",
            "Class Osteichthyes Bony Fishes",
            "Class Amphibia Characteristics and Examples",
            "Class Reptilia Characteristics and Examples",
            "Class Aves Characteristics and Examples",
            "Class Mammalia Characteristics and Examples",
        ],
        "Morphology of Flowering Plants": [
            "Root Regions and Functions",
            "Types of Root Systems Taproot Fibrous Adventitious",
            "Modifications of Root Storage Respiratory Climbing",
            "Stem Characteristics and Functions",
            "Modifications of Stem Underground Aerial",
            "Leaf Parts Lamina Petiole Leaf Base",
            "Venation Reticulate Parallel",
            "Types of Leaves Simple Compound",
            "Phyllotaxy Alternate Opposite Whorled",
            "Inflorescence Racemose Cymose",
            "Parts of a Flower Calyx Corolla Androecium Gynoecium",
            "Aestivation Valvate Twisted Imbricate",
            "Placentation Marginal Axile Parietal Basal Free Central Superficial",
            "Fruit True False Parthenocarpic",
            "Types of Fruits Simple Aggregate Multiple",
            "Seed Structure Dicot Monocot",
            "Semi-technical Description of a Flowering Plant",
            "Family Fabaceae Papilionaceae Characteristics",
            "Family Solanaceae Characteristics",
            "Family Liliaceae Characteristics",
        ],
        "Anatomy of Flowering Plants": [
            "Meristematic Tissue Apical Lateral Intercalary",
            "Permanent Tissue Simple Compound",
            "Parenchyma Characteristics and Types",
            "Collenchyma Characteristics",
            "Sclerenchyma Fibres Sclereids",
            "Xylem Tracheids Vessels Xylem Fibres Xylem Parenchyma",
            "Phloem Sieve Tubes Companion Cells Phloem Parenchyma Phloem Fibres",
            "Epidermal Tissue System",
            "Ground Tissue System",
            "Vascular Tissue System",
            "Anatomy of Dicotyledonous Root",
            "Anatomy of Monocotyledonous Root",
            "Anatomy of Dicotyledonous Stem",
            "Anatomy of Monocotyledonous Stem",
            "Anatomy of Dorsiventral Dicot Leaf",
            "Anatomy of Isobilateral Monocot Leaf",
            "Secondary Growth in Dicot Stem",
            "Secondary Growth in Dicot Root",
            "Vascular Cambium and Cork Cambium",
            "Annual Rings",
        ],
        "Structural Organisation in Animals": [
            "Epithelial Tissue Types Squamous Cuboidal Columnar Ciliated",
            "Glandular Epithelium",
            "Connective Tissue Loose Dense Specialised",
            "Cartilage Types Hyaline Elastic Fibrocartilage",
            "Bone Structure",
            "Blood as Connective Tissue",
            "Muscular Tissue Skeletal Smooth Cardiac",
            "Neural Tissue Neuron Neuroglia",
            "Organ and Organ System Concept",
            "Earthworm Morphology External Features",
            "Earthworm Anatomy Digestive Circulatory Nervous Reproductive",
            "Cockroach Morphology External Features",
            "Cockroach Anatomy Digestive Circulatory Nervous Reproductive",
            "Frog Morphology External Features",
            "Frog Anatomy Digestive Respiratory Circulatory Nervous Reproductive",
        ],
        "Cell: The Unit of Life": [
            "Cell Theory and its Modifications",
            "Overview of Cell Prokaryotic vs Eukaryotic",
            "Prokaryotic Cell Structure",
            "Cell Envelope Cell Wall Plasma Membrane Glycocalyx",
            "Eukaryotic Cell Overview",
            "Cell Membrane Structure Fluid Mosaic Model",
            "Transport across Membrane Passive Active",
            "Cell Wall Composition Function",
            "Endoplasmic Reticulum Rough Smooth Functions",
            "Golgi Apparatus Structure and Functions",
            "Lysosomes Structure and Functions",
            "Vacuoles Types and Functions",
            "Mitochondria Structure and Functions",
            "Plastids Chloroplast Chromoplast Leucoplast",
            "Chloroplast Structure Thylakoid Stroma",
            "Ribosomes 70S 80S Subunits",
            "Cytoskeleton Microfilaments Microtubules Intermediate Filaments",
            "Cilia and Flagella Structure",
            "Centrosome and Centriole",
            "Nucleus Structure Nuclear Envelope Nucleolus Chromatin",
            "Chromosomes Structure Types",
            "Microbodies Peroxisomes Glyoxysomes",
        ],
        "Biomolecules": [
            "Chemical Constituents of Living Cells",
            "Amino Acids Structure Classification",
            "Peptide Bond Formation",
            "Proteins Primary Structure",
            "Proteins Secondary Structure Alpha Helix Beta Sheet",
            "Proteins Tertiary and Quaternary Structure",
            "Denaturation of Proteins",
            "Nature of Bond Linking Subunits in Proteins",
            "Carbohydrates Monosaccharides Disaccharides Polysaccharides",
            "Starch Cellulose Glycogen Chitin",
            "Lipids Structure Saturated Unsaturated",
            "Phospholipids Waxes Sterols",
            "Nucleotides and Nucleic Acids",
            "DNA Structure Double Helix",
            "RNA Types mRNA tRNA rRNA",
            "Enzymes Classification Nomenclature",
            "Enzyme as Protein Cofactors Coenzymes Prosthetic Groups",
            "How Enzymes Work Lock and Key Induced Fit",
            "Factors Affecting Enzyme Activity",
            "Enzyme Kinetics Km Vmax Michaelis Menten",
            "Inhibition Competitive Non-competitive",
            "Metabolic Pathways Anabolic Catabolic",
            "Living State Dynamic Steady State",
        ],
        "Cell Cycle and Cell Division": [
            "Cell Cycle Overview G1 S G2 M Phase",
            "Interphase G1 S G2",
            "M Phase Overview",
            "Mitosis Prophase Chromatin Condensation",
            "Mitosis Metaphase Alignment at Equator",
            "Mitosis Anaphase Separation of Chromatids",
            "Mitosis Telophase Cytokinesis",
            "Significance of Mitosis",
            "Meiosis Overview Meiosis I Meiosis II",
            "Meiosis I Prophase I Leptotene Zygotene Pachytene Diplotene Diakinesis",
            "Crossing Over Recombination",
            "Meiosis I Metaphase I Anaphase I Telophase I",
            "Meiosis II Prophase II Metaphase II Anaphase II Telophase II",
            "Significance of Meiosis Genetic Variation",
            "Cell Cycle Checkpoints",
        ],
        "Transport in Plants": [
            "Means of Transport Diffusion Facilitated Diffusion Active Transport",
            "Osmosis Water Potential Solute Potential Pressure Potential",
            "Plasmolysis and Deplasmolysis",
            "Imbibition",
            "Apoplast Pathway",
            "Symplast Pathway",
            "Long Distance Transport of Water",
            "Cohesion Tension Theory of Water Ascent",
            "Root Pressure Guttation",
            "Transpiration Types Stomatal Cuticular Lenticular",
            "Transpiration Pull",
            "Factors Affecting Transpiration",
            "Stomatal Movement Guard Cells Mechanism",
            "Antitranspirants",
            "Uptake and Transport of Mineral Nutrients",
            "Translocation of Organic Solutes Phloem",
            "Mass Flow Hypothesis Munch Hypothesis",
            "Evidence for Phloem Transport",
        ],
        "Mineral Nutrition": [
            "Essential Mineral Elements Criteria for Essentiality",
            "Macronutrients Nitrogen Phosphorus Potassium Calcium Magnesium Sulphur",
            "Micronutrients Iron Manganese Zinc Copper Molybdenum Boron Chlorine Nickel",
            "Role of Macronutrients",
            "Role of Micronutrients",
            "Deficiency Symptoms of Nitrogen",
            "Deficiency Symptoms of Phosphorus Potassium Calcium Magnesium Sulphur",
            "Deficiency Symptoms of Micronutrients",
            "Toxicity of Micronutrients",
            "Mechanism of Absorption of Elements",
            "Translocation of Solutes",
            "Soil as Reservoir of Essential Elements",
            "Nitrogen Cycle Overview",
            "Ammonification Nitrification Denitrification",
            "Biological Nitrogen Fixation Symbiotic Free Living",
            "Rhizobium and Root Nodules",
            "Nitrate Assimilation",
        ],
        "Photosynthesis in Higher Plants": [
            "Early Experiments Priestley Ingenhousz Sachs Engelmann",
            "Site of Photosynthesis Chloroplast",
            "Photosynthetic Pigments Chlorophyll a b Carotenoids Xanthophylls",
            "Absorption Spectrum and Action Spectrum",
            "Light Reactions Overview",
            "Photosystems I and II",
            "Electron Transport Chain Z Scheme",
            "Cyclic Photophosphorylation",
            "Non-cyclic Photophosphorylation",
            "Chemiosmosis and ATP Synthesis",
            "Photolysis of Water Oxygen Evolution",
            "Calvin Cycle C3 Pathway Dark Reactions",
            "Rubisco and Carbon Fixation",
            "Regeneration of RuBP",
            "C4 Pathway Hatch Slack Pathway",
            "Bundle Sheath Cells Kranz Anatomy",
            "Photorespiration",
            "CAM Plants",
            "Factors Affecting Photosynthesis Light CO2 Temperature Water",
        ],
        "Respiration in Plants": [
            "Aerobic and Anaerobic Respiration",
            "Glycolysis EMP Pathway",
            "Fermentation Alcoholic Lactic Acid",
            "Fate of Pyruvate",
            "Aerobic Respiration Pyruvate Oxidation Acetyl CoA",
            "Krebs Cycle TCA Cycle Steps",
            "Products of Krebs Cycle NADH FADH2 CO2",
            "Electron Transport Chain",
            "Oxidative Phosphorylation Chemiosmosis",
            "ATP Yield from Glucose Aerobic",
            "Amphibolic Pathway",
            "Respiratory Quotient RQ",
            "Energy Relationships",
        ],
        "Plant Growth and Development": [
            "Growth Characteristics Phases",
            "Arithmetic and Geometric Growth",
            "Growth Rate Absolute Relative",
            "Conditions for Growth",
            "Differentiation Dedifferentiation Redifferentiation",
            "Plant Growth Regulators Overview",
            "Auxins Discovery Went Experiment",
            "Physiological Effects of Auxins",
            "Gibberellins Discovery",
            "Physiological Effects of Gibberellins",
            "Cytokinins Discovery",
            "Physiological Effects of Cytokinins",
            "Ethylene Discovery and Physiological Effects",
            "Abscisic Acid Stress Hormone",
            "Photoperiodism Short Day Long Day Day Neutral Plants",
            "Phytochrome",
            "Vernalisation",
            "Seed Dormancy and Germination",
        ],
        # ── CLASS 11 ZOOLOGY (Human Physiology) ──────────────────────────────
        "Digestion and Absorption": [
            "Alimentary Canal Structure Mouth Oesophagus Stomach",
            "Small Intestine Duodenum Jejunum Ileum",
            "Large Intestine Caecum Colon Rectum",
            "Wall of Alimentary Canal Layers",
            "Accessory Glands Salivary Glands Liver Pancreas",
            "Digestion in Buccal Cavity Salivary Amylase",
            "Swallowing and Peristalsis",
            "Digestion in Stomach Gastric Juice Pepsin HCl",
            "Digestion in Small Intestine Bile Pancreatic Juice",
            "Intestinal Juice Enzymes Erepsin Maltase Lactase Sucrase",
            "Absorption in Small Intestine Mechanisms",
            "Absorption of Carbohydrates Proteins Fats",
            "Absorption of Water and Minerals",
            "Assimilation",
            "Egestion",
            "Caloric Value of Foods",
            "Disorders Jaundice Vomiting Diarrhoea Constipation Indigestion PEM",
        ],
        "Breathing and Exchange of Gases": [
            "Respiratory Organs in Different Animals",
            "Human Respiratory System Nostrils Nasal Passage Pharynx Larynx",
            "Trachea Bronchi Bronchioles Alveoli",
            "Lungs Structure and Position",
            "Mechanism of Breathing Inspiration Expiration",
            "Respiratory Muscles Diaphragm Intercostals",
            "Respiratory Volumes and Capacities TV IRV ERV RV TLC VC FRC",
            "Partial Pressure of Gases Daltons Law",
            "Exchange of Gases at Alveoli",
            "Exchange of Gases at Tissues",
            "Transport of Oxygen Oxyhaemoglobin",
            "Oxygen Dissociation Curve Bohr Effect",
            "Transport of Carbon Dioxide Bicarbonate Carbamino",
            "Regulation of Respiration Neural Rhythm Centre",
            "Disorders Asthma Emphysema Occupational Respiratory Disease",
        ],
        "Body Fluids and Circulation": [
            "Blood Composition",
            "Blood Plasma Composition Functions",
            "Red Blood Cells Erythrocytes Structure Function",
            "White Blood Cells Leucocytes Types Functions",
            "Platelets Thrombocytes",
            "Blood Groups ABO System",
            "Blood Groups Rh System",
            "Coagulation of Blood Clotting Mechanism",
            "Lymph Composition and Function",
            "Human Circulatory System Heart Chambers Valves",
            "Blood Vessels Arteries Veins Capillaries",
            "Cardiac Cycle Systole Diastole",
            "Cardiac Output Stroke Volume Heart Rate",
            "ECG Electrocardiogram P QRS T Waves",
            "Double Circulation Pulmonary Systemic",
            "Regulation of Cardiac Activity Neural Hormonal",
            "Disorders Hypertension Coronary Artery Disease Angina Heart Failure",
        ],
        "Excretory Products and their Elimination": [
            "Modes of Excretion Ammonotelism Ureotelism Uricotelism",
            "Excretory Organs in Different Animals",
            "Human Excretory System Kidneys Ureters Bladder Urethra",
            "Kidney Structure Cortex Medulla Pelvis",
            "Nephron Structure Malpighian Body Tubule",
            "Glomerular Filtration Ultrafiltration",
            "Tubular Reabsorption Selective",
            "Tubular Secretion",
            "Urine Formation and Composition",
            "Mechanism of Concentration of Filtrate Counter Current",
            "Regulation of Kidney Function ADH Aldosterone ANF",
            "Juxtaglomerular Apparatus Renin Angiotensin",
            "Micturition Reflex",
            "Role of Liver in Excretion",
            "Role of Lungs Skin in Excretion",
            "Disorders Uraemia Renal Failure Renal Calculi Nephritis",
            "Dialysis Kidney Transplant",
        ],
        "Locomotion and Movement": [
            "Types of Movement Amoeboid Ciliary Muscular",
            "Skeletal Muscle Structure Gross and Microscopic",
            "Sarcomere A Band I Band H Zone Z Line",
            "Mechanism of Muscle Contraction Sliding Filament Theory",
            "Role of Actin Myosin ATP Ca2+",
            "Neuromuscular Junction",
            "Red Slow Fibres White Fast Fibres",
            "Skeletal System Axial Skeleton Skull Vertebral Column Ribs Sternum",
            "Appendicular Skeleton Pectoral Pelvic Girdle Limb Bones",
            "Joints Fibrous Cartilaginous Synovial",
            "Types of Synovial Joints Ball Socket Hinge Pivot Gliding",
            "Disorders Myasthenia Gravis Muscular Dystrophy Tetany Gout Arthritis Osteoporosis",
        ],
        "Neural Control and Coordination": [
            "Neuron Structure Cyton Dendrites Axon",
            "Types of Neurons Sensory Motor Interneuron",
            "Myelinated and Non-myelinated Nerve Fibres",
            "Nerve Impulse Generation Resting Membrane Potential",
            "Action Potential Depolarisation Repolarisation",
            "Conduction of Nerve Impulse Saltatory",
            "Synapse Structure",
            "Synaptic Transmission Chemical Neurotransmitters",
            "Inhibitory and Excitatory Synapses",
            "Central Nervous System Brain Spinal Cord",
            "Brain Structure Forebrain Midbrain Hindbrain",
            "Cerebrum Functions Lobes",
            "Cerebellum Medulla Oblongata Pons Functions",
            "Limbic System Hypothalamus",
            "Spinal Cord Structure Functions Reflex Arc",
            "Peripheral Nervous System Cranial Spinal Nerves",
            "Autonomic Nervous System Sympathetic Parasympathetic",
            "Sense Organs Eye Structure Cornea Lens Retina",
            "Photoreceptors Rods Cones",
            "Mechanism of Vision Phototransduction",
            "Defects of Vision Myopia Hypermetropia Astigmatism",
            "Ear Structure External Middle Inner",
            "Mechanism of Hearing",
            "Organ of Corti",
            "Mechanism of Equilibrium Utricle Saccule Semicircular Canals",
        ],
        "Chemical Coordination and Integration": [
            "Endocrine Glands and Hormones Overview",
            "Hypothalamus Releasing Inhibiting Hormones",
            "Pituitary Gland Structure Anterior Posterior",
            "Anterior Pituitary Hormones GH TSH ACTH LH FSH Prolactin",
            "Posterior Pituitary ADH Vasopressin Oxytocin",
            "Thyroid Gland Hormones T3 T4 Calcitonin",
            "Disorders of Thyroid Goitre Cretinism Myxoedema Exophthalmic Goitre",
            "Parathyroid Gland PTH Calcium Regulation",
            "Adrenal Gland Cortex Zona Glomerulosa Fasciculata Reticularis",
            "Adrenal Cortex Hormones Glucocorticoids Mineralocorticoids Sex Corticoids",
            "Adrenal Medulla Adrenaline Noradrenaline Fight or Flight",
            "Pancreas Islets of Langerhans Alpha Beta Delta Cells",
            "Insulin and Glucagon Actions",
            "Diabetes Mellitus Types",
            "Gonads Testes Testosterone",
            "Gonads Ovaries Oestrogen Progesterone",
            "Thymus Thymosins",
            "Pineal Gland Melatonin",
            "Heart as Endocrine Organ ANF",
            "Kidney as Endocrine Organ Erythropoietin",
            "Gastric and Intestinal Mucosa Hormones Gastrin Secretin CCK GIP",
            "Mechanism of Hormone Action Second Messenger",
            "Steroid Hormone Mechanism Gene Activation",
        ],
        # ── CLASS 12 ─────────────────────────────────────────────────────────
        "Reproduction in Organisms": [
            "Reproduction Definition and Significance",
            "Asexual Reproduction Binary Fission",
            "Asexual Reproduction Budding",
            "Asexual Reproduction Fragmentation",
            "Asexual Reproduction Regeneration",
            "Vegetative Propagation Natural",
            "Vegetative Propagation Artificial Cutting Grafting Layering",
            "Sexual Reproduction Overview",
            "Events in Sexual Reproduction Pre-fertilisation",
            "Gametogenesis",
            "Events in Sexual Reproduction Fertilisation",
            "Events in Sexual Reproduction Post-fertilisation",
            "Significance of Sexual vs Asexual Reproduction",
            "Juvenile Vegetative Phase Adult Reproductive Phase",
            "Monoecious and Dioecious Organisms",
        ],
        "Sexual Reproduction in Flowering Plants": [
            "Flower as a Site of Sexual Reproduction",
            "Stamen Structure Filament Anther",
            "Microsporogenesis Development of Pollen",
            "Microspore Mother Cell Meiosis",
            "Structure of Pollen Grain Exine Intine Germ Pores",
            "Pollen Viability",
            "Pistil Structure Stigma Style Ovary",
            "Anatropous Ovule Structure Integuments Nucellus Chalaza Micropyle",
            "Megasporogenesis",
            "Female Gametophyte Embryo Sac Development Monosporic",
            "Mature Embryo Sac 7 Celled 8 Nucleate",
            "Pollination Definition",
            "Autogamy Cleistogamy Chasmogamy",
            "Geitonogamy Cross Pollination Xenogamy",
            "Anemophily Wind Pollination",
            "Entomophily Insect Pollination",
            "Other Agents Water Animal",
            "Outbreeding Devices Self Incompatibility Heterostyly",
            "Pollen-Pistil Interaction Recognition",
            "Artificial Hybridisation Emasculation Bagging",
            "Double Fertilisation Syngamy Triple Fusion",
            "Post-fertilisation Endosperm Development",
            "Embryo Development Dicot Monocot",
            "Seed Structure and Dormancy",
            "Fruit Development True and False",
            "Apomixis Parthenocarpy",
            "Polyembryony",
        ],
        "Human Reproduction": [
            "Male Reproductive System Testes Epididymis Vas Deferens",
            "Male Accessory Glands Seminal Vesicles Prostate Bulbourethral",
            "Spermatogenesis Spermatogonia Primary Secondary Spermatocyte Spermatid Sperm",
            "Structure of Sperm Head Midpiece Tail",
            "Hormonal Control of Spermatogenesis FSH LH Testosterone",
            "Female Reproductive System Ovaries Fallopian Tubes Uterus Vagina",
            "Oogenesis Oogonia Primary Secondary Oocyte Ovum",
            "Ovarian Follicle Development Primordial Primary Secondary Graafian",
            "Corpus Luteum Formation",
            "Menstrual Cycle Phases Menstrual Follicular Ovulatory Luteal",
            "Hormonal Regulation of Menstrual Cycle FSH LH Oestrogen Progesterone",
            "Fertilisation Site Capacitation",
            "Cleavage Morula Blastocyst",
            "Implantation",
            "Placenta Formation and Functions",
            "Embryonic Development Gastrulation Organogenesis",
            "Foetal Development Trimester",
            "Parturition Labour Mechanism",
            "Lactation Colostrum",
        ],
        "Reproductive Health": [
            "Reproductive Health Definition",
            "Sexually Transmitted Infections Gonorrhoea Syphilis Genital Herpes AIDS Hepatitis",
            "Population Explosion Growth Rate",
            "Birth Control Need and Importance",
            "Natural Methods Periodic Abstinence Coitus Interruptus Lactational Amenorrhoea",
            "Barrier Methods Condoms Diaphragm Cervical Cap Vault",
            "Intrauterine Devices Copper T Hormone Releasing",
            "Oral Contraceptive Pills Mechanism",
            "Injectables and Implants",
            "Surgical Methods Vasectomy Tubectomy",
            "Emergency Contraception",
            "Medical Termination of Pregnancy MTP",
            "Infertility Causes Male Female",
            "Assisted Reproductive Technologies IVF ET",
            "ZIFT GIFT ICSI",
            "Amniocentesis",
        ],
        "Principles of Inheritance and Variation": [
            "Mendels Work Experimental Material Garden Pea",
            "Monohybrid Cross Law of Segregation",
            "Test Cross and Back Cross",
            "Dihybrid Cross Law of Independent Assortment",
            "Phenotype Genotype Homozygous Heterozygous",
            "Incomplete Dominance",
            "Codominance ABO Blood Group",
            "Multiple Alleles",
            "Pleiotropy",
            "Polygenic Inheritance Skin Colour Height",
            "Chromosomal Theory of Inheritance Sutton Boveri",
            "Linkage Morgan Drosophila",
            "Recombination Crossing Over",
            "Sex Determination XX XY in Humans",
            "Sex Determination ZW ZZ in Birds",
            "Sex Determination XO in Grasshopper",
            "Sex-linked Inheritance Haemophilia",
            "Sex-linked Inheritance Colour Blindness",
            "Pedigree Analysis",
            "Mutation Gene Mutation",
            "Mutation Chromosomal Aberrations Deletion Duplication Inversion Translocation",
            "Aneuploidy Polyploidy",
            "Genetic Disorders Chromosomal Down Syndrome Turner Klinefelter",
            "Genetic Disorders Mendelian Phenylketonuria Sickle Cell Anaemia Thalassaemia",
        ],
        "Molecular Basis of Inheritance": [
            "DNA as Genetic Material Griffiths Transformation Experiment",
            "Avery McCarty Macleod Experiment",
            "Hershey Chase Experiment",
            "RNA as Genetic Material Tobacco Mosaic Virus",
            "DNA Structure Watson Crick Double Helix",
            "DNA Base Pairing Chargaffs Rule",
            "B Form of DNA",
            "DNA Packaging Nucleosome Chromatin",
            "Histone Proteins",
            "Euchromatin and Heterochromatin",
            "DNA Replication Semi-conservative",
            "Meselson Stahl Experiment",
            "Enzymes in Replication Helicase Primase DNA Polymerase Ligase",
            "Transcription Unit Template Strand Coding Strand",
            "Transcription in Prokaryotes RNA Polymerase",
            "Transcription in Eukaryotes RNA Polymerase I II III",
            "Post-transcriptional Processing Capping Tailing Splicing",
            "Introns and Exons",
            "Genetic Code Properties Triplet Degenerate Universal Non-overlapping",
            "Codons and Anticodons",
            "Start Codon AUG Stop Codons UAA UAG UGA",
            "Translation Ribosomes tRNA Aminoacyl tRNA Synthetase",
            "Steps of Translation Initiation Elongation Termination",
            "Regulation of Gene Expression Concept",
            "Lac Operon Inducible System",
            "Human Genome Project Goals Methodologies",
            "DNA Fingerprinting VNTR",
        ],
        "Evolution": [
            "Origin of Life Chemical Evolution Oparin Haldane",
            "Miller Urey Experiment",
            "Theory of Spontaneous Generation",
            "Theory of Biogenesis Pasteur",
            "Early Earth Conditions",
            "Theories of Organic Evolution Lamarckism",
            "Darwinism Natural Selection",
            "Neo Darwinism Modern Synthesis",
            "Evidence of Evolution Morphological Homologous Analogous Vestigial Organs",
            "Evidence from Palaeontology Fossils",
            "Evidence from Embryology Biogenetic Law",
            "Evidence from Biogeography",
            "Evidence from Molecular Biology",
            "Mechanism of Evolution Variation Mutations Recombination",
            "Hardy Weinberg Principle Equilibrium",
            "Disturbances to Hardy Weinberg Gene Flow Genetic Drift",
            "Natural Selection Types Stabilising Directional Disruptive",
            "Adaptive Radiation",
            "Convergent Evolution",
            "Speciation Allopatric Sympatric",
            "Reproductive Isolation Mechanisms",
            "Human Evolution Dryopithecus Ramapithecus Australopithecus Homo habilis erectus sapiens",
        ],
        "Human Health and Disease": [
            "Common Diseases Bacterial Typhoid Pneumonia",
            "Common Diseases Viral Common Cold Dengue Malaria",
            "Amoebiasis Ascariasis Elephantiasis Ringworm",
            "Life Cycle of Plasmodium Malaria Parasite",
            "Immunity Innate Non-specific",
            "Immunity Acquired Specific Active Passive",
            "Humoral Immunity B Cells Antibodies",
            "Cell-mediated Immunity T Cells",
            "Lymphoid Organs Primary Secondary",
            "Antibody Structure IgG IgM IgA IgE IgD",
            "Antigen-Antibody Interaction",
            "Vaccines and Vaccination",
            "Allergies Hypersensitivity Mast Cells IgE",
            "Autoimmunity",
            "AIDS HIV Life Cycle Transmission Prevention",
            "Cancer Types Benign Malignant Metastasis",
            "Carcinogens Chemical Physical Biological",
            "Detection and Treatment of Cancer",
            "Drugs Opioids Cannabinoids Cocaine Tobacco",
            "Alcohol Effects",
            "Addiction and Dependence",
        ],
        "Strategies for Enhancement in Food Production": [
            "Animal Husbandry Dairy Farm Management",
            "Poultry Farm Management",
            "Fisheries Freshwater Marine Aquaculture",
            "Bee Keeping Apiculture",
            "Plant Breeding Overview",
            "Steps in Plant Breeding",
            "Plant Breeding Hybridisation",
            "Mutation Breeding Polyploidy",
            "Plant Breeding for Disease Resistance",
            "Plant Breeding for Pest Resistance",
            "Biofortification Golden Rice Iron Wheat",
            "Single Cell Protein Spirulina",
            "Tissue Culture Micropropagation",
            "Somatic Hybridisation Protoplast Fusion",
        ],
        "Microbes in Human Welfare": [
            "Microbes in Household Products Curd Bread Dosa Idli",
            "Microbes in Industrial Products",
            "Antibiotics Penicillin Streptomycin",
            "Chemicals Organic Acids Ethanol Citric Acid Acetic Acid",
            "Enzymes Lipases Proteases Streptokinase",
            "Bioactive Molecules Cyclosporin A Statins",
            "Microbes in Sewage Treatment Primary Secondary",
            "BOD Biochemical Oxygen Demand",
            "Microbes in Biogas Production Methanogens",
            "Biogas Plants",
            "Microbes as Biofertilisers Rhizobium Azospirillum Azotobacter",
            "Mycorrhiza as Biofertiliser",
            "Biocontrol Agents Bacillus thuringiensis",
            "Baculovirus as Biocontrol Agent",
            "Integrated Pest Management",
        ],
        "Biotechnology Principles and Processes": [
            "Principles of Biotechnology Genetic Engineering",
            "Principle of Bioprocess Engineering",
            "Restriction Enzymes Types I II III",
            "Action of Restriction Enzymes Palindromic Sequences",
            "Sticky Ends Blunt Ends",
            "Cloning Vectors Plasmid pBR322",
            "Bacteriophage Lambda as Vector",
            "Cosmids BAC YAC Vectors",
            "Expression Vectors",
            "Competent Host Transformation Methods",
            "Heat Shock CaCl2 Method",
            "Electroporation Microinjection Biolistics",
            "Selectable Markers Antibiotic Resistance",
            "Insertional Inactivation",
            "PCR Polymerase Chain Reaction Principle",
            "Steps of PCR Denaturation Annealing Extension",
            "Applications of PCR",
            "Gel Electrophoresis Agarose",
            "Elution of DNA Bands",
            "Expression of Recombinant Proteins",
            "Bioreactors Types",
            "Downstream Processing",
        ],
        "Biotechnology and its Applications": [
            "Bt Crops Bacillus thuringiensis Cry Proteins",
            "Bt Cotton Bt Brinjal",
            "RNA Interference RNAi Silencing",
            "Nematode Resistance in Tobacco",
            "Transgenic Animals Mice Cattle Fish",
            "Transgenic Animals for Drug Production",
            "Molecular Diagnosis ELISA",
            "Molecular Diagnosis PCR for Pathogens",
            "Recombinant Insulin Eli Lilly",
            "Gene Therapy Somatic Germline",
            "Recombinant Vaccines Hepatitis B",
            "Genetically Modified Organisms Benefits",
            "Concerns about GMOs Safety Ethical",
            "Bioethics",
            "Patents and Intellectual Property",
            "Biopiracy",
        ],
        "Organisms and Populations": [
            "Organism and its Environment",
            "Abiotic Factors Temperature",
            "Abiotic Factors Water",
            "Abiotic Factors Light Photoperiod",
            "Abiotic Factors Soil Edaphic",
            "Responses to Abiotic Factors Regulate Conform Migrate Suspend",
            "Homeostasis",
            "Adaptations Desert Animals Plants",
            "Adaptations Aquatic Organisms",
            "Population Attributes Birth Rate Death Rate",
            "Age Distribution Pyramid",
            "Population Density Measurement",
            "Population Growth Exponential J Curve",
            "Population Growth Logistic S Curve",
            "Carrying Capacity K",
            "Life History Variation r K Strategies",
            "Population Interactions Overview",
            "Mutualism Examples",
            "Competition Interspecific Intraspecific",
            "Gauses Competitive Exclusion Principle",
            "Predation Prey Predator Relationship",
            "Co-evolution",
            "Parasitism Ectoparasite Endoparasite",
            "Commensalism Examples",
            "Amensalism",
        ],
        "Ecosystem": [
            "Ecosystem Structure and Function",
            "Biotic and Abiotic Components",
            "Productivity Primary Gross Net",
            "Secondary Productivity",
            "Decomposition Steps Fragmentation Leaching Catabolism Humification Mineralisation",
            "Factors Affecting Decomposition",
            "Energy Flow Concept",
            "Trophic Levels Producers Consumers Decomposers",
            "Food Chains Grazing Detritus",
            "Food Webs",
            "Ecological Pyramids Pyramid of Numbers Biomass Energy",
            "Lindeman Efficiency Ten Percent Law",
            "Ecological Succession Primary Secondary",
            "Hydrarch and Xerarch Succession",
            "Climax Community",
            "Nutrient Cycling Biogeochemical Cycles",
            "Carbon Cycle",
            "Phosphorus Cycle",
            "Ecosystem Services",
        ],
        "Biodiversity and Conservation": [
            "Levels of Biodiversity Genetic Species Ecosystem",
            "Global Biodiversity Distribution",
            "Patterns of Biodiversity Latitudinal Gradient",
            "Species Area Relationship",
            "Importance of Biodiversity Narrowly Utilitarian Broadly Utilitarian Ethical",
            "Loss of Biodiversity HIPPO Framework",
            "Habitat Loss Fragmentation",
            "Overexploitation Invasive Species Pollution",
            "Extinction Co-extinction",
            "Biodiversity Conservation In-situ",
            "Protected Areas National Parks Wildlife Sanctuaries Biosphere Reserves",
            "Hotspots of Biodiversity",
            "Sacred Groves",
            "Biodiversity Conservation Ex-situ",
            "Zoological Parks Botanical Gardens Seed Banks Gene Banks",
            "Cryopreservation",
            "Convention on Biological Diversity CBD",
            "Earth Summit Rio de Janeiro",
            "IUCN Red List Categories",
        ],
        "Environmental Issues": [
            "Air Pollution Causes Primary Secondary Pollutants",
            "Effects of Air Pollution on Health Plants Materials",
            "Smog Photochemical Smog",
            "Control of Air Pollution Scrubbers Electrostatic Precipitators",
            "Water Pollution Sources Domestic Industrial Agricultural",
            "Effects of Water Pollution BOD Eutrophication",
            "Biomagnification Bioaccumulation",
            "Control of Water Pollution",
            "Solid Waste Municipal Hazardous Biomedical Electronic",
            "Agro Chemicals Pesticides DDT",
            "Radioactive Waste Disposal",
            "Greenhouse Effect CO2 Methane CFC N2O",
            "Global Warming Consequences",
            "Kyoto Protocol",
            "Ozone Layer Depletion CFCs UV Radiation",
            "Ozone Hole Antarctica",
            "Deforestation Causes and Effects",
            "Reforestation and Afforestation",
            "Noise Pollution Sources Effects Control",
            "Case Studies Chipko Movement Arabari Experiment",
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Taxonomy helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_subject(subject: str) -> str:
    s = subject.strip().title()
    if s.upper() in ("MATHS", "MATH"):
        return "Mathematics"
    if s.upper() in ("BIO", "BIOLOGY", "BOTANY", "ZOOLOGY", "LIFE SCIENCE", "LIFE SCIENCES"):
        return "Biology"
    for key in _TAXONOMY:
        if key.upper() == s.upper():
            return key
    return s


def _get_hardcoded_taxonomy(subject: str) -> dict:
    key = _normalise_subject(subject)
    return _TAXONOMY.get(key, {})


async def _get_db_taxonomy(pool, subject: str) -> dict:
    if pool is None:
        return {}
    key = _normalise_subject(subject)
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT c.name AS chapter, t.name AS topic
                FROM chapters c
                JOIN subjects s ON s.id = c.subject_id
                LEFT JOIN topics t ON t.chapter_id = c.id
                WHERE LOWER(s.name) = LOWER($1)
                ORDER BY c.name, t.name
            """, key)
        taxonomy: dict = {}
        for row in rows:
            ch = row["chapter"]; tp = row["topic"]
            if ch not in taxonomy:
                taxonomy[ch] = []
            if tp:
                taxonomy[ch].append(tp)
        return taxonomy
    except Exception:
        return {}


async def _load_taxonomy(pool, subject: str) -> dict:
    hardcoded = _get_hardcoded_taxonomy(subject)
    db_tax    = await _get_db_taxonomy(pool, subject)
    if not db_tax:
        return hardcoded
    # Merge: DB wins if richer for a chapter; hardcoded fills gaps
    merged = dict(hardcoded)
    for ch, topics in db_tax.items():
        if ch not in merged or len(topics) > len(merged[ch]):
            merged[ch] = topics
    return merged


def _build_prompt_list(taxonomy: dict) -> str:
    """
    Numbered chapter list with all topics shown (full list for accurate topic matching).
    Sorted alphabetically so index is stable across calls.
    """
    lines = []
    for i, (ch, topics) in enumerate(sorted(taxonomy.items()), 1):
        topic_str = ", ".join(topics) if topics else ""
        lines.append(f"{i}. {ch}" + (f" → {topic_str}" if topic_str else ""))
    return "\n".join(lines)


def _build_full_topic_list(taxonomy: dict, chapter_name: str) -> str:
    """Return all topics for a specific chapter as a numbered list."""
    topics = taxonomy.get(chapter_name, [])
    return "\n".join(f"  {i+1}. {t}" for i, t in enumerate(topics))


def _build_full_taxonomy_prompt() -> str:
    """
    Builds a single numbered chapter list across ALL subjects for use when
    the subject of a question is unknown. Each entry is prefixed with the
    subject name so the LLM can identify both subject and chapter.
    Sorted alphabetically within each subject for a stable index.
    """
    lines = []
    idx = 1
    for subj in ("Physics", "Chemistry", "Mathematics", "Biology"):
        chapters = _TAXONOMY.get(subj, {})
        for ch, topics in sorted(chapters.items()):
            topic_str = ", ".join(topics) if topics else ""
            lines.append(
                f"{idx}. [{subj}] {ch}"
                + (f" → {topic_str}" if topic_str else "")
            )
            idx += 1
    return "\n".join(lines)


# Pre-built at module load — never changes at runtime.
_FULL_TAXONOMY_PROMPT: str = _build_full_taxonomy_prompt()

# Flat index: "1" → ("Physics", "Alternating Current"), etc.
# Same ordering as _build_full_taxonomy_prompt so numbers always match.
_FULL_CHAPTER_INDEX: dict[str, tuple[str, str]] = {}
_idx = 1
for _subj in ("Physics", "Chemistry", "Mathematics", "Biology"):
    for _ch in sorted(_TAXONOMY.get(_subj, {}).keys()):
        _FULL_CHAPTER_INDEX[str(_idx)] = (_subj, _ch)
        _idx += 1
del _idx, _subj, _ch   # clean up module-level temporaries


# ─────────────────────────────────────────────────────────────────────────────
# Single-question tagger — SYNC
# ─────────────────────────────────────────────────────────────────────────────

def _strip_latex(text: str) -> str:
    t = re.sub(r'\[IMAGE:[^\]]+\]', '', text)
    t = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', t)
    t = re.sub(r'\\sqrt\{([^}]*)\}', r'sqrt(\1)', t)
    t = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', t)
    t = re.sub(r'\\[a-zA-Z]+', ' ', t)
    t = re.sub(r'[${}\\]', '', t)
    return re.sub(r'\s+', ' ', t).strip()


def _tag_one_sync(
    api_key: str,
    q: dict,
    subject: str,
    taxonomy: dict,
    chapter_list_text: str,
    valid_chapters: set,
) -> dict:
    # Use "or" fallback so None values (DB nullable fields) don't crash .strip().
    # These lines are OUTSIDE the try/except block — an AttributeError here
    # silently kills every tagging task, which is why chapters never get written.
    already_has_chapter    = bool((q.get("chapter_name") or "").strip())
    already_has_difficulty = (q.get("difficulty") or "").strip() in ("easy", "medium", "hard")

    if already_has_chapter and already_has_difficulty:
        return q

    text = _strip_latex(q.get("question", ""))[:600]
    opts = "\n".join(
        f"({chr(65+i)}) {_strip_latex(o)}"
        for i, o in enumerate(q.get("options", [])[:4])
        if o and o.strip()
    )

    # ── Unknown-subject mode: use full cross-subject taxonomy ────────────────
    is_unknown = (subject == UNKNOWN_SUBJECT)

    if not already_has_chapter:
        if is_unknown:
            chapter_instructions = (
                "\nThe subject of this question is UNKNOWN."
                "\nThe FULL JEE taxonomy for Physics, Chemistry and Mathematics is below."
                "\nPick the NUMBER of the SINGLE best-matching chapter:\n"
                + _FULL_TAXONOMY_PROMPT
                + "\n\nRULES:"
                "\n- Respond with the chapter NUMBER only (e.g. 47), not the name."
                "\n- Also return the subject in the \"subject\" field: Physics, Chemistry, or Mathematics."
                "\n- You MUST pick from the numbered list above ONLY — no invented chapters."
                "\n- Pick the chapter the question is PRIMARILY about."
                "\n- If the question touches multiple topics, pick the MOST central one."
                "\n- Put the exact topic name in the \"topic\" field."
                "\n- Topic must come from the topics listed after → for that chapter."
                "\n- If unsure of topic, leave it as empty string."
            )
            chapter_output = (
                '"subject": "<Physics|Chemistry|Mathematics>", '
                '"chapter": "<number e.g. 47>", '
                '"topic": "<exact topic name or empty>", '
            )
        else:
            subj_title = _normalise_subject(subject)
            chapter_instructions = (
                f"\nThis is a {subj_title} question."
                f" ONLY the following {subj_title} chapters are valid —"
                " do NOT pick chapters from any other subject:\n"
                + chapter_list_text
                + "\n\nRULES:"
                "\n- Respond with the chapter NUMBER only (e.g. 5), not the name."
                "\n- You MUST pick from the numbered list above ONLY."
                "\n- Do NOT use chapters from Physics, Chemistry or Mathematics"
                " unless that is the current subject."
                "\n- Pick the chapter the question is PRIMARILY about."
                "\n- If the question touches multiple topics, pick the MOST central one."
                "\n- Put the exact topic name in the \"topic\" field."
                "\n- Topic must come from the topics listed after → for that chapter."
                "\n- If unsure of topic, leave it as empty string."
            )
            chapter_output = '"chapter": "<number e.g. 5>", "topic": "<exact topic name or empty>", '
    else:
        # Chapter already set — only fill in topic if missing
        subj_title = _normalise_subject(subject) if not is_unknown else "JEE"
        chapter_instructions = ""
        if not (q.get("topic_name") or "").strip():
            ch = q["chapter_name"]
            # For unknown-subject mode, look up the subject from the chapter name
            if is_unknown:
                for _s, _chapters in _TAXONOMY.items():
                    if ch in _chapters:
                        all_topics = _chapters[ch]
                        break
                else:
                    all_topics = []
            else:
                all_topics = taxonomy.get(ch, [])
            if all_topics:
                topic_list = ", ".join(all_topics)
                chapter_instructions = (
                    f'\nThe chapter is already set to "{ch}".'
                    f'\nTopics for this chapter: {topic_list}'
                    '\nPick the SINGLE most specific matching topic from this list.'
                    '\nPut it in the "topic" field. If none match, leave empty.'
                )
        chapter_output = f'"chapter": "{q["chapter_name"]}", "topic": "<exact topic name or empty>", '

    subj_label = "JEE Main" if is_unknown else f"JEE Main {_normalise_subject(subject)}"
    prompt = (
        f"You are an expert {subj_label} teacher classifying a question.\n\n"
        f"QUESTION:\n{text}\n"
        + (f"\nOPTIONS:\n{opts}\n" if opts else "")
        + chapter_instructions
        + "\n\nDIFFICULTY:\n"
        "- easy: single concept, direct formula, 1 step\n"
        "- medium: 2-3 steps, combine 2 concepts, moderate calculation\n"
        "- hard: multi-concept, non-obvious insight, lengthy or tricky approach\n"
        "\nReturn ONLY a JSON object. No markdown. No explanation.\n"
        f'Format: {{{chapter_output}"difficulty": "easy|medium|hard"}}'
    )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0,
        )
        raw  = resp.choices[0].message.content.strip()
        raw  = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw, flags=re.MULTILINE).strip()
        logger.debug(f"[tagger] Q{q.get('number','?')} raw={raw!r}")
        data = json.loads(raw)

        # ── Chapter resolution ───────────────────────────────────────────────
        if not already_has_chapter:
            ch_raw = str(data.get("chapter", "")).strip()
            tp     = str(data.get("topic",   "")).strip()

            if is_unknown:
                # ── Unknown-subject path ─────────────────────────────────────
                resolved_pair = _FULL_CHAPTER_INDEX.get(ch_raw)
                if resolved_pair:
                    resolved_subj, resolved_ch = resolved_pair
                else:
                    # LLM may have returned a name instead of a number — scan
                    resolved_pair = None
                    for _s, _chapters in _TAXONOMY.items():
                        for _ch in _chapters:
                            if _ch.lower() == ch_raw.lower():
                                resolved_pair = (_s, _ch)
                                break
                        if resolved_pair:
                            break
                    if resolved_pair:
                        resolved_subj, resolved_ch = resolved_pair
                    else:
                        resolved_subj = resolved_ch = ""

                if resolved_ch:
                    q["chapter_name"] = resolved_ch
                    q["subject"]      = resolved_subj   # write confirmed subject back
                    valid_topics = _TAXONOMY[resolved_subj].get(resolved_ch, [])
                    if tp and valid_topics:
                        if tp in valid_topics:
                            q["topic_name"] = tp
                        else:
                            tp_lower = tp.lower()
                            for vt in valid_topics:
                                if vt.lower() == tp_lower or tp_lower in vt.lower():
                                    q["topic_name"] = vt
                                    break
                            else:
                                if len(tp) > 3:
                                    q["topic_name"] = tp
                    elif tp:
                        q["topic_name"] = tp
                else:
                    logger.warning(
                        f"[tagger] Q{q.get('number','?')} unknown-subject mode:"
                        f" no chapter match for LLM value '{ch_raw}'"
                    )

            else:
                # ── Known-subject path (original logic, unchanged) ────────────
                subject_chapter_index = {
                    str(i): ch for i, ch in enumerate(sorted(valid_chapters), 1)
                }
                resolved = subject_chapter_index.get(ch_raw, "")
                if not resolved:
                    if ch_raw in valid_chapters:
                        resolved = ch_raw
                    else:
                        for vc in valid_chapters:
                            if vc.lower() == ch_raw.lower():
                                resolved = vc
                                break
                        if not resolved:
                            for vc in valid_chapters:
                                if ch_raw.lower() in vc.lower() or vc.lower() in ch_raw.lower():
                                    resolved = vc
                                    break

                # Safety: resolved chapter must belong to this subject's taxonomy
                if resolved and resolved not in valid_chapters:
                    logger.warning(
                        f"[tagger] Q{q.get('number','?')} resolved='{resolved}'"
                        f" not in {subject} taxonomy — discarding"
                    )
                    resolved = ""

                if resolved:
                    q["chapter_name"] = resolved
                    valid_topics = taxonomy.get(resolved, [])
                    if tp and valid_topics:
                        if tp in valid_topics:
                            q["topic_name"] = tp
                        else:
                            tp_lower = tp.lower()
                            for vt in valid_topics:
                                if vt.lower() == tp_lower:
                                    q["topic_name"] = vt
                                    break
                            else:
                                for vt in valid_topics:
                                    if tp_lower in vt.lower() or vt.lower() in tp_lower:
                                        q["topic_name"] = vt
                                        break
                                else:
                                    if len(tp) > 3:
                                        q["topic_name"] = tp
                    elif tp:
                        q["topic_name"] = tp
                else:
                    logger.warning(
                        f"[tagger] Q{q.get('number','?')} no chapter match for GPT='{ch_raw}'"
                    )

        elif not (q.get("topic_name") or "").strip():
            # Chapter already set, just filling topic
            tp = str(data.get("topic", "")).strip()
            if tp:
                ch = q["chapter_name"]
                if is_unknown:
                    valid_topics = []
                    for _s, _chapters in _TAXONOMY.items():
                        if ch in _chapters:
                            valid_topics = _chapters[ch]
                            break
                else:
                    valid_topics = taxonomy.get(ch, [])

                if not valid_topics or tp in valid_topics:
                    q["topic_name"] = tp
                else:
                    tp_lower = tp.lower()
                    for vt in valid_topics:
                        if vt.lower() == tp_lower or tp_lower in vt.lower():
                            q["topic_name"] = vt
                            break
                    else:
                        q["topic_name"] = tp

        # ── Difficulty ───────────────────────────────────────────────────────
        if not already_has_difficulty:
            diff = str(data.get("difficulty", "")).strip().lower()
            q["difficulty"] = diff if diff in ("easy", "medium", "hard") else "medium"

    except Exception as e:
        logger.warning(f"[tagger] Q{q.get('number','?')} failed: {e}")
        if not already_has_difficulty:
            q["difficulty"] = "medium"

    return q


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

async def tag_questions_async(
    questions:      list,
    subject:        str = "",
    pool            = None,
    max_concurrent: int = 8,
    openai_api_key: str = "",
) -> list:
    """
    Tag all questions with chapter, topic, difficulty using gpt-4o-mini.
    Groups by subject so multi-subject papers each get the right taxonomy.

    Subject resolution order for each question:
      1. q["subject"]  (must be non-empty and a recognised JEE subject)
      2. subject param (function argument, same validation)
      3. UNKNOWN_SUBJECT → full cross-subject taxonomy sent; LLM self-identifies
    """
    if not _OPENAI_AVAILABLE:
        logger.warning("[tagger] openai package not installed — skipping")
        return questions

    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.error("[tagger] No OPENAI_API_KEY — skipping tagging")
        return questions

    logger.info(f"[tagger] API key found, tagging {len(questions)} questions")
    logger.info(f"[tagger] Starting for {len(questions)} questions (key={api_key[:12]}...)")

    # ── Group questions by subject ───────────────────────────────────────────
    # FIX 1: use "or" chain so empty strings ("") also trigger the fallback,
    #         not just missing keys — q.get("subject", default) does NOT catch "".
    # FIX 2: unrecognised subjects go to UNKNOWN_SUBJECT instead of defaulting
    #         to Physics (which loaded the wrong taxonomy and caused hallucinations).
    subject_groups: dict[str, list] = {}
    for q in questions:
        raw_subj  = q.get("subject") or subject or ""
        normalised = _normalise_subject(raw_subj) if raw_subj else ""

        if normalised in _TAXONOMY:
            bucket = normalised
        else:
            if normalised:
                logger.warning(
                    f"[tagger] Q{q.get('number','?')} unrecognised subject"
                    f" '{normalised}' — will use full taxonomy for self-identification"
                )
            bucket = UNKNOWN_SUBJECT

        subject_groups.setdefault(bucket, []).append(q)

    logger.info(f"[tagger] Groups: { {s: len(qs) for s, qs in subject_groups.items()} }")

    # ── Load taxonomy per subject ────────────────────────────────────────────
    taxonomy_cache:       dict[str, dict] = {}
    chapter_list_cache:   dict[str, str]  = {}
    valid_chapters_cache: dict[str, set]  = {}

    for subj in subject_groups:
        if subj == UNKNOWN_SUBJECT:
            # Full taxonomy is used directly inside _tag_one_sync via module-level
            # constants — no per-call taxonomy needed.
            taxonomy_cache[subj]       = {}
            chapter_list_cache[subj]   = ""
            valid_chapters_cache[subj] = set()
        else:
            tax = await _load_taxonomy(pool, subj)
            taxonomy_cache[subj]       = tax
            chapter_list_cache[subj]   = _build_prompt_list(tax)
            valid_chapters_cache[subj] = set(tax.keys())

    # ── Dispatch all questions concurrently ──────────────────────────────────
    loop      = asyncio.get_running_loop()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _run_one(q, subj):
        async with semaphore:
            return await loop.run_in_executor(
                None,
                _tag_one_sync,
                api_key,
                q,
                subj,
                taxonomy_cache[subj],
                chapter_list_cache[subj],
                valid_chapters_cache[subj],
            )

    tasks = []
    for subj, qs in subject_groups.items():
        for q in qs:
            tasks.append(_run_one(q, subj))

    await asyncio.gather(*tasks)

    tagged = sum(1 for q in questions if (q.get("chapter_name") or "").strip())
    diffed = sum(1 for q in questions if (q.get("difficulty") or "").strip() in ("easy", "medium", "hard"))
    logger.info(f"[tagger] Done — {tagged}/{len(questions)} chapter, {diffed}/{len(questions)} difficulty")

    return questions