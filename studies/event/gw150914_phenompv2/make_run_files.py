import os

INI_TEMPLATE = """
label = LABEL
outdir = OUTDIR
prior-file = PRIORFILE


time = 9:00:00
ntasks-per-node = 16
nodes = 2
mem-per-cpu="2GB"


################################################################################
## Constant arguments
################################################################################

frequency-domain-source-model = deep_gw_pe_followup.restricted_source_model.lal_binary_black_hole_qxeff_restricted
enforce_signal_duration = False
trigger-time = 1126259462.4
gaussian-noise = False
detectors = [H1]
duration = 4
coherence-test=False
minimum-frequency = 20
sampling-frequency = 4096
maximum-frequency = 2048
reference-frequency = 20.0
channel_dict = {H1:DCS-CALIB_STRAIN_C02}
data-dict={H1=../data/h1.gwf}
psd-dict = {H1=../data/h1_psd.txt}

distance-marginalization= False
phase-marginalization = False
time-marginalization = False
jitter-time = False


waveform-approximant = IMRPhenomPv2
nlive = 1000
nact = 5
check-point-deltaT=36000

extra-lines = "module --force purge && module load git/2.18.0 git-lfs/2.4.0 gcc/9.2.0 openmpi/4.0.2 numpy/1.19.2-python-3.8.5 mpi4py/3.0.3-python-3.8.5 && module unload zlib && source /fred/oz980/avajpeyi/envs/sstar_venv/bin/activate"

"""

PRIOR_TEMPLATE = """
mass_ratio = MASS_RATIO
chi_eff = CHI_EFF

a_1 = deep_gw_pe_followup.restricted_prior.PlaceholderPrior(name='a_1')
cos_tilt_1 = deep_gw_pe_followup.restricted_prior.PlaceholderPrior(name='cos_tilt_1')
cos_tilt_2  = deep_gw_pe_followup.restricted_prior.PlaceholderPrior(name='cos_tilt_2')
a_2 = Constraint(name="a_2", minimum=0, maximum=1)
chirp_mass = Uniform(minimum=25, maximum=31, name='chirp_mass', latex_label='$\\mathcal{M}$', unit=None, boundary=None)
phi_12 = Uniform(minimum=0, maximum=6.283185307179586, name='phi_12', latex_label='$\\Delta\\phi$', unit=None, boundary='periodic')
phi_jl = Uniform(minimum=0, maximum=6.283185307179586, name='phi_jl', latex_label='$\\phi_{JL}$', unit=None, boundary='periodic')

cos_theta_jn = Uniform(minimum=-1, maximum=1, name='cos_theta_jn', latex_label='$\\cos\\theta_{JN}$', unit=None, boundary=None)
phase = 0
luminosity_distance = 500
ra = 2.64433731
dec = -1.25977494
psi = 1.5707948076735136
geocent_time = 1126259462.4
"""

SUBMIT_TEMPLATE = """#!/bin/bash
#
#SBATCH --job-name=setup_followup
#SBATCH --output=setup.log
#
#SBATCH --ntasks=1
#SBATCH --time=00:40:00
#SBATCH --mem=2GB
#SBATCH --cpus-per-task=1
#SBATCH --array=0-5

module --force purge && module load git/2.18.0 git-lfs/2.4.0 gcc/9.2.0 openmpi/4.0.2 numpy/1.19.2-python-3.8.5 mpi4py/3.0.3-python-3.8.5
module unload zlib
source /fred/oz980/avajpeyi/envs/sstar_venv/bin/activate

deep_followup_setup pt_${SLURM_ARRAY_TASK_ID}.ini
"""

PTS = {
    "0":dict(q=0.85, xeff=0),
    "1":dict(q=0.9, xeff=0),
    "2":dict(q=0.95, xeff=0),
    "3":dict(q=0.85, xeff=0.05),
    "4":dict(q=0.85, xeff=0.1)
}


def make_files():
    outdir = "out"
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(f"{outdir}/priors", exist_ok=True)
    for label, pt in PTS.items():
        make_pt_file(label, pt, outdir)

    with open(f"{outdir}/deep_setup.sh", "w") as f:
        f.write(SUBMIT_TEMPLATE)

def make_pt_file(label, pt, outdir):
    print(f"Making files for pt: {label}")
    ini = f"{outdir}/pt_{label}.ini"
    prior = f"priors/pt_{label}.prior"

    with open(ini,"w") as f:
        ini_txt = INI_TEMPLATE
        l = f"pt{label}"
        ini_txt = ini_txt.replace("LABEL", l)
        ini_txt = ini_txt.replace("OUTDIR", f"out_{l}")
        ini_txt = ini_txt.replace("PRIORFILE", prior)
        f.write(ini_txt)

    with open(f"{outdir}/{prior}","w") as f:
        pri_txt = PRIOR_TEMPLATE
        pri_txt = pri_txt.replace("MASS_RATIO", str(pt['q']))
        pri_txt = pri_txt.replace("CHI_EFF", str(pt['xeff']))
        f.write(pri_txt)







if __name__ == "__main__":
    make_files()
