# wearable-prediction

[![CI](https://github.com/tahnok/colmi_r02_client/actions/workflows/ci.yml/badge.svg)](https://github.com/tahnok/colmi_r02_client/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

## Problem statement & Journey

If an athletes get injured, she/he lose the season. Cortisol is the key ormon to spot dangerous stress buildup.

We originally wanted to make a cortisol patch with MIP technology, but while we wait for the reagents we built a cheap version using colorimetry.

<img src="https://raw.githubusercontent.com/July98293/werable-prediction/main/5954255193178312126.jpg" width="200">
<img src="https://raw.githubusercontent.com/July98293/werable-prediction/main/5954255193178312127.jpg" width="200">
<img src="https://raw.githubusercontent.com/July98293/werable-prediction/main/5954255193178312128.jpg" width="200">

It work with enzimatic pathways 

```
cortisolo --[OH⁻, EtOH]--> enediolo
enediolo + resazurina --> cortisolo ossidato (21-aldeide/acido etianico) + resorufina
```
it also is reduce by glucose and ascorbate much more concentrated than cortisol in sweath so i tneed a filtration layer, or another rection that use nezime could be

```
cortisolo + NAD⁺ →[11β-HSD2]→ cortisone + NADH → [diaforasi] → resorufina
```

Once cortisol is detected the blue patch turn purple/reddish. The other one is for lactate, and it turn complete transparent, is secreated during anaerobic stress.

We also modified a wearable to predict injury and stress with ML, while building the cortisol database and the ML for hormones.

<img src="https://raw.githubusercontent.com/July98293/werable-prediction/main/5954255193178312125.jpg" width="300">

We start from the Colmi R02 repo and connect **stress** and **injury** prediction, which we previously trained separately, and finally top it all off with a dashboard that shows the data.

You can find detailed info on the stress ML and the injury ML below.


> **Disclaimer:** 1)not all model and complete app is uploaded, stress one is in another repo, the main part of injury is in this one 2) Injury prediction is genuinely hard, it's a rare recorded event and a near-unsolved problem in sports science. PR-AUC here is barely above the ~0.6% base rate. This is *not* a bug: the full, non-reduced model in `injury 2/` has the same ceiling (see its README's "Second pass" section). Precision is low bc the data ara umbaance with few recorded case, meaning most "high risk" flags will be false alarms. This is why `colmi_r02_client predict-injury` intentionally only ever reports a coarse **high / medium / low risk over the next 3 days**, not a percentage, see `colmi_r02_client/injury_predict.py`'s docstring for exactly which inputs are real ring data vs. proxies, and treat the output as illustrative, never as medical or coaching advice.

Regenerate these plots with `python "injury 2/scripts/generate_ring_report_plots.py"`.

### Future vision: cortisol & hormones

- **Hormones** are hard because they require night-sleep data, which in the Colmi is still a developing feature.
- **Cortisol** the dataset is missing, so we're creating it ourselves with cortisol strips.

Roadmap priority:

1. Stress
2. Injury

---

Open source python client to read your data from the Colmi R02 family of Smart Rings, a $20 sensor package (HR, SpO2, accelerometer, sleep) with no official SDK. 100% open source, 100% offline.

- **Accelerometer**
  - step tracking
  - sleep tracking
- **Heart Rate (HR)**
- **Blood Oxygen (SpO2)**

[Source code on GitHubfor colmi-02/Credit](https://github.com/tahnok/colmi_r02_client)


## Reverse engineering status

- [x] Real time heart rate and SpO2
- [x] Step logs (still don't quite understand how the day is split up)
- [x] Heart rate logs (aka periodic measurement)
- [x] Set ring time
- [x] Set HR log frequency
- [ ] SpO2 logs
- [~] Sleep tracking (implemented in `colmi_r02_client/sleep.py`, but experimental / not yet verified against a real ring, see below)
- [x] "Stress" measurement (ML model on top of heart rate, see below)

---
## AI features: stress & injury-risk prediction

### Stress prediction

see the model in detail at ![STRESS PREDICTION ML](https://github.com/July98293/stress-ml/blob/main/README.md)

### Injury-risk prediction

Estimates injury risk over the next 3 days from the last week of ring history. This is adapted from a separate sports-science project (`injury 2/`, a SoccerMon-based injury prediction pipeline for professional athletes) The ring version uses a **separately retrained, reduced model** (`injury 2/model_artifact_ring/`) on only the 4 features: `stress` (HRV-based), `sleep_duration` / `sleep_quality` (from the experimental sleep-sync protocol), and `fatigue` (a movement-volume proxy from steps/calories, standing in for training load).

 <img src="https://github.com/July98293/werable-prediction/blob/main/assets/model_readiness.png?raw=true" alt="Injury ring model calibration" width="450">
<img src="https://github.com/July98293/werable-prediction/blob/main/assets/03_confusion_matrix.png?raw=true" alt="Confusion matrix" width="450">
<img src="https://github.com/July98293/werable-prediction/blob/main/assets/05_risk_trajectory_2_TeamA_3e5f6e2b.png?raw=true" alt="Risk trajectory" width="450">


## Dashboard

A small local web dashboard (`colmi_r02_client/webapp.py`) puts all of the above in one page: sleep, steps / heart rate / HRV / SpO2, the stress read, and the injury-risk band, styled as rounded cards over synced + live ring data.

IMAGE

```sh
colmi_r02_dashboard --address=70:CB:0D:D0:34:1C --db=ring_data.sqlite
```

Then open <http://127.0.0.1:5050>. `--address` is optional  historical charts work from a database populated by `colmi_r02_client sync` alone; a live snapshot (current HR/SpO2/HRV, today's activity, last night's sleep, stress, and injury risk) needs an address, entered in the page if not passed on the command line. A live snapshot does ~20–30 sequential BLE reads (the injury model alone needs a 14-day movement baseline plus a week of heart-rate logs), so it can take up to a minute, that's normal, not a hang.

![Ring dashboard](https://github.com/July98293/werable-prediction/blob/main/dashboard.png)

---


## Getting started

### Using the command line


```sh
pipx install git+https://github.com/tahnok/colmi_r02_client
```

Once that is done you can look for nearby rings using:

```sh
colmi_r02_util scan
```

```
Found device(s)
                Name  | Address
--------------------------------------------
            R02_341C  |  70:CB:0D:D0:34:1C
```

Once you have your address you can use it to do things like get real time heart rate:

```sh
colmi_r02_client --address=70:CB:0D:D0:34:1C get-real-time heart-rate
```

```
Starting reading, please wait.
[81, 81, 79, 79, 79, 79]
```

You can also sync the data from your ring to sqlite:

```sh
colmi_r02_client --address=3A:08:6A:6F:EB:EC sync
```

```
Writing to /home/wes/src/colmi_r02_client/ring_data.sqlite
Syncing from 2024-12-01 01:43:04.723232+00:00 to 2024-12-01 02:03:20.150315+00:00
Done
```

The most up to date and comprehensive help for the command line can be found by running:

```sh
colmi_r02_client --help
```

```
Usage: colmi_r02_client [OPTIONS] COMMAND [ARGS]...

Options:
  --debug / --no-debug
  --record / --no-record  Write all received packets to a file
  --address TEXT          Bluetooth address
  --name TEXT             Bluetooth name of the device, slower but will work
                          on macOS
  --help                  Show this message and exit.

Commands:
  get-heart-rate-log           Get heart rate for given date
  get-heart-rate-log-settings  Get heart rate log settings
  get-real-time-heart-rate     Get real time heart rate.
  get-steps                    Get step data
  info                         Get device info and battery level
  raw                          Send the ring a raw command
  reboot                       Reboot the ring
  set-heart-rate-log-settings  Get heart rate log settings
  set-time                     Set the time on the ring, required if you...
  sync                         Sync all data from the ring to a sqlite...
```

# Reference

### Stress

1. [Correlazioni tra biomarcatori e stress](https://www.biorxiv.org/content/10.1101/2023.09.16.557862v1.full) - *bioRxiv*
2. [Validazione sensore da polso per HRV/EDA in guida stressante](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10611310/) - *PMC*
3. [ML per stress da wearable](https://www.sciencedirect.com/science/article/pii/S1386505623000436) - *review sistematica, ScienceDirect*
4. [Validazione Fitbit Charge 5 per HR ed EDA](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12308623/) - *PMC*
5. [EOG + EDA per biomarcatori di ansia](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12270033/) - *cold pressor, PMC*
6. [Wearable e rilevazione precoce dello stress](https://www.medrxiv.org/content/10.1101/2024.07.19.24310732.full.pdf) - *multimodale, medRxiv*
7. [Stress-Predict Dataset](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9654418/) - *PPG, FC, respiro*
8. [Sistema di allerta real-time COVID/stress da wearable](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8799466/) - *Nature*
9. [Rilevazione infezioni respiratorie con wearable](https://formative.jmir.org/2024/1/e53716) - *JMIR*
10. [Validazione modello su operatori sanitari](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11292157/) - *PMC*
11. [Algoritmo smartwatch per infezione polmonare](https://pmc.ncbi.nlm.nih.gov/articles/PMC11512465/) - *PMC*
12. [Wearable, smartphone e IA contro COVID-19](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8709136/) - *HRV/BPM*

### Cortisol

1. [Classificazione cortisolo salivare basata su HRV](https://www.researchgate.net/publication/390800019_The_salivary_cortisol_classification_based_on_the_heart_rate_variability) - *ResearchGate*
2. [Stima del cortisolo da bracciale wearable](https://www.researchgate.net/publication/393453678_Cortisol_Estimation_using_Wearable_Wristbands_Comparing_Multilevel_Modelling_and_Machine_Learning) - *ML, ResearchGate*
3. [Classificazione della fatica con HRV e cortisolo](https://www.nature.com/articles/s41746-025-02320-8) - *approccio multimodale, npj Digital Medicine*
4. [Distinguere stress fisico e psicologico da segnali wearable e cortisolo salivare](https://arxiv.org/pdf/2604.12671) - *arXiv*
5. [Cortisolo e sonno: impatto del sonno sull'asse HPA](https://pmc.ncbi.nlm.nih.gov/articles/PMC2902103/) - *PMC*
6. [Armonizzazione dei dati](https://pmc.ncbi.nlm.nih.gov/articles/PMC8631396/pdf/ijpds-06-1680.pdf) - *IJPDS*

### Injury

1. [Predizione infortuni negli atleti con carico e wellness (Jiang et al.)](https://pubmed.ncbi.nlm.nih.gov/36972679/) - *PMID 36972679, modello a 3 biomarcatori, PubMed*
2. [SoccerMon: dataset longitudinale di monitoraggio di calciatrici d'élite](https://www.nature.com/articles/s41597-024-03212-4) - *GPS, wellness, Scientific Data*
3. [Machine learning per la predizione di infortuni nello sport: review](https://bmcsportsscimedrehabil.biomedcentral.com/articles/10.1186/s13102-024-00860-2) - *review sistematica, BMC*
4. [Cortisolo come marcatore di overtraining e rischio infortunio](https://pmc.ncbi.nlm.nih.gov/articles/PMC6835946/) - *rassegna asse HPA e carico, PMC*
5. [Carico di allenamento e rischio infortunio (acute:chronic workload ratio)](https://bjsm.bmj.com/content/50/5/273) - *Gabbett, BJSM*
6. [HRV come indicatore di recupero e prontezza atletica](https://pmc.ncbi.nlm.nih.gov/articles/PMC5900775/) - *review, PMC*
