# Hälsostudie - Data analys i Python

Detta projekt är en tvådelad analys av ett hälsodataset med information om bland annat ålder, kön, längd, vikt, blodtryck, kolesterol, rökvanor och sjukdomsförekomst.  
Arbetet utfördes i Jupyter Notebook och versionhanterades i två separata branches enligt kursens upplägg.  
**Den färdiga och mest omfattande analysen finns i branch del2.**

## Projektstruktur
Repositoryt innehåller tre branches:  
- **del1** – Grundläggande analys, statistik, simuleringar och hypotesprövning.  
- **del2** – del1, fördjupad analys, pipeline-struktur, funktioner, klasser och linjär algebra.  
- **main** – Tom branch (avsiktligt), används endast för struktur.

## Innehåll
- Deskriptiv statistik och flera visualiseringar;
- Simuleringar (inkl. sannolikheter och konfidensintervall);
- Hypotesprövning och power-analys;
- Objektorienterad struktur (klasser + moduler);
- Linjär regression (enkel och multipel) och PCA.

Datasetet finns i *data/health_study_dataset.csv*. 

## Miljö
- Python: 3.13.7
- Paket: Numpy, Pandas, Matplotlib, Scipy, Jupyter, Scikit-learn (se requirements.txt)

## Kom igång
**klona projetet**  
git clone https://github.com/IreneGrisenti/health_study_project.git  
cd health_study_project  
git checkout -b del2 origin/del2  
  
eller  

**klona bara del2 branch**  
git clone -b del2 --single-branch https://github.com/IreneGrisenti/health_study_project.git  
cd health_study_project

**Skapa och aktivera virtuell miljö**:  
python -m venv .venv

**Windows PowerShell**:  
.venv\Scripts\Activate

**macOS/Linux**:  
source .venv/bin/activate

**Installera beroenden**:  
python -m pip install -r requirements.txt