# Hälsostudie
Denna rapport analyserar en hälsodatamängd med fokus på blodtryck, rökning och andra hälsoindikatorer. Syftet är att beskriva data, genomföra hypotesprövning, beräkna konfidensintervall och utvärdera tillförlitligheten av resultaten genom simuleringar och power analys.

## Rapport
### Beskrivande analys
- Sammanfattande statistik (medelvärde, median, min, max) beräknas för:
age, weight, height, systolic_bp, cholesterol.
- Visualiseringar

### Simulering
- Beräknade andelen personer i datasetet som har sjukdomen.
- Simulerade 1000 slumpmässiga individer med samma sannolikhet för sjukdom.
- Jämförde de simulerade proportionerna med de observerade i datasetet.

### Konfidensintervall
- Beräknade 95%-konfidensintervall för medelvärdet av systoliskt blodtryck med två metoder: normalapproximation och bootstrap

### Hypotesprövning
- Testade hypotesen: "Rökare har högre medelblodtryck än icke-rökare."
- Använde t-test eller bootstrap-metoder.
- Tolkades och förklarades resultaten.

### Power
- Simulerade den statistiska power för att upptäcka en verklig skillnad.
- Beräknade vilken skillnad i medelvärde som krävs för att uppnå 80% styrka med givna stickprovsstorlekar.
- Motiverade metodval och förklarade resultaten.

## Miljö
- Python: 3.13.7
- Paket: Numpy, Pandas, Matplotlib, Scipy, Jupyter (se requirements.txt)

## Kom igång
**klona projetet**  
git clone https://github.com/IreneGrisenti/health_study_project.git  
cd health_study_project

**Skapa och aktivera virtuell miljö**:  
python -m venv .venv

**Windows PowerShell**:  
.venv\Scripts\Activate

**macOS/Linux**:  
source .venv/bin/activate

**installera beroenden**:  
python -m pip install -r requirements.txt