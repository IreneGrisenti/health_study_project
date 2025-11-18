import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.power import TTestIndPower

class HealthAnalyzer:

# Initialize Analyser
    def __init__(self, df):         
        self.df = df.copy()

# Snabb kontroll
    def explore(self):
        print("Data sample:")       
        print(self.df.sample(5))
        print("\nInfo:")  
        print(self.df.info())
        print("\nDescriptive statistics:") 
        print(self.df.describe())

# Cleaning
    def cleaning(self):             
        print("Saknade värde:\n", self.df.isna().sum())
        print("Duplicerade rader: ", self.df.duplicated().sum())

    def compute_stats(self, columns):        # Descriptive stats only specific columns
        return self.df[columns].agg(["mean", "median", "min", "max"])

# Adding BMI column
    def bmi_col(self):
        self.df["bmi"] = self.df["weight"] / (self.df["height"] / 100)**2
        return self.df

# Visualiseringar
# Population overview
    def plot_population_overview(self, save_path=None): 
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle("Understanding sample population", fontsize=17)

    # Histogram: åldersfördelning efter kön
        for s, subset in self.df.groupby("sex"):
            ax1.hist(subset["age"], edgecolor="#777777", bins=20, label=f"{s}", alpha=0.4)
        ax1.set_title("Age distribution per gender")
        ax1.set_xlabel("Age")
        ax1.set_ylabel("Frequence")
        ax1.legend()

    # Stapeldiagram: andel rökare
        smoker_per = (self.df["smoker"].value_counts() / self.df["smoker"].count()) * 100
        ax2.bar(smoker_per.index, smoker_per.values, 
                color="#FBC15E", width=0.7, alpha=0.7)
        ax2.set_title("Non-Smokers vs Smokers")
        ax2.set_xlabel("Smokers")
        ax2.set_ylabel("%")
        for i, v in enumerate(smoker_per.values):
            ax2.text(i, v + 1, f"{v:.1f}%", ha="center")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path)
        plt.show()

# Health metrics
    def plot_health_metrics(self, save_path=None):

    # Histogram: BMI fördelning
        fig, (ax3, ax4) = plt.subplots(1, 2)
        fig.suptitle("Checking health metrics", fontsize=17)

        if "bmi" not in self.df.columns:
            self.bmi_col()

        ax3.hist(self.df["bmi"], color="#8EBA42", edgecolor="#777777", bins=20, alpha=0.7)
        ax3.set_title("BMI distribution")
        ax3.set_xlabel("BMI (kg/m²)")
        ax3.set_ylabel("Frequency")

            
    # Boxplot: kolesterol efter gender
        ax4.boxplot([self.df.loc[self.df.sex == "F", "cholesterol"], 
                self.df.loc[self.df.sex == "M", "cholesterol"]], 
                tick_labels=["Female", "Male"], 
                showmeans=True)
        ax4.set_title("Cholesterol distribution per gender")
        ax4.set_xlabel("Gender")
        ax4.set_ylabel("Cholesterol level (mmol/L)")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path)
        plt.show()

# Relationships
    def plot_relationships(self, save_path=None):
    # Scatter: age vs cholesterol
        fig, (ax5, ax6) = plt.subplots(1, 2)
        fig.suptitle("Exploring relationships", fontsize=17)

        ax5.scatter(self.df["age"], self.df["cholesterol"], color="#988ED5", edgecolor="#777777", alpha=0.4)
        ax5.set_title("Age vs Cholesterol")
        ax5.set_xlabel("Age")
        ax5.set_ylabel("Cholesterol level (mmol/L)")

    # Boxplot: systoliskt blodtryck efter grupp med/utan sjukdom
        ax6.boxplot([self.df.loc[self.df.disease == 1, "systolic_bp"], 
                self.df.loc[self.df.disease == 0, "systolic_bp"]], 
                tick_labels=["Disease", "No disease"], 
                showmeans=True)
        ax6.set_title("Systolic bp distribution")
        ax6.set_xlabel("Group type")
        ax6.set_ylabel("Systolic blood pressure (mmHg)")
        plt.tight_layout()
        if save_path:
                fig.savefig(save_path)
        plt.show()

# Simulation
    def simulation_disease(self, n=1000, seed=42):
        rng = np.random.default_rng(seed)  # Local random number generato
        p_disease = self.df["disease"].mean() # Andelen personer i datasetet som har sjukdomen
        simulated = rng.choice([0,1], size=n, p=[1-p_disease, p_disease]) # Simulering av 1000 slumpmässiga individer med samma sjukdomsandel
        p_simulation = simulated.mean() * 100 # Räkna andelen personer med sjukdom i simulering
        return f"""Verklig sjukdomsfrekvens: {p_disease * 100:.2f}%.
Simulerad sjukdomsfrekvens: {p_simulation:.2f}%

Slutsats:
Skillnaden mellan den verkliga sjukdomsfrekvensen och den simulerade sjukdomsfrekvensen är liten 
och ligger inom den förväntade variationen."""
# Will fix the output

# Function CI 95% + plot
    def plot_systolic_bp_confidence_intervals(self, confidence=0.95, B=3000, seed=42, save_path=None):
        
        rng = np.random.default_rng(seed)  # Local random number generator
        systolic_bp = np.array(self.df["systolic_bp"], dtype=float)

        # Summary stats
        m = np.mean(systolic_bp)
        s = np.std(systolic_bp, ddof=1)
        n = len(systolic_bp)

        # Normal approximation CI
        z = 1.96
        margin_error = z * (s / np.sqrt(n))
        lo = m - margin_error
        hi = m + margin_error

        # Bootstrap CI
        boot_means = np.empty(B)
        for b in range(B):
            boot_sample = rng.choice(systolic_bp, size=n, replace=True)
            boot_means[b] = np.mean(boot_sample)

        alpha = (1 - confidence) / 2
        bmean = np.mean(boot_means)
        blo, bhi = np.percentile(boot_means, [100*alpha, 100*(1 - alpha)])

        # Error bar: jämförelse mellan konfidensintervall beräknat med normalapproximation och med bootstrap
        fig, ax = plt.subplots()
        ax.errorbar(0, m, yerr=margin_error, fmt="o", capsize=5, label="Normal Approximation")
        ax.errorbar(1, bmean, yerr=[[bmean - blo], [bhi - bmean]],
                    fmt="o", capsize=5, label="Bootstrap")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Normal Approx.", "Bootstrap"])
        ax.set_ylabel("Systolic BP (mmHg)")
        ax.set_title(f"{int(confidence*100)}% Confidence Intervals for Systolic BP")
        ax.legend()

        if save_path:
            fig.savefig(save_path)

        plt.show()

        return {
            "Punktuppskattning": m,
            "Standardavvikelse": s,

            "Resultatet av normalapproximationen är": (lo, hi),

            "Resultatet från bootstrap är": (blo, bhi),
            "Medelvärde för statistiken": bmean}
    # Will fix the output