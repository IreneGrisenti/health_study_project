import numpy as np
import matplotlib.pyplot as plt


class HealthAnalyzer:
    """
    A class for exploring, cleaning, visualizing, and analyzing 
    health-related datasets.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing variables such as age, sex, weight, 
        height, blood pressure, cholesterol, etc.

    Attributes
    ----------
    df : pandas.DataFrame
        A copy of the provided dataset.
    """

    def __init__(self, df):
        """
        Initialize the HealthAnalyzer with a dataset.

        Parameters
        ----------
        df : pandas.DataFrame
            Input data.
        """
        self.df = df.copy()

    def explore(self):
        """
        Print a basic exploratory summary of the dataset.

        Outputs
        -------
        - Random sample of rows
        - DataFrame info (types, missing values, memory usage)
        - Descriptive statistics for numeric columns
        """
        print("Data sample:")
        print(self.df.sample(5))
        print("\nInfo:")
        print(self.df.info())
        print("\nDescriptive statistics:")
        print(self.df.describe())

    def cleaning(self):
        """
        Print the number of missing values and duplicated rows in the dataset.
        """
        print("\nSaknade värde:\n", self.df.isna().sum())
        print("\nDuplicerade rader: ", self.df.duplicated().sum())

    def compute_stats(self, columns):
        """
        Compute descriptive statistics for selected columns.

        Parameters
        ----------
        columns : list of str
            Column names to compute statistics for.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing mean, median, min, and max.
        """
        return self.df[columns].agg(["mean", "median", "min", "max"])

    def bmi_col(self):
        """
        Add a BMI column to the dataset: weight / height².

        Returns
        -------
        pandas.DataFrame
            The updated DataFrame including the "bmi" column.
        """
        self.df["bmi"] = self.df["weight"] / (self.df["height"] / 100)**2
        return self.df

    # Population overview
    def plot_population_overview(self, save_path=None):
        """
        Plot population-level distributions:
        - Age distribution per gender (histogram)
        - Smoking status distribution (bar chart)

        Parameters
        ----------
        save_path : str, optional
            Path to save the generated figure. If None, the plot is not saved.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle("Understanding sample population", fontsize=17)

        # Histogram: age distribution by gender
        for s, subset in self.df.groupby("sex"):
            ax1.hist(subset["age"], edgecolor="#777777",
                     bins=20, label=f"{s}", alpha=0.4)
        ax1.set_title("Age distribution per gender")
        ax1.set_xlabel("Age")
        ax1.set_ylabel("Frequence")
        ax1.legend()

        # Bar chart: smoker ratio
        smoker_per = (self.df["smoker"].value_counts() /
                      self.df["smoker"].count()) * 100
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
        """
        Plot key health metrics:
        - BMI distribution (histogram)
        - Cholesterol distribution by gender (boxplot)

        Parameters
        ----------
        save_path : str, optional
            Path to save the figure.
        """

        fig, (ax3, ax4) = plt.subplots(1, 2)
        fig.suptitle("Checking health metrics", fontsize=17)

        # Add BMI if missing
        if "bmi" not in self.df.columns:
            self.bmi_col()
        # BMI histogram
        ax3.hist(self.df["bmi"], color="#8EBA42",
                 edgecolor="#777777", bins=20, alpha=0.7)
        ax3.set_title("BMI distribution")
        ax3.set_xlabel("BMI (kg/m²)")
        ax3.set_ylabel("Frequency")

        # Cholesterol boxplot by gender
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
        """
        Plot relationships between variables:
        - Age vs cholesterol (scatter)
        - Systolic BP distribution for disease vs no disease (boxplot)

        Parameters
        ----------
        save_path : str, optional
            Path to save the figure.
        """
        fig, (ax5, ax6) = plt.subplots(1, 2)
        fig.suptitle("Exploring relationships", fontsize=17)

        # Scatter: age vs cholesterol
        ax5.scatter(self.df["age"], self.df["cholesterol"],
                    color="#988ED5", alpha=0.5)
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

    def simulation_disease(self, n=1000, seed=42):
        """
        Simulate disease occurrence based on the empirical disease rate 
        in the dataset.

        Parameters
        ----------
        n : int, optional
            Number of simulated individuals. Default is 1000.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        str
            A formatted string comparing real and simulated disease frequency.
        """
        rng = np.random.default_rng(seed)
        p_disease = self.df["disease"].mean()
        simulated = rng.choice([0, 1], size=n, p=[1-p_disease, p_disease])
        p_simulation = simulated.mean() * 100
        p_real = p_disease * 100

        return {
            "verklig förekomst"     : p_real,
            "simulerad förekomst"   : p_simulation,
            "skillnad"              : round((p_simulation - p_real), 2)
        }

    def plot_systolic_bp_confidence_intervals(self, confidence=0.95, B=3000, seed=42, save_path=None):
        """
        Compute and visualize confidence intervals for systolic blood pressure 
        using both normal approximation and bootstrap resampling.

        Parameters
        ----------
        confidence : float, optional
            Confidence level for the intervals. Default is 0.95.
        B : int, optional
            Number of bootstrap samples. Default is 3000.
        seed : int, optional
            Random seed for reproducibility.
        save_path : str, optional
            Path to save the generated plot.

        Returns
        -------
        dict
            Dictionary containing:
            - Punktuppskattning : float
            - Standardavvikelse : float
            - Normalapproximation CI : tuple (lower, upper)
            - Bootstrap CI : tuple (lower, upper)
            - Bootstrap mean : float
        """
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
        ax.errorbar(0, m, yerr=margin_error, fmt="o",
                    capsize=5, label="Normal Approximation")
        ax.errorbar(1, bmean, yerr=[[bmean - blo], [bhi - bmean]],
                    fmt="o", capsize=5, label="Bootstrap")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Normal Approx.", "Bootstrap"])
        ax.set_ylabel("Systolic BP (mmHg)")
        ax.set_title(
            f"{int(confidence*100)}% Confidence Intervals for Systolic BP")
        ax.legend()

        if save_path:
            fig.savefig(save_path)

        plt.show()

        print(
            f"Punktuppskattning                     : {m:.2f}\n"
            f"Standardavvikelse                     : {s:.2f}\n\n"

            f"Resultatet av normalapproximationen är: {lo:.2f} - {hi:.2f}\n\n"

            f"Resultatet från bootstrap är          : {blo:.2f} - {bhi:.2f}\n"
            f"Medelvärde för statistiken            : {bmean:.2f}\n\n"
        )

        return {
            "Punktuppskattning": m,
            "Standardavvikelse": s,
            "Resultatet av normalapproximationen är": (lo, hi),
            "Resultatet från bootstrap är": (blo, bhi),
            "Medelvärde för statistiken": bmean}