
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("../data/hospital_patients.csv")

print("Dataset loaded successfully!")
print(df.head())
print("\nDataset Shape:", df.shape)

# -----------------------------
# 2. Data Cleaning
# -----------------------------
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Remove duplicate records
df = df.drop_duplicates()

# Convert date columns
date_columns = ["admission_date", "discharge_date"]

for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

# Remove rows with missing important values
df = df.dropna(subset=["patient_id"])

# -----------------------------
# 3. Calculate Length of Stay
# -----------------------------
if "admission_date" in df.columns and "discharge_date" in df.columns:
    df["length_of_stay"] = (
        df["discharge_date"] - df["admission_date"]
    ).dt.days

    df["length_of_stay"] = df["length_of_stay"].clip(lower=0)

# -----------------------------
# 4. Basic KPIs
# -----------------------------
total_patients = df["patient_id"].nunique()

total_admissions = len(df)

total_revenue = df["billing_amount"].sum()

average_bill = df["billing_amount"].mean()

if "length_of_stay" in df.columns:
    average_stay = df["length_of_stay"].mean()
else:
    average_stay = 0

print("\n========== HOSPITAL KPIs ==========")
print("Total Patients:", total_patients)
print("Total Admissions:", total_admissions)
print("Total Revenue: ₹", round(total_revenue, 2))
print("Average Bill: ₹", round(average_bill, 2))
print("Average Length of Stay:", round(average_stay, 2), "days")

# -----------------------------
# 5. Patients by Department
# -----------------------------
if "department" in df.columns:

    department_count = df["department"].value_counts()

    print("\nPatients by Department:")
    print(department_count)

    plt.figure(figsize=(10, 6))
    department_count.plot(kind="bar")
    plt.title("Patients by Department")
    plt.xlabel("Department")
    plt.ylabel("Number of Patients")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("../images/patients_by_department.png")
    plt.show()

# -----------------------------
# 6. Top Diseases
# -----------------------------
if "disease" in df.columns:

    disease_count = df["disease"].value_counts().head(10)

    print("\nTop Diseases:")
    print(disease_count)

    plt.figure(figsize=(10, 6))
    disease_count.plot(kind="bar")
    plt.title("Top 10 Diseases")
    plt.xlabel("Disease")
    plt.ylabel("Number of Patients")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("../images/top_diseases.png")
    plt.show()

# -----------------------------
# 7. Age Group Analysis
# -----------------------------
if "age" in df.columns:

    bins = [0, 18, 30, 45, 60, 100]
    labels = [
        "0-18",
        "19-30",
        "31-45",
        "46-60",
        "60+"
    ]

    df["age_group"] = pd.cut(
        df["age"],
        bins=bins,
        labels=labels
    )

    age_group = df["age_group"].value_counts().sort_index()

    print("\nPatients by Age Group:")
    print(age_group)

    plt.figure(figsize=(8, 5))
    age_group.plot(kind="bar")
    plt.title("Patients by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig("../images/age_group_analysis.png")
    plt.show()

# -----------------------------
# 8. Gender Analysis
# -----------------------------
if "gender" in df.columns:

    gender_count = df["gender"].value_counts()

    print("\nGender Distribution:")
    print(gender_count)

    plt.figure(figsize=(7, 5))
    gender_count.plot(kind="pie", autopct="%1.1f%%")
    plt.title("Patient Gender Distribution")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig("../images/gender_distribution.png")
    plt.show()

# -----------------------------
# 9. Monthly Admissions
# -----------------------------
if "admission_date" in df.columns:

    df["month"] = df["admission_date"].dt.to_period("M")

    monthly_admissions = df.groupby("month").size()

    print("\nMonthly Admissions:")
    print(monthly_admissions)

    plt.figure(figsize=(12, 6))
    monthly_admissions.plot(kind="line", marker="o")
    plt.title("Monthly Patient Admissions")
    plt.xlabel("Month")
    plt.ylabel("Admissions")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../images/monthly_admissions.png")
    plt.show()

# -----------------------------
# 10. Department Revenue
# -----------------------------
if "department" in df.columns and "billing_amount" in df.columns:

    department_revenue = (
        df.groupby("department")["billing_amount"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\nRevenue by Department:")
    print(department_revenue)

    plt.figure(figsize=(10, 6))
    department_revenue.plot(kind="bar")
    plt.title("Revenue by Department")
    plt.xlabel("Department")
    plt.ylabel("Revenue")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("../images/revenue_by_department.png")
    plt.show()

# -----------------------------
# 11. Export Clean Dataset
# -----------------------------
df.to_csv(
    "../data/cleaned_hospital_patients.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")

print("\n========== ANALYSIS COMPLETED ==========")
