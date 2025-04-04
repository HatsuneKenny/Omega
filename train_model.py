import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Nacteni datasetu
data_path = "f1_race_results.csv"
df = pd.read_csv(data_path)

# Vyber relevantnich sloupcu
features = ["Season", "Circuit", "Grid Position", "Constructor", "Driver"]
target = "Final Position"
df = df[features + [target]]

# Pretypovani finalni pozice
df[target] = df[target].astype(int)
df = df.dropna()

# Vytvoreni cilove promenne (1 = vyhral, 0 = ne)
df["Winner"] = (df[target] == 1).astype(int)

# Label encodery
encoder_circuit = LabelEncoder()
encoder_constructor = LabelEncoder()
encoder_driver = LabelEncoder()

df["Circuit"] = encoder_circuit.fit_transform(df["Circuit"])
df["Constructor"] = encoder_constructor.fit_transform(df["Constructor"])
df["Driver"] = encoder_driver.fit_transform(df["Driver"])

# Vstupy a vystupy
X = df[features]
y = df["Winner"]

# Rozdeleni na trenovaci a testovaci sadu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trening modelu
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Ulozeni modelu a encoderu
joblib.dump(model, "f1_winner_model.pkl")
joblib.dump(encoder_circuit, "enc_circuit.pkl")
joblib.dump(encoder_constructor, "enc_constructor.pkl")
joblib.dump(encoder_driver, "enc_driver.pkl")

print("Model a encodery byly ulozeny.")