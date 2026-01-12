{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "885ae5af-0e4f-4c20-a1b0-ab317fc0a39d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ preprocessor.pkl saved successfully\n"
     ]
    }
   ],
   "source": [
    "import pickle\n",
    "import pandas as pd\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n",
    "from sklearn.pipeline import Pipeline\n",
    "\n",
    "# ===============================\n",
    "# Feature Lists (MATCH your dataset)\n",
    "# ===============================\n",
    "numeric_features = [\n",
    "    \"temp\",\n",
    "    \"atemp\",\n",
    "    \"humidity\",\n",
    "    \"windspeed\"\n",
    "]\n",
    "\n",
    "categorical_features = [\n",
    "    \"season\",\n",
    "    \"yr\",\n",
    "    \"mnth\",\n",
    "    \"holiday\",\n",
    "    \"weekday\",\n",
    "    \"workingday\",\n",
    "    \"weathersit\"\n",
    "]\n",
    "\n",
    "# ===============================\n",
    "# Pipelines\n",
    "# ===============================\n",
    "numeric_transformer = Pipeline(steps=[\n",
    "    (\"scaler\", StandardScaler())\n",
    "])\n",
    "\n",
    "categorical_transformer = Pipeline(steps=[\n",
    "    (\"onehot\", OneHotEncoder(handle_unknown=\"ignore\"))\n",
    "])\n",
    "\n",
    "# ===============================\n",
    "# ColumnTransformer\n",
    "# ===============================\n",
    "preprocessor = ColumnTransformer(\n",
    "    transformers=[\n",
    "        (\"num\", numeric_transformer, numeric_features),\n",
    "        (\"cat\", categorical_transformer, categorical_features)\n",
    "    ]\n",
    ")\n",
    "\n",
    "# ===============================\n",
    "# Save Preprocessor\n",
    "# ===============================\n",
    "with open(\"preprocessor.pkl\", \"wb\") as f:\n",
    "    pickle.dump(preprocessor, f)\n",
    "\n",
    "print(\"✅ preprocessor.pkl saved successfully\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9255f575-0372-4f71-903e-1bac045a7a6f",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
